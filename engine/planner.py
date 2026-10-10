"""Turn a source profile + learned history into a shortlist of candidate strategies."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from .common import MODELS_DIR
from .strategy import Strategy

# seconds per megapixel of AI input on the reference GPU (RX 6700 XT); refined from measurements
SPEED_PRIOR = {"fast": 0.45, "slow": 5.6}
STRONG_DENOISERS = {"realesrgan-x4plus", "RealESRGAN_General_x4_v3"}


def rule_score(meta: dict, mid: str, prof: dict) -> tuple[float, list[str]]:
    s, why = 50.0, []
    h = set(meta.get("handles", []))
    comp, noise = prof["compression"], prof["noise"]
    if comp in ("heavy", "moderate"):
        if "jpeg" in h:
            s += 15 if comp == "heavy" else 9
            why.append("trained on compression")
        elif "clean" in h:
            s -= 12
            why.append("expects clean input")
    else:
        if "clean" in h:
            s += 10
            why.append("clean-source specialist")
    if noise in ("medium", "high") and not prof["grain"]:
        if "noise" in h:
            s += 10
            why.append("denoises")
    if prof["metrics"]["eff"] >= 1.5 or (prof["scan"].get("ff_blur") or 0) > 8:
        if "blur" in h:
            s += 8
            why.append("handles blur")
    keep_texture = prof["grain"] or prof["metrics"]["texture"] > 0.25
    if keep_texture and prof["content"] != "anime":
        if mid in STRONG_DENOISERS:
            s -= 6
            why.append("may smooth texture")
        if "WDN" in mid:
            s += 4
            why.append("weak-denoise keeps texture")
    if prof.get("small_faces") and meta.get("speed") == "slow" and "clean" in h:
        s -= 4  # big GAN on tiny faces = invented features
    return s, why


def est_seconds(meta: dict, mid: str, mp_per_frame: float, frames: int, kb) -> float:
    spd = kb.d.get("speed", {}).get(mid) or SPEED_PRIOR.get(meta.get("speed", "slow"), 5.6)
    return spd * mp_per_frame * frames


def plan_candidates(prof: dict, registry, kb, geo: dict, frames: int, args) -> list[dict]:
    """Return stage-1 candidates: [{strategy, why, est}] (AI-affecting variants only)."""
    domains = {"live": ["live"], "anime": ["anime"], "mixed": ["live", "anime"]}[prof["content"]]
    if args.model:
        domains = ["live", "anime"]
    models = registry.models(allow_nc=args.allow_nc)
    mp = geo["ai_in"][0] * geo["ai_in"][1] / 1e6
    budget = args.max_days * 86400 if args.max_days else None

    scored = []
    for mid, meta in models.items():
        if meta.get("domain") not in domains + ["any"] or meta.get("scale", 4) != 4:
            continue
        if args.model and mid != args.model:
            continue
        s, why = rule_score(meta, mid, prof)
        kb_mean, n = kb.model_prior(prof["category"], mid)
        if n:
            # Rules are a guess; measured results are evidence. Put the rule score on the judge's
            # scale, then let evidence dominate as it accumulates (1 run: 50/50, 3 runs: ~86%).
            a = min(0.9, n / (n + 0.5)) if n > 1 else 0.5
            s = (1 - a) * (60 + (s - 50) * 0.8) + a * kb_mean
            why.append(f"history {kb_mean:.0f} pts over {n} runs")
        est = est_seconds(meta, mid, mp, frames, kb)
        if budget and est > budget:
            s -= 30
            why.append(f"too slow for budget ({est / 3600:.0f} h)")
        scored.append((s, mid, meta, why, est))
    scored.sort(key=lambda x: -x[0])

    active = [x for x in scored if x[2].get("status") == "active"]
    trials = [x for x in scored if x[2].get("status") == "trial"]
    per_domain = max(2, args.candidates // len(domains)) if len(domains) > 1 else args.candidates
    picked = []
    for d in domains:
        picked += [x for x in active if x[2]["domain"] == d][:per_domain]
    # champion from history always defends its title
    champ = kb.champion(prof["category"])
    if champ:
        cm = champ.split("|")[0]
        if cm in models and all(p[1] != cm for p in picked):
            row = next((x for x in scored if x[1] == cm), None)
            if row:
                picked.append(row)
    # new tech: best-matching trial models challenge the incumbents
    picked += trials[:args.trials]

    cands = []
    for s, mid, meta, why, est in picked:
        cands.append({"strategy": Strategy(model=mid, tta=args.tta), "prior": s, "why": why,
                      "est": est, "status": meta.get("status")})

    # pre-processing variants for the top model when the source needs them
    if cands:
        top = cands[0]
        if prof["compression"] == "heavy" or prof["metrics"]["block"] > 1.15:
            v = Strategy(model=top["strategy"].model, deblock="strong", tta=args.tta)
            cands.append({"strategy": v, "prior": top["prior"] - 1, "why": ["deblock first"],
                          "est": top["est"], "status": "variant"})
        if prof["noise"] in ("medium", "high") and not prof["grain"]:
            v = Strategy(model=top["strategy"].model, denoise="nlm", tta=args.tta)
            cands.append({"strategy": v, "prior": top["prior"] - 1, "why": ["NL-means denoise first"],
                          "est": top["est"], "status": "variant"})
    # de-duplicate, cap
    seen, out = set(), []
    for c in cands:
        k = c["strategy"].ai_key()
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out[: args.candidates + args.trials + 2]


def post_variants(base: Strategy, prof: dict, raw: dict) -> list[Strategy]:
    """Stage-2 finishing variants (cheap: re-uses the AI frames)."""
    out = []
    blends = [0.0, 0.15, 0.3] if prof["content"] != "anime" else [0.0, 0.1]
    sharps = [0.0, 0.3] if raw.get("detail_ratio", 2) < 1.6 else [0.0]
    for b in blends:
        for sh in sharps:
            out.append(Strategy(**{**base.to_dict(), "blend": b, "sharpen": sh}))
    return out


def finishing(strat: Strategy, prof: dict, raw: dict, out_scale: float) -> tuple[Strategy, list[str]]:
    """Rule-based 'preserve' steps applied to the winner."""
    notes = []
    s = Strategy(**strat.to_dict())
    tsig = prof["metrics"]["temporal_sigma"]
    if prof["grain"] and tsig == tsig:
        s.grain = round(min(4.0, max(0.8, tsig * 0.9)), 1)
        notes.append(f"re-grain sigma {s.grain} to preserve the source's film grain")
    elif prof["content"] != "anime" and raw.get("texture_ratio", 1) == raw.get("texture_ratio", 1) \
            and raw.get("texture_ratio", 1) < 0.6:
        s.grain = 0.8
        notes.append("light grain to mask AI smoothing ('plastic' look)")
    if prof["banding"] or prof["dark"]:
        s.deband = True
        notes.append("deband (gradients/dark scenes) + 10-bit output")
    fr = raw.get("flicker_ratio", 1)
    if fr == fr and fr > 1.15:
        s.deflicker = True
        notes.append(f"temporal deflicker (AI shimmer {fr:.2f}x source)")
    return s, notes


def run_engine(registry, in_dir: Path, out_dir: Path, strat: Strategy, args) -> float:
    """Run the Vulkan AI engine over a folder. Returns seconds taken."""
    out_dir.mkdir(parents=True, exist_ok=True)
    exe = registry.engine
    cmd = [str(exe), "-i", str(in_dir), "-o", str(out_dir), "-m", str(MODELS_DIR),
           "-n", strat.model, "-s", str(strat.model_scale), "-g", str(args.gpu),
           "-j", "2:2:2", "-f", "png", "-t", str(args.tile)]
    if registry.engine_kind == "upscayl":
        cmd += ["-z", str(strat.model_scale)]
    if strat.tta:
        cmd.append("-x")
    t = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    n_in = len(list(in_dir.glob("*.png")))
    n_out = len(list(out_dir.glob("*.png")))
    if res.returncode != 0 or n_out != n_in:
        raise RuntimeError(f"AI engine failed for {strat.model} ({n_out}/{n_in}):\n{res.stderr[-1500:]}")
    return time.time() - t


def mp_seconds(sec: float, frames: int, w: int, h: int) -> float:
    return sec / max(1e-6, frames * w * h / 1e6)

