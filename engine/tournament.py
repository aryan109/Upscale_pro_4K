"""A/B tournament: render every candidate on the same sample crops, judge, pick a winner."""
from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np

from . import judge
from .analyze import grab_frames, luma
from .common import log, run
from .planner import finishing, mp_seconds, post_variants, run_engine
from .strategy import Strategy, decode_filters, geometry, post_graph

WARMUP, KEEP = 2, 3          # frames decoded per sample point (warm-up for temporal filters)
CROP_MAX = 288               # max crop side at AI-input resolution (keeps tournaments fast)


def pick_samples(prof: dict, k: int) -> list[float]:
    s = sorted(prof["samples"], key=lambda p: -(p["texture"] + p["edge_density"] + 0.02 * p["sigma"]))
    chosen = s[:max(1, k - 1)]
    dark = [p for p in prof["samples"] if p["mean_luma"] < 50 and p not in chosen]
    if dark:
        chosen.append(dark[0])
    elif len(s) > len(chosen):
        chosen.append(s[len(chosen)])
    return sorted(p["t"] for p in chosen[:k])


def pick_crop(frame: np.ndarray) -> tuple[int, int, int, int]:
    h, w = frame.shape[:2]
    cw, ch = min(w, CROP_MAX) // 2 * 2, min(h, CROP_MAX) // 2 * 2
    if (cw, ch) == (w, h):
        return (cw, ch, 0, 0)
    y = luma(frame)
    m = cv2.blur(y, (9, 9))
    std = np.sqrt(np.maximum(cv2.blur(y * y, (9, 9)) - m * m, 0))
    integ = cv2.integral(std)
    best, pos = -1, (0, 0)
    for yy in range(0, h - ch + 1, max(8, ch // 6)):
        for xx in range(0, w - cw + 1, max(8, cw // 6)):
            v = integ[yy + ch, xx + cw] - integ[yy, xx + cw] - integ[yy + ch, xx] + integ[yy, xx]
            if v > best:
                best, pos = v, (xx, yy)
    return (cw, ch, pos[0] // 2 * 2, pos[1] // 2 * 2)


def _imread(p: Path) -> np.ndarray:
    return cv2.cvtColor(cv2.imread(str(p), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


class Tournament:
    def __init__(self, src: Path, info: dict, prof: dict, registry, kb, args, workdir: Path):
        self.src, self.info, self.prof = src, info, prof
        self.registry, self.kb, self.args = registry, kb, args
        self.work = workdir
        shutil.rmtree(self.work, ignore_errors=True)
        self.work.mkdir(parents=True)
        self.times = pick_samples(prof, args.samples)
        self.geo = geometry(info, prof, args.target, args.mode, 4, not args.no_effective)
        self.ref: list[list[np.ndarray]] = []
        self.crops: list[tuple] = []
        self._prepare_reference()

    # ------------------------------------------------------------------ inputs
    def _frames(self, strat: Strategy | None, t: float, crop=None) -> list[np.ndarray]:
        vf = decode_filters(self.info, self.prof, self.geo, strat, crop, self.args.no_deinterlace)
        w, h = (crop[0], crop[1]) if crop else self.geo["ai_in"]
        t0 = max(0.0, t - WARMUP / float(self.info["fps"]))
        fr = grab_frames(self.src, t0, vf, w, h, WARMUP + KEEP)
        return fr[-KEEP:] if len(fr) > KEEP else fr

    def _prepare_reference(self) -> None:
        for t in self.times:
            full = self._frames(None, t)
            if not full:
                continue
            crop = pick_crop(full[0])
            cw, ch, cx, cy = crop
            self.crops.append(crop)
            self.ref.append([f[cy:cy + ch, cx:cx + cw].copy() for f in full])
        self.times = self.times[:len(self.ref)]
        if not self.ref:
            raise RuntimeError("Tournament: could not decode sample frames.")

    def _write_inputs(self, strat: Strategy, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for i, (t, crop) in enumerate(zip(self.times, self.crops)):
            fr = self._frames(strat, t, crop)
            for j, f in enumerate(fr):
                cv2.imwrite(str(d / f"s{i}_{j}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))

    def out_size(self, crop) -> tuple[int, int]:
        k = self.geo["out"][0] / self.geo["ai_in"][0]
        return int(round(crop[0] * k / 2) * 2), int(round(crop[1] * k / 2) * 2)

    def _post(self, strat: Strategy, ai_dir: Path, up_dir: Path, out_dir: Path) -> list[list[np.ndarray]]:
        out_dir.mkdir(parents=True, exist_ok=True)
        seqs = []
        for i, crop in enumerate(self.crops):
            w, h = self.out_size(crop)
            inputs = ["-start_number", "0", "-i", str(up_dir / f"s{i}_%d.png")]
            if strat.blend > 0:
                inputs += ["-start_number", "0", "-i", str(ai_dir / f"s{i}_%d.png")]
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
                 post_graph(strat, w, h, final="rgb24"), "-map", "[v]", "-start_number", "0",
                 str(out_dir / f"s{i}_%d.png")])
            seqs.append([_imread(out_dir / f"s{i}_{j}.png") for j in range(len(self.ref[i]))])
        return seqs

    def _judge(self, seqs) -> tuple[dict, dict]:
        sig = self.prof["metrics"]["sigma"]
        raws = [judge.raw_metrics(o, r, sig) for o, r in zip(seqs, self.ref)]
        raw = {k: float(np.nanmean([r.get(k, np.nan) for r in raws])) for k in raws[0]}
        return raw, judge.normalise(raw)

    # ------------------------------------------------------------------ run
    def run(self, candidates: list[dict]) -> dict:
        w = self.kb.weights
        results, ai_dirs = [], {}
        log(f"Tournament: {len(candidates)} AI candidates x {len(self.times)} sample scenes "
            f"(crops {self.crops[0][0]}x{self.crops[0][1]} -> {self.out_size(self.crops[0])})")
        for n, c in enumerate(candidates):
            st: Strategy = c["strategy"]
            if not self.registry.ensure(st.model):
                log(f"  [{n + 1}] {st.label()}: model unavailable, skipped")
                continue
            tag = f"c{n}"
            ai_dir, up_dir = self.work / tag / "in", self.work / tag / "up"
            try:
                self._write_inputs(st, ai_dir)
                sec = run_engine(self.registry, ai_dir, up_dir, st, self.args)
                seqs = self._post(Strategy(**{**st.to_dict(), "blend": 0}), ai_dir, up_dir,
                                  self.work / tag / "post0")
            except Exception as e:
                log(f"  [{n + 1}] {st.label()}: FAILED ({str(e).splitlines()[0][:100]})")
                continue
            nfr = sum(len(r) for r in self.ref)
            cw, ch = self.crops[0][0], self.crops[0][1]
            spd = mp_seconds(sec, nfr, cw, ch)
            self.kb.d.setdefault("speed", {})[st.model] = round(
                0.7 * self.kb.d.get("speed", {}).get(st.model, spd) + 0.3 * spd, 3)
            raw, comp = self._judge(seqs)
            sc = judge.score(comp, w)
            ai_dirs[st.ai_key()] = (ai_dir, up_dir)
            results.append({"id": f"C{n + 1}", "label": st.label(), "key": st.key(),
                            "model": st.model, "strategy": st.to_dict(), "raw": raw,
                            "components": comp, "score": sc, "stage": 1, "status": c.get("status"),
                            "why": c.get("why", []), "sec_per_mp": spd, "seqs": seqs})
            log(f"  [C{n + 1}] {sc:5.1f}  {st.label():45s} fid {raw['fidelity']:.3f} "
                f"detail x{raw['detail_ratio']:.2f} texture {raw['texture_ratio']:.2f} "
                f"flicker {raw['flicker_ratio']:.2f} ring {raw['ringing']:.3f}  "
                f"({spd:.2f} s/MP{', ' + c['status'] if c.get('status') != 'active' else ''})")
        if not results:
            raise RuntimeError("No candidate could be rendered.")

        # ---- stage 2: finishing variants on the two best AI outputs
        top = sorted(results, key=lambda r: -r["score"])[:2]
        stage2 = []
        for r in top:
            base = Strategy.from_dict(r["strategy"])
            ai_dir, up_dir = ai_dirs[base.ai_key()]
            for k, v in enumerate(post_variants(base, self.prof, r["raw"])):
                if v.blend == 0 and v.sharpen == 0:
                    continue
                seqs = self._post(v, ai_dir, up_dir, self.work / f"{r['id']}_p{k}")
                raw, comp = self._judge(seqs)
                sc = judge.score(comp, w)
                stage2.append({"id": f"{r['id']}.{k}", "label": v.label(), "key": v.key(),
                               "model": v.model, "strategy": v.to_dict(), "raw": raw,
                               "components": comp, "score": sc, "stage": 2, "seqs": seqs})
                log(f"  [{r['id']}.{k}] {sc:5.1f}  {v.label():45s} texture {raw['texture_ratio']:.2f} "
                    f"detail x{raw['detail_ratio']:.2f} fid {raw['fidelity']:.3f}")
        allr = results + stage2
        best = max(allr, key=lambda r: r["score"])

        # ---- Full-HD sanity check: would a 1080p master shown on a 4K screen look as good?
        fhd = self._fhd_check(best)

        # ---- finishing (rule-based preserve steps)
        final, notes = finishing(Strategy.from_dict(best["strategy"]), self.prof, best["raw"],
                                 self.geo["scale"])
        self._sheet(results, best, final)
        for r in allr:
            r.pop("seqs", None)
        return {"candidates": allr, "best": best, "final": final, "finishing": notes,
                "fhd": fhd, "times": self.times, "crops": self.crops, "geo": self.geo}

    def _fhd_check(self, best: dict) -> dict:
        tw, th = self.geo["target"]
        k = min(1920 / tw, 1080 / th) if tw >= th else min(1080 / tw, 1920 / th)
        seqs = []
        for seq in best["seqs"]:
            h, w = seq[0].shape[:2]
            seqs.append([cv2.resize(cv2.resize(f, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA),
                                    (w, h), interpolation=cv2.INTER_LANCZOS4) for f in seq])
        raw, comp = self._judge(seqs)
        sc = judge.score(comp, self.kb.weights)
        log(f"  [FHD] {sc:5.1f}  best candidate mastered at Full HD, shown on a 4K screen")
        return {"score": sc, "raw": raw, "components": comp}

    def _sheet(self, results: list[dict], best: dict, final: Strategy) -> None:
        """Contact sheet of sample 0: Lanczos vs each stage-1 candidate vs winner."""
        try:
            ref = self.ref[0][0]
            w, h = self.out_size(self.crops[0])
            tiles = [("Lanczos (plain resize)", cv2.resize(ref, (w, h), interpolation=cv2.INTER_LANCZOS4))]
            for r in sorted(results, key=lambda r: -r["score"]):
                tiles.append((f"{r['id']} {r['score']:.0f}: {r['model']}", r["seqs"][0][0]))
            if best["stage"] == 2:
                tiles.append((f"WINNER {best['id']} {best['score']:.0f}: {best['label']}", best["seqs"][0][0]))
            scale = min(1.0, 640 / w)
            tw, th = int(w * scale), int(h * scale)
            cols = 3
            rows = (len(tiles) + cols - 1) // cols
            sheet = np.full((rows * (th + 34), cols * tw, 3), 20, np.uint8)
            for i, (name, img) in enumerate(tiles):
                r, c = divmod(i, cols)
                y0, x0 = r * (th + 34), c * tw
                sheet[y0 + 34:y0 + 34 + th, x0:x0 + tw] = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
                cv2.putText(sheet, name[:60], (x0 + 6, y0 + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                            (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imwrite(str(self.work.parent / "tournament_sheet.png"), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))
        except Exception as e:
            log(f"  (contact sheet skipped: {e})")

