"""Upscale Pro 4K - smart, self-improving local AI video upscaler.

  analyse source -> plan candidate pipelines -> A/B tournament on sample scenes -> pick winner
  -> (Full-HD fallback only if 4K would honestly look worse) -> render -> learn from the result

Commands:
  py -3.11 upscale.py VIDEO [options]          analyse, A/B test, render (default command)
  py -3.11 upscale.py analyze VIDEO            diagnosis + plan only (no AI work)
  py -3.11 upscale.py update                   check for new models / engine now
  py -3.11 upscale.py models                   model registry with status and track record
  py -3.11 upscale.py runs                     past runs
  py -3.11 upscale.py feedback RUN_ID --prefer C2 --rating 4
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

from engine.analyze import analyze, feasibility
from engine.common import RUNS_DIR, fmt_time, load_json, log, save_json
from engine.judge import iqa_available
from engine.knowledge import Knowledge
from engine.planner import est_seconds, finishing, plan_candidates
from engine.probe import probe
from engine.registry import Registry
from engine.render import make_compare, render
from engine.strategy import Strategy, analysis_filters, geometry, output_fps
from engine.tournament import Tournament

COMMANDS = {"run", "analyze", "update", "models", "runs", "feedback"}
FHD = (1920, 1080)
QUALITY_FLOOR = 35.0


def parse_target(s: str) -> tuple[int, int]:
    presets = {"4k": (3840, 2160), "uhd": (3840, 2160), "dci4k": (4096, 2160),
               "1440p": (2560, 1440), "1080p": FHD, "fhd": FHD}
    if s.lower() in presets:
        return presets[s.lower()]
    w, h = s.lower().split("x")
    return int(w), int(h)


def print_profile(info: dict, prof: dict, feas: dict, geo: dict) -> None:
    m = prof["metrics"]
    log(f"Source : {info['width']}x{info['height']} {info['codec']} {float(info['fps']):.3f} fps, "
        f"{fmt_time(info['duration'])}, {info['bitrate'] / 1000:.0f} kb/s")
    log(f"Profile: {prof['category']}  (content={prof['content']} [anime score {prof['anime_score']:.2f}], "
        f"true detail ~{prof['effective_short']:.0f}p, compression={prof['compression']}, "
        f"noise={prof['noise']} sigma {m['sigma']:.1f}, grain={'yes' if prof['grain'] else 'no'}, "
        f"motion={prof['motion']})")
    for f in prof["diagnosis"]["fix"]:
        log(f"  FIX      {f}")
    for p in prof["diagnosis"]["preserve"]:
        log(f"  PRESERVE {p}")
    for n in prof["diagnosis"]["notes"]:
        log(f"  NOTE     {n}")
    log(f"4K feasibility: needs {feas['factor']:.1f}x enlargement of real detail -> risk {feas['risk'].upper()}")
    log(f"Geometry: display {geo['disp'][0]}x{geo['disp'][1]} -> AI input {geo['ai_in'][0]}x{geo['ai_in'][1]} "
        f"(x{geo['prescale']:.2f}) -> 4x AI -> output {geo['out'][0]}x{geo['out'][1]}")


def decide_fallback(prof: dict, feas: dict, tour: dict | None, target) -> list[str]:
    if max(target) < 3000:
        return []
    reasons = []
    eff = prof["effective_short"]
    if feas["risk"] == "extreme":
        reasons.append(f"the source holds only ~{eff:.0f}p of real detail; 4K needs a {feas['factor']:.1f}x "
                       "enlargement, which means the AI would be inventing most of the picture")
    if tour:
        best, fhd = tour["best"]["score"], tour["fhd"]["score"]
        if feas["risk"] in ("high", "extreme") and fhd >= best - 1.0:
            reasons.append(f"in A/B testing a Full-HD master scored {fhd:.1f} vs {best:.1f} for 4K "
                           "(4K adds no visible quality, only risk)")
        if best < QUALITY_FLOOR:
            reasons.append(f"best 4K candidate scored {best:.1f}, below the quality floor ({QUALITY_FLOOR:.0f})")
    return reasons


def ask_fallback(reasons: list[str], policy: str) -> str:
    """Returns 'fhd', '4k' or 'stop'."""
    print("\n" + "=" * 78)
    print("RECOMMENDATION: master in FULL HD (1080p) instead of 4K")
    for r in reasons:
        print(f"  - {r}")
    print("A 1080p master shown on a 4K screen will look cleaner than a 4K master full of invented detail.")
    print("=" * 78)
    if policy == "auto":
        return "fhd"
    if policy == "never":
        print("--fallback never: continuing with 4K as requested.")
        return "4k"
    if sys.stdin.isatty():
        try:
            while True:
                a = input("Choose: [F]ull HD (recommended) / [4]K anyway / [C]ancel > ").strip().lower()
                if a in ("f", "fhd", ""):
                    return "fhd"
                if a in ("4", "4k"):
                    return "4k"
                if a in ("c", "cancel"):
                    return "stop"
        except EOFError:
            print()
    print("Not running interactively - stopping before the long render so you can decide.")
    print("Re-run with  --fallback auto  (Full HD)  or  --fallback never  (4K anyway).")
    return "stop"


def write_report(run: dict, path: Path) -> None:
    p, t = run["profile"], run.get("tournament")
    L = [f"# Upscale run {run['id']}", "", f"- Source: `{run['source']}`",
         f"- Category: `{p['category']}`", f"- Decision: **{run.get('decision', '?')}**", ""]
    L += ["## Diagnosis", ""] + [f"- Fix: {x}" for x in p["diagnosis"]["fix"]] + \
         [f"- Preserve: {x}" for x in p["diagnosis"]["preserve"]] + \
         [f"- Note: {x}" for x in p["diagnosis"]["notes"]] + [""]
    L += [f"4K feasibility: {run['feasibility']['factor']:.1f}x of real detail, risk "
          f"**{run['feasibility']['risk']}**", ""]
    if t:
        L += ["## A/B tournament", "", "| id | score | strategy | fidelity | detail | texture | flicker | ringing |",
              "|---|---|---|---|---|---|---|---|"]
        for c in sorted(t["candidates"], key=lambda c: -c["score"]):
            r = c["raw"]
            L.append(f"| {c['id']} | {c['score']:.1f} | {c['label']} | {r['fidelity']:.3f} | "
                     f"x{r['detail_ratio']:.2f} | {r['texture_ratio']:.2f} | {r['flicker_ratio']:.2f} | "
                     f"{r['ringing']:.3f} |")
        L += ["", f"Full-HD check: {t['fhd']['score']:.1f}", "",
              f"Winner: **{t['best']['id']} {t['best']['label']}**", ""]
        L += [f"- Finishing: {n}" for n in t["finishing"]]
        L += ["", "Contact sheet: `tournament_sheet.png`"]
    if run.get("events"):
        L += ["", "## Learning / self-update"] + [f"- {e}" for e in run["events"]]
    L += ["", f"Feedback: `py -3.11 upscale.py feedback {run['id']} --prefer <id> --rating 1-5`"]
    path.write_text("\n".join(L), encoding="utf-8")


def cmd_run(args, plan_only=False) -> None:
    src = Path(args.input).resolve()
    if not src.exists():
        sys.exit(f"Input not found: {src}")
    t_start = time.time()
    registry, kb = Registry(), Knowledge()
    events = []
    if not args.offline:
        events += registry.check_updates(force=args.update, allow_engine=not args.no_engine_update)

    info = probe(src)
    prof = analyze(src, info, analysis_filters)
    target = args.target
    feas = feasibility(prof, min(target))
    geo = geometry(info, prof, target, args.mode, 4, not args.no_effective)
    print_profile(info, prof, feas, geo)

    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + re.sub(r"[^A-Za-z0-9]+", "-", src.stem)[:30]
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    fps = output_fps(info, prof)
    frames = int((args.duration or max(0.0, info["duration"] - args.start)) * float(fps))
    cands = plan_candidates(prof, registry, kb, geo, frames, args)
    log("Planned candidates:")
    for i, c in enumerate(cands):
        log(f"  {i + 1}. {c['strategy'].label():40s} prior {c['prior']:5.1f}  "
            f"[{c['status']}] full-length est {fmt_time(c['est'])}  ({'; '.join(c['why'])})")
    if args.command == "analyze":
        return

    run = {"id": run_id, "source": str(src), "profile": prof, "feasibility": feas,
           "category": prof["category"], "info": {k: str(v) for k, v in info.items()
                                                    if k not in ("audio", "subs")}}
    tour = None
    if args.quick or len(cands) == 1:
        base = cands[0]["strategy"]
        final, notes = finishing(base, prof, {}, geo["scale"])
        run["finishing"] = notes
    else:
        if iqa_available():
            log("Learned perceptual metric (TOPIQ-NR) enabled in the judge.")
        tournament = Tournament(src, info, prof, registry, kb, args, run_dir / "tournament")
        tour = tournament.run(cands)
        final = tour["final"]
        best = tour["best"]
        log(f"WINNER {best['id']} ({best['score']:.1f}): {best['label']}")
        for n in tour["finishing"]:
            log(f"  + {n}")
        res = [{"key": c["key"], "model": c["model"], "score": c["score"]} for c in tour["candidates"]]
        events += kb.record(prof["category"], res, registry)
        run["tournament"] = {k: v for k, v in tour.items() if k != "final"}
        run["candidates"] = tour["candidates"]
        run["chosen_key"] = best["key"]
        if not args.keep_work:
            import shutil
            shutil.rmtree(run_dir / "tournament", ignore_errors=True)

    # ---- 4K or Full HD?
    decision = "4k" if max(target) >= 3000 else f"{target[0]}x{target[1]}"
    reasons = decide_fallback(prof, feas, tour, target)
    if reasons:
        choice = ask_fallback(reasons, args.fallback)
        if choice == "stop":
            decision = "stopped: Full HD recommended"
        elif choice == "fhd":
            target, decision = FHD, "fullhd (fallback)"
            geo = geometry(info, prof, target, args.mode, 4, not args.no_effective)
    run.update(decision=decision, final_strategy=final.to_dict(), events=events,
               fallback_reasons=reasons)
    save_json(run_dir / "report.json", run)
    write_report(run, run_dir / "report.md")
    log(f"Run report: {run_dir / 'report.md'}")
    if plan_only or decision.startswith("stopped"):
        return

    suffix = "_FHD" if decision.startswith("fullhd") else "_4K"
    out = Path(args.output).resolve() if args.output else src.with_name(f"{src.stem}{suffix}.mp4")
    est = est_seconds(registry.data["models"].get(final.model, {}), final.model,
                      geo["ai_in"][0] * geo["ai_in"][1] / 1e6, frames, kb)
    log(f"Rendering with: {final.label()}  -> {geo['out'][0]}x{geo['out'][1]}  (estimated {fmt_time(est)})")
    log(f"Output: {out}")
    t_r = time.time()
    render(src, out, info, prof, geo, final, registry, args)
    if args.compare:
        log(f"Comparison video: {make_compare(src, out, geo, fps, args)}")
    run.update(output=str(out), render_seconds=time.time() - t_r, total_seconds=time.time() - t_start)
    save_json(run_dir / "report.json", run)
    write_report(run, run_dir / "report.md")
    log(f"Done in {fmt_time(time.time() - t_start)} -> {out}")


def cmd_models(args) -> None:
    reg, kb = Registry(), Knowledge()
    print(f"Engine: {reg.engine} ({reg.engine_kind} {reg.data['engine'].get('tag')})")
    print(f"{'model':34s} {'status':8s} {'domain':6s} {'speed':5s} {'licence':16s} {'trials':>6s} "
          f"{'wins':>4s} {'s/MP':>6s}")
    for mid, m in sorted(reg.data["models"].items(), key=lambda kv: (kv[1].get("status"), kv[0])):
        st = kb.d["models"].get(mid, {})
        spd = kb.d.get("speed", {}).get(mid)
        print(f"{mid:34s} {m.get('status', ''):8s} {m.get('domain', ''):6s} {m.get('speed', ''):5s} "
              f"{m.get('license', ''):16s} {st.get('trials', 0):6d} {st.get('wins', 0):4d} "
              f"{spd if spd else '-':>6}")
    if reg.data["watchlist"]:
        print("\nResearch watchlist (newly published, not yet runnable on Vulkan):")
        for w in reg.data["watchlist"][-15:]:
            print(f"  {w['found']}  {w['name']} ({w.get('arch')}, x{w.get('scale')}) {w.get('license')}")


def cmd_runs(args) -> None:
    for d in sorted(RUNS_DIR.iterdir()):
        r = load_json(d / "report.json", None)
        if r:
            best = r.get("tournament", {}).get("best", {})
            print(f"{r['id']:45s} {r['category']:28s} {r.get('decision', ''):22s} "
                  f"{best.get('label', r.get('final_strategy', {}).get('model', ''))}")


def cmd_feedback(args) -> None:
    path = RUNS_DIR / args.run_id / "report.json"
    run = load_json(path, None)
    if not run:
        sys.exit(f"Run not found: {args.run_id}")
    kb = Knowledge()
    for m in kb.feedback(run, args.prefer, args.rating):
        log(m)


def main() -> None:
    argv = sys.argv[1:]
    if argv and argv[0] not in COMMANDS and not argv[0].startswith("-"):
        argv = ["run"] + argv
    ap = argparse.ArgumentParser(description="Smart self-improving AI video upscaler (local, free)")
    sub = ap.add_subparsers(dest="command", required=True)

    def media_args(p):
        p.add_argument("input")
        p.add_argument("-o", "--output")
        p.add_argument("--target", type=parse_target, default=(3840, 2160),
                       help="4k (default), dci4k, 1440p, 1080p or WxH")
        p.add_argument("--mode", choices=["fit", "fill"], default="fit",
                       help="fit keeps aspect (default); fill crops to cover the frame")
        p.add_argument("--pad", action="store_true", help="letterbox to exactly the target size")
        p.add_argument("--fallback", choices=["ask", "auto", "never"], default="ask",
                       help="what to do when Full HD would look better than 4K")
        p.add_argument("--candidates", type=int, default=3, help="incumbent models to A/B test")
        p.add_argument("--trials", type=int, default=1, help="new/trial models to challenge them")
        p.add_argument("--samples", type=int, default=4, help="sample scenes in the tournament")
        p.add_argument("--model", help="force one model (skips the planner's choice)")
        p.add_argument("--allow-nc", action="store_true", help="allow non-commercial-licence models")
        p.add_argument("--tta", action="store_true", help="test-time augmentation (~8x slower)")
        p.add_argument("--quick", action="store_true", help="skip the A/B tournament")
        p.add_argument("--plan-only", action="store_true", help="analyse + tournament, no render")
        p.add_argument("--no-effective", action="store_true",
                       help="don't shrink 'fake HD' sources to their true detail level first")
        p.add_argument("--max-days", type=float, default=0, help="time budget for the render")
        p.add_argument("--codec", choices=["x265", "av1", "amf"], default="x265")
        p.add_argument("--crf", type=int, default=16)
        p.add_argument("--preset", default="slow")
        p.add_argument("--start", type=float, default=0.0)
        p.add_argument("--duration", type=float, default=0.0)
        p.add_argument("--compare", action="store_true", help="write a plain-vs-AI split-screen")
        p.add_argument("--no-deinterlace", action="store_true")
        p.add_argument("--gpu", type=int, default=0)
        p.add_argument("--tile", type=int, default=0)
        p.add_argument("--chunk", type=int, default=0)
        p.add_argument("--keep-work", action="store_true")
        p.add_argument("--offline", action="store_true", help="skip the new-tech check")
        p.add_argument("--update", action="store_true", help="force the new-tech check now")
        p.add_argument("--no-engine-update", action="store_true",
                       help="never auto-adopt a new engine release")

    media_args(sub.add_parser("run"))
    media_args(sub.add_parser("analyze"))
    sub.add_parser("update")
    sub.add_parser("models")
    sub.add_parser("runs")
    fb = sub.add_parser("feedback")
    fb.add_argument("run_id")
    fb.add_argument("--prefer", help="candidate id you think looks best (e.g. C2 or C1.3)")
    fb.add_argument("--rating", type=int, choices=range(1, 6), help="how good the final output is")
    args = ap.parse_args(argv)

    if args.command in ("run", "analyze"):
        cmd_run(args, plan_only=getattr(args, "plan_only", False))
    elif args.command == "update":
        Registry().check_updates(force=True)
    elif args.command == "models":
        cmd_models(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "feedback":
        cmd_feedback(args)


if __name__ == "__main__":
    main()
