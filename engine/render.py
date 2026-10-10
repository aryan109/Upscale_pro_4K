"""Full-length chunked render of the winning strategy (resumable)."""
from __future__ import annotations

import hashlib
import json
import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image

from .common import even, fmt_time, log, run
from .planner import run_engine
from .strategy import Strategy, decode_filters, final_tail, output_fps, post_graph

MP4_AUDIO_OK = {"aac", "mp3", "ac3", "eac3", "opus", "alac", "flac"}


def encoder_args(args) -> list[str]:
    color = ["-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709",
             "-color_range", "tv"]
    if args.codec == "x265":
        return ["-c:v", "libx265", "-preset", args.preset, "-crf", str(args.crf), "-x265-params",
                "aq-mode=3:psy-rd=2.0:psy-rdoq=1.0:no-sao=1:deblock=-1,-1:log-level=error", *color]
    if args.codec == "av1":
        return ["-c:v", "libsvtav1", "-preset", "4", "-crf", str(args.crf + 4),
                "-svtav1-params", "tune=0:enable-overlays=1", *color]
    if args.codec == "amf":
        return ["-c:v", "hevc_amf", "-quality", "quality", "-rc", "cqp", "-qp_i", str(args.crf),
                "-qp_p", str(args.crf + 2), "-profile:v", "main10", "-pix_fmt", "p010le", *color]
    raise ValueError(args.codec)


def mux_final(video: Path, src: Path, out: Path, info: dict, args) -> None:
    is_mp4 = out.suffix.lower() in (".mp4", ".mov", ".m4v")
    trim = (["-ss", str(args.start)] if args.start else []) + \
           (["-t", str(args.duration)] if args.duration else [])
    tag = (["-tag:v", "hvc1" if args.codec != "av1" else "av01", "-movflags", "+faststart"]
           if is_mp4 else [])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), *trim,
           "-i", str(src), "-map", "0:v:0"]
    acodec = []
    for i, a in enumerate(info["audio"]):
        cmd += ["-map", f"1:{a['index']}"]
        if is_mp4 and a.get("codec_name") not in MP4_AUDIO_OK:
            acodec += [f"-c:a:{i}", "aac", f"-b:a:{i}", "320k"]
    for s in info["subs"]:
        if not is_mp4 or s.get("codec_name") == "mov_text":
            cmd += ["-map", f"1:{s['index']}"]
    try:
        run(cmd + ["-c", "copy", *acodec, "-map_metadata", "1", "-map_chapters", "1", *tag, str(out)])
    except RuntimeError as e:
        log(f"Copy-mux failed, retrying with AAC audio ({str(e).splitlines()[-1][:120]})")
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), *trim,
             "-i", str(src), "-map", "0:v:0", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac",
             "-b:a", "320k", "-map_metadata", "1", *tag, str(out)])


def make_compare(src: Path, out: Path, geo: dict, fps, args) -> Path:
    cmp_path = out.with_name(out.stem + "_compare.mp4")
    ow, oh = geo["out"]
    trim = (["-ss", str(args.start)] if args.start else []) + \
           (["-t", str(args.duration)] if args.duration else [])
    font = "fontfile='C\\:/Windows/Fonts/arial.ttf':"
    crop = f"crop={geo['crop'][0]}:{geo['crop'][1]}:{geo['crop'][2]}:{geo['crop'][3]}," if geo["crop"] else ""
    label = (f"drawtext={font}text='Plain resize':x=40:y=40:fontsize=64:fontcolor=white:box=1:"
             f"boxcolor=black@0.5,drawtext={font}text='AI':x={ow // 2 + 40}:y=40:fontsize=64:"
             f"fontcolor=white:box=1:boxcolor=black@0.5,")
    for labels in (label, ""):
        fc = (f"[1:v]{crop}fps={fps.numerator}/{fps.denominator},scale={ow}:{oh}:flags=lanczos,setsar=1,"
              f"crop={ow // 2}:{oh}:0:0[l];[0:v]crop={ow}:{oh},setsar=1,crop={ow // 2}:{oh}:{ow // 2}:0[r];"
              f"[l][r]hstack,drawbox=x={ow // 2 - 2}:y=0:w=4:h={oh}:color=white@0.8:t=fill,"
              f"{labels}format=yuv420p[v]")
        try:
            run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(out), *trim,
                 "-i", str(src), "-filter_complex", fc, "-map", "[v]", "-c:v", "libx264", "-crf", "14",
                 "-preset", "medium", "-shortest", str(cmp_path)])
            break
        except RuntimeError:
            if not labels:
                raise
    return cmp_path


def render(src: Path, out: Path, info: dict, prof: dict, geo: dict, strat: Strategy,
           registry, args) -> Path:
    fps = output_fps(info, prof)
    est_frames = int((args.duration or max(0.0, info["duration"] - (args.start or 0))) * float(fps))
    key = json.dumps([str(src), src.stat().st_size, strat.to_dict(), geo, args.crf, args.codec,
                      args.preset, args.pad, args.start, args.duration, args.chunk,
                      args.no_deinterlace], default=str, sort_keys=True)
    work = out.with_name(out.stem + ".work") / hashlib.sha1(key.encode()).hexdigest()[:10]
    segs_dir = work / "segments"
    segs_dir.mkdir(parents=True, exist_ok=True)

    in_w, in_h = geo["ai_in"]
    ai_w, ai_h = geo["ai_out"] if geo["use_ai"] else geo["ai_in"]
    chunk = args.chunk or max(24, min(600, int(8e9 / (ai_w * ai_h * 3))))
    done = sorted(segs_dir.glob("seg_*.mkv"))

    tw, th = geo["target"]
    if geo["mode"] == "fill":
        cw, ch = even(geo["disp"][0] * geo["scale"]), even(geo["disp"][1] * geo["scale"])
        post_w, post_h, tail = cw, ch, f"crop={tw}:{th}"
    else:
        (post_w, post_h), tail = geo["out"], final_tail(geo, args.pad)

    frame_bytes = in_w * in_h * 3
    dec = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if args.start:
        dec += ["-ss", str(args.start)]
    dec += ["-i", str(src)]
    if args.duration:
        dec += ["-t", str(args.duration)]
    dec += ["-map", "0:v:0", "-vf",
            decode_filters(info, prof, geo, strat, None, args.no_deinterlace),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"]
    decoder = subprocess.Popen(dec, stdout=subprocess.PIPE, bufsize=frame_bytes * 4)

    def read_frame() -> bytes | None:
        buf = decoder.stdout.read(frame_bytes)
        return buf if buf and len(buf) == frame_bytes else None

    skipped = 0
    while skipped < len(done) * chunk and read_frame() is not None:
        skipped += 1
    if done:
        log(f"Resuming: {len(done)} chunks already finished ({skipped} frames).")

    enc_q: queue.Queue = queue.Queue(maxsize=1)
    enc_err: list[Exception] = []

    def encode(idx: int, frames_dir: Path, in_dir: Path) -> None:
        seg = segs_dir / f"seg_{idx:05d}.mkv"
        tmp = seg.with_name("tmp_" + seg.name)
        inputs = ["-framerate", f"{fps.numerator}/{fps.denominator}", "-start_number", "0",
                  "-i", str(frames_dir / "%08d.png")]
        if strat.blend > 0 and geo["use_ai"]:
            inputs += ["-framerate", f"{fps.numerator}/{fps.denominator}", "-start_number", "0",
                       "-i", str(in_dir / "%08d.png")]
        s = strat if geo["use_ai"] else Strategy(**{**strat.to_dict(), "blend": 0})
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *inputs, "-filter_complex",
             post_graph(s, post_w, post_h, tail=tail, seed=idx + 1), "-map", "[v]",
             *encoder_args(args), str(tmp)])
        tmp.replace(seg)

    def worker() -> None:
        while True:
            item = enc_q.get()
            if item is None:
                return
            idx, frames_dir, in_dir, root = item
            try:
                if not enc_err:
                    encode(idx, frames_dir, in_dir)
            except Exception as e:
                enc_err.append(e)
            finally:
                shutil.rmtree(root, ignore_errors=True)

    th_enc = threading.Thread(target=worker, daemon=True)
    th_enc.start()
    idx, frames_done, t0, new = len(done), skipped, time.time(), 0
    try:
        while True:
            if enc_err:
                raise enc_err[0]
            root = work / f"chunk_{idx:05d}"
            in_dir, up_dir = root / "in", root / "up"
            shutil.rmtree(root, ignore_errors=True)
            in_dir.mkdir(parents=True)
            n = 0
            while n < chunk:
                buf = read_frame()
                if buf is None:
                    break
                Image.frombytes("RGB", (in_w, in_h), buf).save(in_dir / f"{n:08d}.png", compress_level=1)
                n += 1
            if n == 0:
                shutil.rmtree(root, ignore_errors=True)
                break
            if geo["use_ai"]:
                run_engine(registry, in_dir, up_dir, strat, args)
                frames_dir = up_dir
            else:
                frames_dir = in_dir
            enc_q.put((idx, frames_dir, in_dir, root))
            idx += 1
            frames_done += n
            new += n
            el = time.time() - t0
            rate = new / el if el else 0
            eta = (est_frames - frames_done) / rate if rate and est_frames else 0
            pct = f"{100 * frames_done / est_frames:5.1f}%" if est_frames else ""
            log(f"chunk {idx:4d}  {frames_done}/{est_frames or '?'} frames {pct}  {rate:.2f} fps  "
                f"elapsed {fmt_time(el)}  ETA {fmt_time(eta)}")
    finally:
        enc_q.put(None)
        th_enc.join()
        decoder.kill()
    if enc_err:
        raise enc_err[0]

    segs = sorted(segs_dir.glob("seg_*.mkv"))
    if not segs:
        raise RuntimeError("No frames were decoded - check --start/--duration.")
    log(f"Joining {len(segs)} segments ...")
    lst = work / "concat.txt"
    lst.write_text("".join(f"file '{s.as_posix()}'\n" for s in segs), encoding="utf-8")
    joined = work / "video.mkv"
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "concat", "-safe", "0",
         "-i", str(lst), "-c", "copy", str(joined)])
    log("Muxing audio / subtitles ...")
    mux_final(joined, src, out, info, args)
    if not args.keep_work:
        parent = work.parent
        shutil.rmtree(work, ignore_errors=True)
        if parent.exists() and not any(parent.iterdir()):
            parent.rmdir()
    return out
