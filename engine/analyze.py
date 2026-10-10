"""Source analysis: measure what the video needs fixed and what must be preserved."""
from __future__ import annotations

import math
import re
import subprocess
from pathlib import Path

import cv2
import numpy as np

from .common import log

CODEC_EFFICIENCY = {  # bits needed relative to H.264 for similar quality
    "mpeg2video": 2.0, "mpeg1video": 2.4, "mpeg4": 1.4, "msmpeg4v3": 1.6, "wmv3": 1.4,
    "h263": 2.2, "vc1": 1.2, "h264": 1.0, "vp8": 1.1, "hevc": 0.65, "vp9": 0.7, "av1": 0.55,
    "prores": 5.0, "dnxhd": 5.0, "ffv1": 8.0, "rawvideo": 20.0,
}


# --------------------------------------------------------------------------- ffmpeg pass

def ffmpeg_scan(src: Path, info: dict) -> dict:
    """One decode pass (<= 40 s of video) through FFmpeg's detection filters."""
    dur = info["duration"] or 0
    ss = max(0.0, dur * 0.1) if dur > 20 else 0.0
    cmd = ["ffmpeg", "-hide_banner", "-nostats", "-ss", f"{ss:.2f}", "-i", str(src), "-t", "40",
           "-map", "0:v:0", "-vf",
           "idet,cropdetect=round=2:reset=0:limit=24,blockdetect,blurdetect", "-an", "-f", "null", "-"]
    err = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace").stderr
    out: dict = {}
    m = re.search(r"Multi frame detection: TFF:\s*(\d+) BFF:\s*(\d+) Progressive:\s*(\d+) "
                  r"Undetermined:\s*(\d+)", err)
    if m:
        tff, bff, prog, und = map(int, m.groups())
        total = max(1, tff + bff + prog)
        out["interlaced_ratio"] = (tff + bff) / total
        out["field_order"] = "tff" if tff >= bff else "bff"
    m = re.search(r"Repeated Fields: Neither:\s*(\d+) Top:\s*(\d+) Bottom:\s*(\d+)", err)
    if m:
        nei, top, bot = map(int, m.groups())
        out["repeated_ratio"] = (top + bot) / max(1, nei + top + bot)
    crops = re.findall(r"crop=(\d+):(\d+):(\d+):(\d+)", err)
    if crops:
        out["crop"] = tuple(map(int, crops[-1]))
    m = re.search(r"block mean:\s*([\d.]+)", err)
    out["ff_block"] = float(m.group(1)) if m else None
    m = re.search(r"blur mean:\s*([\d.]+)", err)
    out["ff_blur"] = float(m.group(1)) if m else None
    return out


# --------------------------------------------------------------------------- frame sampling

def grab_frames(src: Path, t: float, vf: str, w: int, h: int, n: int = 3) -> list[np.ndarray]:
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.3f}", "-i", str(src),
           "-map", "0:v:0", "-frames:v", str(n), "-vf", vf, "-f", "rawvideo", "-pix_fmt", "rgb24",
           "pipe:1"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fb = w * h * 3
    return [np.frombuffer(raw[i * fb:(i + 1) * fb], np.uint8).reshape(h, w, 3)
            for i in range(len(raw) // fb)]


def luma(rgb: np.ndarray) -> np.ndarray:
    r, g, b = [rgb[..., i].astype(np.float32) for i in range(3)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


# --------------------------------------------------------------------------- metrics

def noise_sigma(y: np.ndarray) -> float:
    """Immerkaer noise estimate restricted to low-gradient (non-texture) pixels."""
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    conv = np.abs(cv2.filter2D(y, -1, k))[1:-1, 1:-1]
    gx, gy = cv2.Sobel(y, cv2.CV_32F, 1, 0), cv2.Sobel(y, cv2.CV_32F, 0, 1)
    grad = np.hypot(gx, gy)[1:-1, 1:-1]
    mask = grad < np.percentile(grad, 40)
    if mask.sum() < 1000:
        mask = np.ones_like(grad, bool)
    return float(math.sqrt(math.pi / 2) * conv[mask].mean() / 6)


def temporal_noise(y0: np.ndarray, y1: np.ndarray) -> tuple[float, float]:
    d = y1 - y0
    motion = float(np.mean(np.abs(d)))
    still = cv2.GaussianBlur(np.abs(d), (0, 0), 3) < 2.5
    if still.mean() < 0.05:
        return float("nan"), motion
    return float(1.4826 * np.median(np.abs(d[still])) / math.sqrt(2)), motion


def blockiness(y: np.ndarray) -> float:
    """Ratio of luma steps on the strongest 8-px grid vs. elsewhere (1.0 = no blocking)."""
    def axis(dd):
        cols = np.array([dd[:, o::8].mean() for o in range(8)])
        return cols.max() / max(1e-6, np.median(cols))
    dh = np.abs(np.diff(y, axis=1))
    dv = np.abs(np.diff(y, axis=0)).T
    return float((axis(dh) + axis(dv)) / 2)


def effective_factor(y8: np.ndarray, sigma: float) -> float:
    """How far the frame can be shrunk and restored without losing real detail.
    2.0 means the picture only holds half its pixel resolution of true detail."""
    clean = cv2.fastNlMeansDenoising(y8, None, h=float(max(3.0, 1.3 * sigma)),
                                     templateWindowSize=7, searchWindowSize=15).astype(np.float32)
    h, w = clean.shape
    best = 1.0
    for f in (1.25, 1.5, 2.0, 2.5, 3.0, 4.0):
        sw, sh = max(8, int(w / f)), max(8, int(h / f))
        rec = cv2.resize(cv2.resize(clean, (sw, sh), interpolation=cv2.INTER_AREA), (w, h),
                         interpolation=cv2.INTER_CUBIC)
        mse = float(np.mean((clean[4:-4, 4:-4] - rec[4:-4, 4:-4]) ** 2))
        psnr = 99.0 if mse == 0 else 10 * math.log10(255 ** 2 / mse)
        if psnr >= 39.0:
            best = f
        else:
            break
    return best


def banding_score(y: np.ndarray) -> float:
    """Share of gentle-gradient areas rendered as flat steps (staircase) - 8-bit banding."""
    smooth = cv2.GaussianBlur(y, (0, 0), 6)
    # Sobel (3x3) reports ~8x the per-pixel slope
    slope = np.hypot(cv2.Sobel(smooth, cv2.CV_32F, 1, 0), cv2.Sobel(smooth, cv2.CV_32F, 0, 1)) / 8
    resid = y - smooth
    texture = np.sqrt(cv2.blur(resid * resid, (9, 9)))
    ramp = (slope > 0.04) & (slope < 0.6) & (texture < 1.2)   # gentle, texture-free gradients
    if ramp.mean() < 0.03:
        return 0.0
    dx = np.abs(np.diff(np.round(y), axis=1))[:, :-1][ramp[:, 1:-1]]
    if dx.size == 0:
        return 0.0
    # A dithered/noisy ramp changes level often; a banded one sits flat then jumps by 1.
    zeros = float((dx == 0).mean())
    return float(np.clip((zeros - 0.7) / 0.25, 0, 1) * min(1.0, ramp.mean() / 0.1))


def content_features(rgb: np.ndarray, y: np.ndarray) -> dict:
    y8 = np.clip(y, 0, 255).astype(np.uint8)
    blur = cv2.GaussianBlur(y, (0, 0), 1.0)
    mean = cv2.blur(blur, (7, 7))
    std = np.sqrt(np.maximum(cv2.blur(blur * blur, (7, 7)) - mean * mean, 0))
    flat = float((std < 1.5).mean())
    texture = float((std > 6).mean())
    edges = cv2.Canny(y8, 60, 160)
    edge_density = float((edges > 0).mean())
    q = (rgb[::2, ::2] >> 3).reshape(-1, 3).astype(np.int32)
    colors = len(np.unique(q[:, 0] * 1024 + q[:, 1] * 32 + q[:, 2])) / q.shape[0]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    sat = float(hsv[..., 1].mean() / 255)
    return dict(flat=flat, texture=texture, edge_density=edge_density, color_ratio=colors,
                saturation=sat, mean_luma=float(y.mean()), dark=float((y < 40).mean()),
                std_map_mean=float(std.mean()))


_FACE = None


def faces(y8: np.ndarray) -> list[tuple[int, int, int, int]]:
    global _FACE
    if _FACE is None:
        try:
            _FACE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        except AttributeError:  # OpenCV build without the cascade module
            _FACE = False
    if not _FACE:
        return []
    return list(_FACE.detectMultiScale(y8, scaleFactor=1.1, minNeighbors=6,
                                       minSize=(max(16, y8.shape[0] // 30),) * 2))


# --------------------------------------------------------------------------- main

def analyze(src: Path, info: dict, base_vf_fn) -> dict:
    """base_vf_fn(info, scan) -> (filter string, width, height) for clean display-res frames."""
    log("Analysing source ...")
    scan = ffmpeg_scan(src, info)
    vf, w, h = base_vf_fn(info, scan)

    dur = info["duration"] or 1
    n_pts = int(min(12, max(4, dur / 3)))
    times = [dur * (0.05 + 0.9 * i / max(1, n_pts - 1)) for i in range(n_pts)] if dur > 3 else [0]
    per = []
    for t in times:
        fr = grab_frames(src, t, vf, w, h, 3)
        if len(fr) < 2:
            continue
        ys = [luma(f) for f in fr]
        y8 = np.clip(ys[0], 0, 255).astype(np.uint8)
        sig = noise_sigma(ys[0])
        tn, motion = temporal_noise(ys[0], ys[1])
        feats = content_features(fr[0], ys[0])
        fc = faces(y8)
        per.append(dict(
            t=t, sigma=sig, temporal_sigma=tn, motion=motion, block=blockiness(ys[0]),
            eff=effective_factor(y8, sig), banding=banding_score(ys[0]),
            faces=len(fc), face_h=max([f[3] for f in fc], default=0) / h, **feats))
    if not per:
        raise RuntimeError("Could not decode sample frames for analysis.")

    def med(k):
        vals = [p[k] for p in per if p[k] == p[k]]  # drop NaN
        return float(np.median(vals)) if vals else float("nan")

    def pct(k, q):
        return float(np.percentile([p[k] for p in per], q))

    prof: dict = {"scan": scan, "samples": per}
    m = {k: med(k) for k in ("sigma", "temporal_sigma", "motion", "block", "eff", "banding", "flat",
                             "texture", "edge_density", "color_ratio", "saturation", "mean_luma",
                             "dark", "std_map_mean")}
    m["face_frames"] = sum(1 for p in per if p["faces"]) / len(per)
    m["face_h"] = pct("face_h", 90)
    m["banding_max"] = pct("banding", 90)
    prof["metrics"] = m

    # ---- bitrate / compression
    disp_w, disp_h = w, h
    fps = float(info["fps"])
    bpp = info["bitrate"] / max(1, info["width"] * info["height"] * fps)
    norm_bpp = bpp / CODEC_EFFICIENCY.get(info["codec"], 1.0)
    m["bpp"], m["norm_bpp"] = bpp, norm_bpp
    ffb = scan.get("ff_block") or 0
    if norm_bpp < 0.05 or m["block"] > 1.25 or ffb > 1.5:
        compression = "heavy"
    elif norm_bpp < 0.12 or m["block"] > 1.08 or ffb > 1.15:
        compression = "moderate"
    else:
        compression = "light"

    # ---- noise / grain
    sig, tsig = m["sigma"], m["temporal_sigma"]
    noise = "clean" if sig < 1.2 else "low" if sig < 2.5 else "medium" if sig < 5 else "high"
    # Real grain: random in time AND at least as strong as the spatial estimate.
    grain = bool(tsig == tsig and tsig > 1.4 and tsig > 0.7 * sig and compression != "heavy")

    # ---- content type
    # Flat colour fills + line edges + little natural texture = drawn content. (Low noise alone
    # is NOT evidence: heavy compression also wipes noise from camera footage.)
    anime_score = 0.0
    anime_score += np.clip((m["flat"] - 0.3) / 0.3, 0, 1) * 0.55
    anime_score += np.clip((m["edge_density"] - 0.03) / 0.07, 0, 1) * 0.2
    anime_score += np.clip((0.15 - m["texture"]) / 0.15, 0, 1) * 0.25
    anime_score -= 0.3 * m["face_frames"]
    anime_score = float(np.clip(anime_score, 0, 1))
    content = "anime" if anime_score > 0.65 else "live" if anime_score < 0.4 else "mixed"

    # ---- effective resolution
    short = min(disp_w, disp_h)
    eff_short = short / m["eff"]
    res_bucket = ("tiny" if eff_short < 300 else "sd" if eff_short < 500 else
                  "hd" if eff_short < 800 else "fhd" if eff_short < 1300 else "uhd")

    # ---- interlace / telecine / bars
    inter_ratio = scan.get("interlaced_ratio", 0)
    repeated = scan.get("repeated_ratio", 0)
    telecine = repeated > 0.12 and abs(fps - 29.97) < 0.1
    interlaced = (info["field_order"] not in ("progressive", "unknown") or inter_ratio > 0.3)
    crop = scan.get("crop")
    bars = None
    if crop:
        cw, ch, cx, cy = crop
        if (info["width"] - cw) >= 0.04 * info["width"] or (info["height"] - ch) >= 0.04 * info["height"]:
            bars = crop

    prof.update(
        content=content, anime_score=anime_score, compression=compression, noise=noise,
        grain=grain, effective_short=eff_short, res_bucket=res_bucket,
        interlaced=interlaced and not telecine, telecine=telecine, bars=bars,
        banding=m["banding_max"] > 0.25, dark=m["dark"] > 0.35 or m["mean_luma"] < 55,
        faces=m["face_frames"] > 0.2, small_faces=m["face_frames"] > 0.2 and m["face_h"] < 0.12,
        motion="high" if m["motion"] > 12 else "medium" if m["motion"] > 4 else "low",
        hdr=info["hdr"], disp=(disp_w, disp_h),
    )
    prof["category"] = f"{content}/{res_bucket}/{compression}/{noise}"
    prof["diagnosis"] = diagnose(prof, info)
    return prof


def feasibility(prof: dict, target_short: int) -> dict:
    factor = target_short / max(1.0, prof["effective_short"])
    levels = ["low", "medium", "high", "extreme"]
    lvl = 0 if factor <= 4.2 else 1 if factor <= 6 else 2 if factor <= 9 else 3
    if prof["compression"] == "heavy" and lvl < 3 and factor > 3:
        lvl += 1
    if prof["noise"] == "high" and lvl < 3 and factor > 3:
        lvl += 1
    return {"factor": factor, "risk": levels[lvl]}


def diagnose(p: dict, info: dict) -> dict:
    fix, keep, notes = [], [], []
    m = p["metrics"]
    if p["telecine"]:
        fix.append("3:2 pulldown (telecined film) -> inverse telecine to the original 23.976 fps")
    elif p["interlaced"]:
        fix.append("interlaced fields -> motion-adaptive deinterlace (bwdif)")
    if p["bars"]:
        cw, ch, *_ = p["bars"]
        fix.append(f"black bars -> crop to active picture {cw}x{ch} before upscaling (re-padded after)")
    if p["hdr"]:
        fix.append("HDR (PQ/HLG) -> tone-map to SDR BT.709 (AI models are trained on SDR)")
    if p["compression"] != "light":
        fix.append(f"{p['compression']} compression (block ratio {m['block']:.2f}, "
                   f"{m['norm_bpp']:.3f} bits/pixel H.264-equiv) -> compression-aware model / deblock")
    if p["noise"] in ("medium", "high") and not p["grain"]:
        fix.append(f"{p['noise']} noise (sigma {m['sigma']:.1f}) -> denoise before/within AI")
    if p["banding"]:
        fix.append("colour banding in gradients -> deband + 10-bit output")
    eff = p["effective_short"]
    disp_short = min(p["disp"])
    if eff < disp_short * 0.8:
        notes.append(f"'fake' resolution: the file is {disp_short}p but only holds ~{eff:.0f}p of "
                     "real detail (it was upscaled before) - the plan uses the true detail level")
    if p["grain"]:
        keep.append(f"natural film grain (temporal sigma {m['temporal_sigma']:.1f}) -> re-grain after "
                    "the AI so it doesn't look plastic")
    if p["faces"]:
        keep.append("faces present -> favour fidelity over invented detail"
                    + (" (faces are small: high hallucination risk)" if p["small_faces"] else ""))
    if p["content"] == "anime":
        keep.append("flat colour areas and clean line art (anime/cartoon)")
    elif p["content"] == "mixed":
        notes.append(f"content type ambiguous (anime score {p['anime_score']:.2f}) - "
                     "both live-action and anime models will compete")
    if m["texture"] > 0.25 and p["content"] != "anime":
        keep.append("fine natural texture (foliage/grass/fabric) -> avoid over-smoothing")
    if p["dark"]:
        keep.append("dark scenes -> protect shadow detail; banding/noise most visible here")
    if p["motion"] == "high":
        notes.append("high motion: motion blur cannot be sharpened; temporal stability matters most")
    return {"fix": fix, "preserve": keep, "notes": notes}
