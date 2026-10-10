"""Container / stream metadata via ffprobe."""
from __future__ import annotations

import json
import sys
from fractions import Fraction
from pathlib import Path

from .common import run

HDR_TRANSFERS = ("smpte2084", "arib-std-b67")


def _frac(s) -> Fraction | None:
    try:
        f = Fraction(s)
        return f if f > 0 else None
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def probe(path: Path) -> dict:
    data = json.loads(run(["ffprobe", "-v", "error", "-print_format", "json", "-show_streams",
                           "-show_format", str(path)]).stdout)
    vids = [s for s in data["streams"] if s["codec_type"] == "video"
            and not s.get("disposition", {}).get("attached_pic")]
    if not vids:
        sys.exit("No video stream found.")
    v = vids[0]
    w, h = int(v["width"]), int(v["height"])

    rotation = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rotation = int(sd["rotation"])
    rotation = int(v.get("tags", {}).get("rotate", rotation))
    if abs(rotation) % 180 == 90:
        w, h = h, w

    sar = Fraction(1)
    if v.get("sample_aspect_ratio") not in (None, "0:1", "1:1", "N/A"):
        num, den = v["sample_aspect_ratio"].split(":")
        if int(num) and int(den):
            sar = Fraction(int(num), int(den))

    r_fps, avg_fps = _frac(v.get("r_frame_rate")), _frac(v.get("avg_frame_rate"))
    fps = r_fps or avg_fps or Fraction(30)
    vfr = bool(r_fps and avg_fps and abs(float(r_fps) - float(avg_fps)) > 0.01 * float(r_fps))
    if vfr or fps > 240:
        fps = (avg_fps or Fraction(30)).limit_denominator(1001)

    duration = float(v.get("duration") or data["format"].get("duration") or 0)
    size = int(data["format"].get("size") or Path(path).stat().st_size)
    audio = [s for s in data["streams"] if s["codec_type"] == "audio"]
    v_bitrate = int(v.get("bit_rate") or 0)
    if not v_bitrate and duration:
        a_bits = sum(int(a.get("bit_rate") or 128000) for a in audio)
        v_bitrate = max(1, int(size * 8 / duration) - a_bits)

    return {
        "width": w, "height": h, "sar": sar, "fps": fps, "vfr": vfr, "duration": duration,
        "field_order": v.get("field_order", "unknown"),
        "transfer": v.get("color_transfer", ""), "matrix": v.get("color_space", ""),
        "range": v.get("color_range", "tv"), "pix_fmt": v.get("pix_fmt", ""),
        "codec": v.get("codec_name", "?"), "bitrate": v_bitrate, "size": size,
        "hdr": v.get("color_transfer", "") in HDR_TRANSFERS,
        "audio": audio,
        "subs": [s for s in data["streams"] if s["codec_type"] == "subtitle"],
    }
