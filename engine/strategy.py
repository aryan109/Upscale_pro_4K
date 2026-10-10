"""A Strategy is one complete upscaling recipe. Geometry + FFmpeg filter builders live here so the
tournament (sample crops) and the full render use exactly the same processing."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from fractions import Fraction

from .common import even

DENOISE = {
    "none": None,
    "light": "hqdn3d=1.5:1.5:3:3",
    "medium": "hqdn3d=3:3:5:5",
    "strong": "hqdn3d=5:4:8:7",
    "nlm": "nlmeans=s=2.5:p=7:r=15",  # slow, detail-preserving
}
DEBLOCK = {
    "none": None,
    "weak": "deblock=filter=weak:block=8",
    "strong": "deblock=filter=strong:block=8",
}
# noise-filter strength per 1.0 of target luma sigma (8-bit); measured: c0s 5/10/20 -> sigma 2.9/6.2/12.8
GRAIN_GAIN = 1.65


@dataclass
class Strategy:
    model: str
    model_scale: int = 4
    deblock: str = "none"
    denoise: str = "none"
    blend: float = 0.0      # share of the plain Lanczos picture mixed back in (texture recovery)
    sharpen: float = 0.0    # CAS strength
    grain: float = 0.0      # target luma sigma of re-applied grain
    deband: bool = False
    deflicker: bool = False
    tta: bool = False
    tags: list = field(default_factory=list)

    def key(self) -> str:
        """Identity for learning (grain/deband are rule-based finishing, not learned)."""
        return (f"{self.model}|db={self.deblock}|dn={self.denoise}|bl={self.blend:.2f}"
                f"|sh={self.sharpen:.2f}|tta={int(self.tta)}")

    def ai_key(self) -> str:
        """Everything that changes the AI output frames (post steps excluded)."""
        return f"{self.model}|db={self.deblock}|dn={self.denoise}|tta={int(self.tta)}"

    def label(self) -> str:
        bits = [self.model]
        if self.deblock != "none":
            bits.append(f"deblock-{self.deblock}")
        if self.denoise != "none":
            bits.append(f"denoise-{self.denoise}")
        if self.blend:
            bits.append(f"blend{int(self.blend * 100)}%")
        if self.sharpen:
            bits.append(f"cas{self.sharpen:.2f}")
        if self.grain:
            bits.append(f"grain{self.grain:.1f}")
        if self.deband:
            bits.append("deband")
        if self.deflicker:
            bits.append("deflicker")
        if self.tta:
            bits.append("tta")
        return " + ".join(bits)

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Strategy":
        return Strategy(**{k: v for k, v in d.items() if k in Strategy.__dataclass_fields__})


# --------------------------------------------------------------------------- geometry

def output_fps(info: dict, prof: dict) -> Fraction:
    fps = info["fps"]
    if prof.get("telecine"):
        return Fraction(24000, 1001)
    return fps


def geometry(info: dict, prof: dict, target: tuple[int, int], mode: str, model_scale: int = 4,
             use_effective: bool = True) -> dict:
    sar = float(info["sar"])
    if prof.get("bars"):
        cw, ch, cx, cy = prof["bars"]
    else:
        cw, ch, cx, cy = info["width"], info["height"], 0, 0
    disp_w, disp_h = even(cw * sar), even(ch)

    tw, th = target
    if disp_h > disp_w and tw > th:
        tw, th = th, tw
    if mode == "fill":
        s = max(tw / disp_w, th / disp_h)
        out_w, out_h = tw, th
    else:
        s = min(tw / disp_w, th / disp_h)
        out_w, out_h = even(disp_w * s), even(disp_h * s)
        if abs(out_w - tw) <= tw * 0.01 and abs(out_h - th) <= th * 0.01:
            out_w, out_h = tw, th

    # Feed the model the picture at its TRUE detail level: a "fake" 1080p holding 540p of detail
    # is reduced to ~540p so the model sees detail at the scale it was trained on.
    in_s = 1.0
    eff_factor = min(disp_w, disp_h) / max(1.0, prof.get("effective_short", min(disp_w, disp_h)))
    if use_effective and eff_factor >= 1.5:
        in_s = 1.0 / eff_factor
    need_s = s / model_scale            # smallest input scale that still lands on the target
    in_s = max(in_s, need_s)
    if s <= 1.0:
        in_s = 1.0
    ai_w, ai_h = even(disp_w * in_s), even(disp_h * in_s)
    return {
        "crop": (cw, ch, cx, cy) if prof.get("bars") else None,
        "disp": (disp_w, disp_h), "ai_in": (ai_w, ai_h),
        "ai_out": (ai_w * model_scale, ai_h * model_scale),
        "scale": s, "out": (out_w, out_h), "target": (tw, th), "mode": mode,
        "use_ai": s > 1.0, "model_scale": model_scale,
        "prescale": in_s, "eff_factor": eff_factor,
    }


# --------------------------------------------------------------------------- filters

def _matrix(info: dict) -> str:
    m = info["matrix"]
    if m in ("bt709",):
        return "bt709"
    if m in ("bt2020nc", "bt2020c"):
        return "bt2020"
    if m in ("smpte170m", "bt470bg"):
        return "bt601"
    return "bt709" if info["height"] >= 720 else "bt601"


def base_filters(info: dict, prof: dict, no_deinterlace: bool = False) -> list[str]:
    """Format-level fixes that every path needs (no quality decisions)."""
    f = []
    fps = output_fps(info, prof)
    if prof.get("telecine") and not no_deinterlace:
        f.append("fieldmatch=order=auto:combmatch=full,bwdif=deint=interlaced,decimate")
    elif prof.get("interlaced") and not no_deinterlace:
        f.append("bwdif=mode=send_frame:parity=auto:deint=all")
    if info["hdr"]:
        f.append("zscale=t=linear:npl=100,format=gbrpf32le,zscale=p=bt709,"
                 "tonemap=hable:desat=0,zscale=t=bt709:m=bt709:r=tv,format=yuv420p")
    f.append(f"fps={fps.numerator}/{fps.denominator}")
    if prof.get("bars"):
        cw, ch, cx, cy = prof["bars"]
        f.append(f"crop={cw}:{ch}:{cx}:{cy}")
    return f


def to_rgb(info: dict, w: int, h: int) -> str:
    matrix = "bt709" if info["hdr"] else _matrix(info)
    rng = "pc" if info["range"] == "pc" else "tv"
    return (f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int:"
            f"in_color_matrix={matrix}:in_range={rng}:out_range=pc,setsar=1,format=rgb24")


def analysis_filters(info: dict, scan: dict) -> tuple[str, int, int]:
    """Display-resolution frames for the analyser (deinterlaced, tone-mapped, square pixels)."""
    prof = {"interlaced": info["field_order"] not in ("progressive", "unknown")
            or scan.get("interlaced_ratio", 0) > 0.3, "telecine": False}
    w, h = even(info["width"] * float(info["sar"])), even(info["height"])
    return ",".join(base_filters(info, prof) + [to_rgb(info, w, h)]), w, h


def decode_filters(info: dict, prof: dict, geo: dict, strat: Strategy | None,
                   extra_crop: tuple | None = None, no_deinterlace: bool = False) -> str:
    """Source -> AI input frames (rgb24). strat=None gives the clean reference (no cleanup)."""
    f = base_filters(info, prof, no_deinterlace)
    if strat is not None:
        if DEBLOCK.get(strat.deblock):
            f.append(DEBLOCK[strat.deblock])
        if DENOISE.get(strat.denoise):
            f.append(DENOISE[strat.denoise])
    w, h = geo["ai_in"]
    f.append(to_rgb(info, w, h))
    if extra_crop:
        cw, ch, cx, cy = extra_crop
        f.append(f"crop={cw}:{ch}:{cx}:{cy}")
    return ",".join(f)


def post_graph(strat: Strategy, w: int, h: int, final: str = "yuv420p10le",
               tail: str = "", seed: int = 1) -> str:
    """filter_complex: [0:v]=AI frames, [1:v]=AI-input frames (used only when blending)."""
    sc = f"scale={w}:{h}:flags=lanczos+accurate_rnd+full_chroma_int"
    parts = [f"[0:v]{sc},format=gbrp[a]"]
    cur = "[a]"
    if strat.blend > 0:
        r = strat.blend
        parts.append(f"[1:v]{sc},format=gbrp[b]")
        parts.append(f"[a][b]blend=all_expr='A*{1 - r:.3f}+B*{r:.3f}'[m]")
        cur = "[m]"
    chain = ["scale=out_color_matrix=bt709:out_range=tv", "format=yuv444p"]
    if strat.deflicker:
        chain.append("hqdn3d=0:0:2.5:2.5")
    if strat.deband:
        chain.append("deband=1thr=0.015:2thr=0.015:3thr=0.015:range=16:blur=1")
    if strat.sharpen > 0:
        chain.append(f"cas=strength={strat.sharpen:.2f}")
    if strat.grain > 0:
        chain.append(f"noise=c0s={max(1, round(strat.grain * GRAIN_GAIN))}:c0f=t:all_seed={seed}")
    if tail:
        chain.append(tail)
    if final == "rgb24":
        chain.append("scale=in_color_matrix=bt709:in_range=tv:out_range=pc,format=rgb24")
    else:
        chain.append(f"format={final}")
    parts.append(f"{cur}{','.join(chain)}[v]")
    return ";".join(parts)


def final_tail(geo: dict, pad: bool) -> str:
    """Fill-crop / pad applied after the post chain in the full render."""
    tw, th = geo["target"]
    ow, oh = geo["out"]
    if geo["mode"] == "fill":
        return ""  # handled by scaling to cover + crop in render
    if pad and (ow, oh) != (tw, th):
        return f"pad={tw}:{th}:(ow-iw)/2:(oh-ih)/2:black"
    return ""
