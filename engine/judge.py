"""No-reference judge: scores a candidate's output crops against the source.

There is no 'true 4K' to compare with, so the judge measures what matters relative to the source:
  fidelity   - downscaled back, does it still match the source? (no invented shapes / colour shifts)
  detail     - real sharpness gain over a plain Lanczos upscale
  texture    - is natural texture kept (anti-'plastic' / waxy look)?
  temporal   - does it shimmer/flicker between frames more than the source?
  blocking   - compression blocks surviving into the output
  ringing    - halos / overshoot around edges (over-sharpening)
  colour     - chroma drift
Weights live in the knowledge base and are tuned by user feedback.
"""
from __future__ import annotations

import math

import cv2
import numpy as np

from .analyze import blockiness, luma

DEFAULT_WEIGHTS = {"fidelity": 0.25, "detail": 0.22, "texture": 0.20, "temporal": 0.15,
                   "blocking": 0.10, "ringing": 0.10, "colour": 0.05, "iqa": 0.0}
POSITIVE = ("fidelity", "detail", "texture", "temporal", "iqa")

_IQA = None


def _iqa_model():
    """Optional learned no-reference metric (pyiqa TOPIQ-NR) if installed."""
    global _IQA
    if _IQA is None:
        try:
            import pyiqa  # noqa
            import torch  # noqa
            _IQA = pyiqa.create_metric("topiq_nr", device="cpu")
        except Exception:
            _IQA = False
    return _IQA or None


def iqa_available() -> bool:
    return _iqa_model() is not None


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a, mu_b = cv2.GaussianBlur(a, (11, 11), 1.5), cv2.GaussianBlur(b, (11, 11), 1.5)
    sa = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a ** 2
    sb = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b ** 2
    sab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * sab + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (sa + sb + c2))
    return float(m[5:-5, 5:-5].mean())


def _chroma(rgb: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb).astype(np.float32)
    return ycc[..., 1:]


def _local_std(x: np.ndarray, k: int = 7) -> np.ndarray:
    m = cv2.blur(x, (k, k))
    return np.sqrt(np.maximum(cv2.blur(x * x, (k, k)) - m * m, 0))


def raw_metrics(out_seq: list[np.ndarray], ref_seq: list[np.ndarray], sigma: float) -> dict:
    """out_seq: candidate RGB frames (output res); ref_seq: clean source RGB (AI-input res)."""
    oh, ow = out_seq[0].shape[:2]
    rh, rw = ref_seq[0].shape[:2]
    k = ow / rw
    Y = [luma(o) for o in out_seq]
    R = [luma(r) for r in ref_seq]
    B = [cv2.resize(r, (ow, oh), interpolation=cv2.INTER_LANCZOS4) for r in R]
    O_ds = [cv2.resize(y, (rw, rh), interpolation=cv2.INTER_AREA) for y in Y]

    fid, col, det, tex, blk, ring = [], [], [], [], [], []
    for j in range(len(Y)):
        fid.append(_ssim(cv2.GaussianBlur(O_ds[j], (0, 0), 1.2), cv2.GaussianBlur(R[j], (0, 0), 1.2)))
        co = cv2.resize(_chroma(out_seq[j]), (rw, rh), interpolation=cv2.INTER_AREA)
        cr = _chroma(ref_seq[j])
        col.append(float(np.mean(np.abs(cv2.GaussianBlur(co, (0, 0), 2) - cv2.GaussianBlur(cr, (0, 0), 2)))))
        hf_o = float(cv2.Laplacian(Y[j], cv2.CV_32F).var())
        hf_b = float(cv2.Laplacian(B[j], cv2.CV_32F).var()) + 1e-3
        det.append(hf_o / hf_b)

        # texture mask: textured (not edge, not noise) areas of the source
        rstd = _local_std(cv2.GaussianBlur(R[j], (0, 0), 0.7))
        edges = cv2.dilate(cv2.Canny(np.clip(R[j], 0, 255).astype(np.uint8), 80, 200),
                           np.ones((5, 5), np.uint8)) > 0
        tmask = (rstd > max(4.0, 3.0 * sigma)) & ~edges
        if tmask.mean() > 0.02:
            tm = cv2.resize(tmask.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
            s = max(0.8, 0.6 * k)
            bp_o = np.abs(Y[j] - cv2.GaussianBlur(Y[j], (0, 0), s))
            bp_b = np.abs(B[j] - cv2.GaussianBlur(B[j], (0, 0), s))
            tex.append(float(bp_o[tm].mean() / (bp_b[tm].mean() + 1e-3)))

        blk.append(blockiness(O_ds[j]))

        # ringing: overshoot beyond the local range of the Lanczos baseline near strong edges
        e = cv2.dilate(cv2.Canny(np.clip(B[j], 0, 255).astype(np.uint8), 60, 160),
                       np.ones((int(2 * k) + 1,) * 2, np.uint8)) > 0
        kern = np.ones((int(2 * k) + 3,) * 2, np.uint8)
        hi, lo = cv2.dilate(B[j], kern), cv2.erode(B[j], kern)
        over = ((Y[j] > hi + 10) | (Y[j] < lo - 10)) & e
        ring.append(float(over.sum() / max(1, e.sum())))

    temporal = []
    for j in range(len(Y) - 1):
        d_ref = np.abs(R[j + 1] - R[j])
        still = cv2.GaussianBlur(d_ref, (0, 0), 2) < 2.0
        if still.mean() < 0.05:
            continue
        st = cv2.resize(still.astype(np.uint8), (ow, oh), interpolation=cv2.INTER_NEAREST) > 0
        d_o = np.abs(Y[j + 1] - Y[j])[st].mean()
        d_b = np.abs(B[j + 1] - B[j])[st].mean()
        temporal.append(float(d_o / (d_b + 0.3)))

    res = {
        "scale_k": float(k),
        "fidelity": float(np.mean(fid)), "colour_err": float(np.mean(col)),
        "detail_ratio": float(np.median(det)),
        "texture_ratio": float(np.median(tex)) if tex else float("nan"),
        "blockiness": float(np.mean(blk)), "ringing": float(np.mean(ring)),
        "flicker_ratio": float(np.median(temporal)) if temporal else float("nan"),
    }
    iqa = _iqa_model()
    if iqa is not None:
        import torch
        t = torch.from_numpy(out_seq[0]).permute(2, 0, 1)[None].float() / 255
        with torch.no_grad():
            res["iqa"] = float(iqa(t).item())
    return res


def normalise(m: dict) -> dict:
    """Map raw metrics to 0..1 component scores (1 = good)."""
    def nz(x, default):
        return default if x != x else x
    # A plain Lanczos upscale by k loses HF energy roughly with k^2, so judge the gain relative to
    # k^2: r=1 ~ 'restores the lost sharpness', r=2 strong, r>4 over-cooked/crunchy.
    k2 = max(1.0, m.get("scale_k", 2.0)) ** 2
    r = max(1e-3, m["detail_ratio"] / k2)
    detail = 0.55 + 0.45 * math.log2(r)
    if r > 4:
        detail -= (math.log2(r) - 2) * 0.6
    # texture vs Lanczos in the source's textured areas: <0.7 = waxy/plastic, ~1.6 = rich,
    # >2.5 = invented noise-like texture
    tr = nz(m["texture_ratio"], 1.0)
    texture = min(tr / 1.6, 1.0) - max(0.0, tr - 2.5) * 0.5
    c = {
        "fidelity": np.clip((m["fidelity"] - 0.95) / 0.05, 0, 1),
        "detail": np.clip(detail, 0, 1),
        "texture": np.clip(texture, 0, 1),
        "temporal": np.clip(1.5 - nz(m["flicker_ratio"], 1.0), 0, 1),
        "blocking": np.clip((m["blockiness"] - 1.03) * 5, 0, 1),
        "ringing": np.clip(m["ringing"] * 15, 0, 1),
        "colour": np.clip(m["colour_err"] / 6, 0, 1),
    }
    if "iqa" in m:
        c["iqa"] = np.clip(m["iqa"], 0, 1)
    return {k: float(v) for k, v in c.items()}


def score(components: dict, weights: dict) -> float:
    pos = sum(weights.get(k, 0) for k in POSITIVE if k in components)
    s = sum(weights.get(k, 0) * components[k] for k in POSITIVE if k in components)
    s -= sum(weights.get(k, 0) * components[k] for k in ("blocking", "ringing", "colour"))
    return float(100 * s / max(1e-6, pos))
