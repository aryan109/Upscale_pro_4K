"""Model + engine registry with self-update ("is there new tech?").

Sources are an allowlist of official repositories. Newly discovered models enter as
status="trial": the planner adds them as challengers to tournaments, and the knowledge
base promotes or retires them based on how they score against the incumbents.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from .common import DATA_DIR, LEGACY_ESRGAN_DIR, MODELS_DIR, TOOLS, load_json, log, save_json

REGISTRY_PATH = DATA_DIR / "registry.json"
CHECK_INTERVAL_H = 24
UA = {"User-Agent": "UpscalePro4K/2.0"}

UPSCAYL_CUSTOM = "https://raw.githubusercontent.com/upscayl/custom-models/main/models/"
UPSCAYL_MAIN = "https://raw.githubusercontent.com/upscayl/upscayl/main/resources/models/"
DISCOVERY_DIRS = [
    ("https://api.github.com/repos/upscayl/custom-models/contents/models", UPSCAYL_CUSTOM),
    ("https://api.github.com/repos/upscayl/upscayl/contents/resources/models", UPSCAYL_MAIN),
]
ENGINE_RELEASES = "https://api.github.com/repos/upscayl/upscayl-ncnn/releases/latest"
OPENMODELDB_TREE = "https://api.github.com/repos/OpenModelDB/open-model-database/git/trees/main?recursive=1"
OPENMODELDB_RAW = "https://raw.githubusercontent.com/OpenModelDB/open-model-database/main/"

# Curated seed catalogue (researched 2026-10). domain: live|anime|any.
# handles: degradations the model was trained to undo.  speed: fast (SRVGG/Compact) | slow (ESRGAN).
SEED = {
    # --- live action / camera footage ---
    "realesrgan-x4plus": dict(domain="live", handles=["noise", "blur", "jpeg"], speed="slow",
                              license="BSD-3", src="local", status="active",
                              notes="Real-ESRGAN general; strong cleanup, can look smooth"),
    "RealESRGAN_General_x4_v3": dict(domain="live", handles=["noise", "blur", "jpeg"], speed="fast",
                                     license="BSD-3", src=UPSCAYL_CUSTOM, status="active",
                                     notes="realesr-general-x4v3; fast, strong denoise"),
    "RealESRGAN_General_WDN_x4_v3": dict(domain="live", handles=["blur", "jpeg"], speed="fast",
                                         license="BSD-3", src=UPSCAYL_CUSTOM, status="active",
                                         notes="weak-denoise twin; keeps more natural texture"),
    "4xLSDIRCompactC3": dict(domain="live", handles=["jpeg"], speed="fast", license="CC-BY-4.0",
                             src=UPSCAYL_CUSTOM, status="active",
                             notes="Compact, trained on JPEG q30-100 (compressed web video)"),
    "4xLSDIRplusC": dict(domain="live", handles=["jpeg"], speed="slow", license="CC-BY-4.0",
                         src=UPSCAYL_CUSTOM, status="active",
                         notes="x4plus fine-tune on LSDIR with JPEG compression"),
    "4xNomos8kSC": dict(domain="live", handles=["jpeg", "blur"], speed="slow", license="CC-BY-4.0",
                        src=UPSCAYL_CUSTOM, status="active",
                        notes="photo upscaler, OTF JPEG + blur degradations"),
    "4x_NMKD-Siax_200k": dict(domain="live", handles=["clean", "jpeg"], speed="slow",
                              license="WTFPL", src=UPSCAYL_CUSTOM, status="active",
                              notes="general; clean or lightly compressed (q75+)"),
    "4x_NMKD-Superscale-SP_178000_G": dict(domain="live", handles=["clean"], speed="slow",
                                           license="WTFPL", src=UPSCAYL_CUSTOM, status="trial",
                                           notes="clean, artefact-free sources"),
    "4xLSDIR": dict(domain="live", handles=["clean"], speed="slow", license="CC-BY-4.0",
                    src=UPSCAYL_CUSTOM, status="trial", notes="clean photos"),
    "remacri-4x": dict(domain="live", handles=["clean"], speed="slow", license="CC-BY-NC-SA-4.0",
                       src=UPSCAYL_MAIN, status="trial", noncommercial=True,
                       notes="Foolhardy Remacri; detail, less smoothing"),
    "ultrasharp-4x": dict(domain="live", handles=["jpeg"], speed="slow", license="CC-BY-NC-SA-4.0",
                          src=UPSCAYL_MAIN, status="trial", noncommercial=True,
                          notes="Kim2091 UltraSharp"),
    "ultramix-balanced-4x": dict(domain="live", handles=["jpeg", "clean"], speed="slow",
                                 license="CC-BY-NC-SA-4.0", src=UPSCAYL_MAIN, status="trial",
                                 noncommercial=True, notes="Kim2091 UltraMix balanced"),
    "high-fidelity-4x": dict(domain="live", handles=["clean"], speed="slow", license="unknown",
                             src=UPSCAYL_MAIN, status="trial", notes="Upscayl high fidelity"),
    # --- anime / cartoon / art ---
    "realesr-animevideov3-x4": dict(domain="anime", handles=["jpeg", "noise", "blur"], speed="fast",
                                    license="BSD-3", src="local", status="active",
                                    notes="video-tuned, temporally stable"),
    "realesrgan-x4plus-anime": dict(domain="anime", handles=["jpeg", "noise", "blur"], speed="slow",
                                    license="BSD-3", src="local", status="active",
                                    notes="sharp line art"),
    "4xHFA2k": dict(domain="anime", handles=["jpeg", "blur"], speed="slow", license="CC-BY-4.0",
                    src=UPSCAYL_CUSTOM, status="active", notes="anime frames, Phhofm"),
    "digital-art-4x": dict(domain="anime", handles=["clean"], speed="slow", license="unknown",
                           src=UPSCAYL_MAIN, status="trial", notes="digital art / illustration"),
}


def _get(url: str, timeout: int = 30) -> bytes:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f)
    tmp.replace(dest)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _guess_meta(name: str, bin_size: int) -> dict:
    n = name.lower()
    anime = any(k in n for k in ("anime", "hfa", "art", "cartoon", "manga", "toon"))
    handles = []
    if any(k in n for k in ("jpg", "jpeg", "dejpeg", "plusc", "compress")) or n.endswith("c"):
        handles.append("jpeg")
    if any(k in n for k in ("denoise", "noise", "dn")):
        handles.append("noise")
    if "blur" in n or "deblur" in n:
        handles.append("blur")
    scale = 2 if ("x2" in n or "2x" in n) else 3 if ("x3" in n or "3x" in n) else \
        1 if ("x1" in n or "1x" in n) else 4
    return dict(domain="anime" if anime else "live", handles=handles or ["clean"],
                speed="fast" if bin_size < 10_000_000 else "slow", scale=scale,
                license="unknown", notes="auto-discovered")


class Registry:
    def __init__(self) -> None:
        self.data = load_json(REGISTRY_PATH, {})
        self.data.setdefault("models", {})
        self.data.setdefault("engine", {})
        self.data.setdefault("watchlist", [])
        self.data.setdefault("omdb_known", [])
        for mid, meta in SEED.items():
            entry = self.data["models"].setdefault(mid, {})
            for k, v in meta.items():
                entry.setdefault(k, v)
            entry.setdefault("scale", 4)
            entry.setdefault("added", "seed")
        self._init_engine()
        self.save()

    # ------------------------------------------------------------------ engine
    def _init_engine(self) -> None:
        eng = self.data["engine"]
        if not eng.get("path") or not Path(eng["path"]).exists():
            ub = TOOLS / "upscayl" / "upscayl-bin.exe"
            legacy = LEGACY_ESRGAN_DIR / "realesrgan-ncnn-vulkan.exe"
            if ub.exists():
                eng.update(path=str(ub), kind="upscayl", tag=eng.get("tag", "20251207-174704"))
            elif legacy.exists():
                eng.update(path=str(legacy), kind="realesrgan", tag="v0.2.5.0")
        # Official Real-ESRGAN models ship with the legacy engine: copy them into the model store.
        src = LEGACY_ESRGAN_DIR / "models"
        if src.exists():
            for f in src.iterdir():
                if not (MODELS_DIR / f.name).exists():
                    shutil.copy2(f, MODELS_DIR / f.name)

    @property
    def engine(self) -> Path:
        return Path(self.data["engine"]["path"])

    @property
    def engine_kind(self) -> str:
        return self.data["engine"].get("kind", "upscayl")

    def save(self) -> None:
        save_json(REGISTRY_PATH, self.data)

    # ------------------------------------------------------------------ models
    def models(self, include_retired=False, allow_nc=False) -> dict:
        out = {}
        for mid, m in self.data["models"].items():
            if m.get("status") == "retired" and not include_retired:
                continue
            if m.get("noncommercial") and not allow_nc:
                continue
            if self.engine_kind != "upscayl" and m.get("src") != "local":
                continue  # legacy engine can only run the official models
            out[mid] = m
        return out

    def ensure(self, mid: str) -> bool:
        """Download a model's .param/.bin if missing. Returns True if usable."""
        m = self.data["models"][mid]
        p, b = MODELS_DIR / f"{mid}.param", MODELS_DIR / f"{mid}.bin"
        if p.exists() and b.exists():
            return True
        src = m.get("src")
        if not src or src == "local":
            return False
        try:
            log(f"Downloading model {mid} ...")
            for f in (p, b):
                _download(src + f.name, f)
            m["sha256"] = _sha256(b)
            self.save()
            return True
        except Exception as e:  # network or missing file
            log(f"  could not download {mid}: {e}")
            for f in (p, b):
                f.unlink(missing_ok=True)
            return False

    def set_status(self, mid: str, status: str, reason: str = "") -> None:
        m = self.data["models"].get(mid)
        if m and m.get("status") != status:
            log(f"Model {mid}: {m.get('status')} -> {status} {reason}")
            m["status"] = status
            m.setdefault("history", []).append([time.strftime("%Y-%m-%d"), status, reason])
            self.save()

    # ------------------------------------------------------------------ self-update
    def check_updates(self, force: bool = False, allow_engine: bool = True) -> list[str]:
        last = self.data.get("last_check", 0)
        if not force and time.time() - last < CHECK_INTERVAL_H * 3600:
            return []
        log("Checking for new upscaling tech (models + engine) ...")
        news: list[str] = []
        news += self._discover_models()
        if allow_engine:
            news += self._check_engine()
        news += self._watch_openmodeldb()
        self.data["last_check"] = time.time()
        self.save()
        if not news:
            log("  nothing new.")
        for n in news:
            log(f"  NEW: {n}")
        return news

    def _discover_models(self) -> list[str]:
        news = []
        for api, raw in DISCOVERY_DIRS:
            try:
                listing = json.loads(_get(api))
            except Exception as e:
                log(f"  discovery failed for {api}: {e}")
                continue
            files = {f["name"]: f.get("size", 0) for f in listing if isinstance(f, dict)}
            for name, size in files.items():
                if not name.endswith(".bin"):
                    continue
                mid = name[:-4]
                if mid in self.data["models"] or f"{mid}.param" not in files:
                    continue
                meta = _guess_meta(mid, size)
                meta.update(src=raw, status="trial", added=time.strftime("%Y-%m-%d"))
                self.data["models"][mid] = meta
                news.append(f"model '{mid}' ({meta['domain']}, {meta['speed']}) -> trial")
        return news

    def _check_engine(self) -> list[str]:
        if self.engine_kind != "upscayl":
            return []
        try:
            rel = json.loads(_get(ENGINE_RELEASES))
        except Exception as e:
            log(f"  engine check failed: {e}")
            return []
        tag = rel.get("tag_name", "")
        if not tag or tag <= self.data["engine"].get("tag", ""):
            return []
        asset = next((a for a in rel.get("assets", []) if a["name"].endswith("windows.zip")), None)
        if not asset:
            return []
        dest = TOOLS / f"upscayl-{tag}"
        try:
            zpath = TOOLS / f"_dl_{tag}.zip"
            _download(asset["browser_download_url"], zpath)
            with zipfile.ZipFile(zpath) as z:
                z.extractall(TOOLS / f"_dl_{tag}")
            exe = next((TOOLS / f"_dl_{tag}").rglob("upscayl-bin.exe"))
            dest.mkdir(exist_ok=True)
            shutil.copy2(exe, dest / "upscayl-bin.exe")
        finally:
            zpath.unlink(missing_ok=True)
            shutil.rmtree(TOOLS / f"_dl_{tag}", ignore_errors=True)
        ok, msg = self._ab_engine(dest / "upscayl-bin.exe")
        if ok:
            self.data["engine"].update(path=str(dest / "upscayl-bin.exe"), tag=tag,
                                       previous=self.data["engine"].get("path"))
            self.save()
            return [f"engine {tag} adopted ({msg})"]
        return [f"engine {tag} rejected ({msg}); kept {self.data['engine'].get('tag')}"]

    def _ab_engine(self, candidate: Path) -> tuple[bool, str]:
        """A/B the new engine against the current one on a synthetic frame."""
        work = DATA_DIR / "_engine_ab"
        shutil.rmtree(work, ignore_errors=True)
        (work / "in").mkdir(parents=True)
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180", "-frames:v", "1", str(work / "in" / "a.png")])
        times, imgs = {}, {}
        for tag, exe in (("old", self.engine), ("new", candidate)):
            out = work / tag
            out.mkdir()
            t = time.time()
            r = subprocess.run([str(exe), "-i", str(work / "in"), "-o", str(out), "-m",
                                str(MODELS_DIR), "-n", "realesr-animevideov3-x4", "-s", "4",
                                "-z", "4", "-f", "png"], capture_output=True)
            times[tag] = time.time() - t
            f = out / "a.png"
            if r.returncode != 0 or not f.exists():
                shutil.rmtree(work, ignore_errors=True)
                return (tag == "old"), f"{tag} engine failed smoke test"
            import cv2
            imgs[tag] = cv2.imread(str(f)).astype(np.float32)
        shutil.rmtree(work, ignore_errors=True)
        if imgs["old"].shape != imgs["new"].shape:
            return False, "output size differs"
        mse = float(np.mean((imgs["old"] - imgs["new"]) ** 2))
        psnr = 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)
        if psnr < 40:
            return False, f"output differs (PSNR {psnr:.1f} dB) - needs a full tournament"
        return True, f"identical output (PSNR {psnr:.1f} dB), {times['old']:.2f}s -> {times['new']:.2f}s"

    def _watch_openmodeldb(self) -> list[str]:
        """Track newly published research models. Most ship as PyTorch .pth (not runnable by the
        Vulkan engine); those go on a watchlist. Any with an ncnn resource become trials."""
        try:
            tree = json.loads(_get(OPENMODELDB_TREE, timeout=60))
        except Exception as e:
            log(f"  OpenModelDB check failed: {e}")
            return []
        paths = [t["path"] for t in tree.get("tree", [])
                 if t["path"].startswith("data/models/") and t["path"].endswith(".json")]
        known = set(self.data["omdb_known"])
        first_run = not known
        new_paths = [p for p in paths if p not in known]
        self.data["omdb_known"] = paths
        if first_run:
            return [f"OpenModelDB baseline recorded ({len(paths)} models)"]
        news = []
        for p in new_paths[:40]:
            try:
                meta = json.loads(_get(OPENMODELDB_RAW + p))
            except Exception:
                continue
            if meta.get("scale") not in (2, 4):
                continue
            tags = set(meta.get("tags", []))
            if not tags & {"photo", "realistic", "anime", "restoration", "compression-removal",
                           "general-upscaler", "video-frame"}:
                continue
            mid = Path(p).stem
            ncnn = next((r for r in meta.get("resources", []) if r.get("platform") == "ncnn"), None)
            item = {"id": mid, "name": meta.get("name"), "scale": meta.get("scale"),
                    "tags": sorted(tags), "license": meta.get("license"),
                    "arch": meta.get("architecture"), "found": time.strftime("%Y-%m-%d")}
            if ncnn and ncnn.get("urls"):
                item["ncnn_urls"] = ncnn["urls"]
            self.data["watchlist"].append(item)
            news.append(f"research model '{meta.get('name')}' ({meta.get('architecture')}, "
                        f"x{meta.get('scale')}) on watchlist"
                        f"{' - ncnn build available' if ncnn else ' - PyTorch only'}")
        return news
