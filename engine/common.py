"""Shared paths and small helpers."""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOLS = ROOT / "tools"
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"
RUNS_DIR = DATA_DIR / "runs"
LEGACY_ESRGAN_DIR = TOOLS / "realesrgan"

for _d in (MODELS_DIR, DATA_DIR, RUNS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                         errors="replace", **kw)
    if res.returncode != 0:
        raise RuntimeError(f"Command failed ({res.returncode}): {' '.join(map(str, cmd))}\n"
                           f"{res.stderr[-3000:]}")
    return res


def even(x: float) -> int:
    return max(2, int(round(x / 2)) * 2)


def fmt_time(sec: float) -> str:
    sec = int(max(0, sec))
    return f"{sec // 3600}h{sec % 3600 // 60:02d}m{sec % 60:02d}s"


def load_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)
