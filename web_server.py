"""Upscale Pro 4K - Modern Local Web Server & Interactive Dashboard.

Provides a rich web interface for:
- Video uploading and selection
- Instant AI video diagnosis (probe, analyze, 4K feasibility & risk estimation)
- Configurable pipeline options (models, tournament, target resolutions, codecs)
- Live background job execution with real-time log streaming (SSE)
- Tournament score analysis and contact sheet visualizer
- Video preview and comparison player
- Run history browser and human feedback learning
- Model catalogue and automatic engine updater
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from engine.analyze import analyze, feasibility
from engine.common import DATA_DIR, MODELS_DIR, ROOT, RUNS_DIR, fmt_time, load_json, save_json
from engine.judge import iqa_available
from engine.knowledge import Knowledge
from engine.probe import probe
from engine.registry import Registry
from engine.strategy import analysis_filters, geometry
from engine.telegram_auth import auth_manager

app = FastAPI(title="Upscale Pro 4K Web Server", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOADS_DIR = ROOT / "uploads"
TEST_DIR = ROOT / "Test"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Background Job Manager
# ---------------------------------------------------------------------------
class Job:
    def __init__(self, job_id: str, command: str, cmd_args: List[str], input_path: str, meta: dict):
        self.id = job_id
        self.command = command
        self.cmd_args = cmd_args
        self.input_path = input_path
        self.meta = meta
        self.status = "queued"  # queued, running, completed, failed, cancelled
        self.progress_stage = "Starting"
        self.logs: List[str] = []
        self.subscribers: List[asyncio.Queue] = []
        self.process: Optional[subprocess.Popen] = None
        self.start_time: float = time.time()
        self.end_time: Optional[float] = None
        self.run_id: Optional[str] = None
        self.output_video: Optional[str] = None
        self.compare_video: Optional[str] = None
        self.error: Optional[str] = None

    def add_log(self, line: str):
        self.logs.append(line)
        # Parse stages from output
        l = line.lower()
        if "analysing source" in l:
            self.progress_stage = "Analysing Source"
        elif "planned candidates" in l:
            self.progress_stage = "Planning Candidates"
        elif "running a/b tournament" in l or "candidate 1" in l or "winner c" in l:
            self.progress_stage = "A/B Tournament & Judging"
        elif "rendering with" in l:
            self.progress_stage = "Rendering Video"
        elif "comparison video" in l:
            self.progress_stage = "Building Comparison"
        elif "done in" in l:
            self.progress_stage = "Completed"

        # Check for run directory / output / compare
        m_run = re.search(r"data[\\/]runs[\\/]([A-Za-z0-9\-_]+)", line)
        if m_run and not self.run_id:
            self.run_id = m_run.group(1)

        m_out = re.search(r"Output:\s*([^\r\n]+)", line)
        if m_out:
            self.output_video = m_out.group(1).strip()

        m_cmp = re.search(r"Comparison video:\s*([^\r\n]+)", line)
        if m_cmp:
            self.compare_video = m_cmp.group(1).strip()

        # Push to async queues
        for q in list(self.subscribers):
            try:
                q.put_nowait(line)
            except Exception:
                pass

    def to_dict(self) -> dict:
        duration = (self.end_time or time.time()) - self.start_time
        return {
            "id": self.id,
            "command": self.command,
            "status": self.status,
            "progress_stage": self.progress_stage,
            "input_path": self.input_path,
            "meta": self.meta,
            "duration": round(duration, 1),
            "duration_fmt": fmt_time(duration),
            "run_id": self.run_id,
            "output_video": self.output_video,
            "compare_video": self.compare_video,
            "error": self.error,
            "logs_count": len(self.logs),
        }


jobs_lock = threading.Lock()
jobs: Dict[str, Job] = {}


def run_job_worker(job: Job):
    job.status = "running"
    job.start_time = time.time()
    job.add_log(f"[{time.strftime('%H:%M:%S')}] Job started: {job.command}")
    job.add_log(f"[{time.strftime('%H:%M:%S')}] Command: {' '.join(job.cmd_args)}")

    try:
        # Use python 3.11 executable
        proc = subprocess.Popen(
            job.cmd_args,
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
        job.process = proc

        if proc.stdout:
            for raw_line in iter(proc.stdout.readline, ""):
                line = raw_line.rstrip()
                if line:
                    job.add_log(line)
            proc.stdout.close()

        proc.wait()
        job.end_time = time.time()

        if job.status == "cancelled":
            job.add_log(f"[{time.strftime('%H:%M:%S')}] Job was cancelled by user.")
        elif proc.returncode == 0:
            job.status = "completed"
            job.progress_stage = "Completed"
            job.add_log(f"[{time.strftime('%H:%M:%S')}] Job finished successfully.")
        else:
            job.status = "failed"
            job.progress_stage = "Failed"
            job.error = f"Process exited with code {proc.returncode}"
            job.add_log(f"[{time.strftime('%H:%M:%S')}] Job failed with exit code {proc.returncode}")

    except Exception as e:
        job.status = "failed"
        job.progress_stage = "Error"
        job.error = str(e)
        job.add_log(f"[{time.strftime('%H:%M:%S')}] Unexpected error: {e}")
    finally:
        job.end_time = time.time()
        # Signal EOF to subscribers
        for q in list(job.subscribers):
            try:
                q.put_nowait(None)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
@app.get("/api/system")
def get_system():
    reg = Registry()
    kb = Knowledge()
    runs_count = len(list(RUNS_DIR.glob("*"))) if RUNS_DIR.exists() else 0
    return {
        "engine": str(reg.engine),
        "engine_kind": reg.engine_kind,
        "engine_tag": reg.data.get("engine", {}).get("tag", "custom"),
        "models_count": len(reg.data.get("models", {})),
        "watchlist_count": len(reg.data.get("watchlist", [])),
        "runs_count": runs_count,
        "learned_categories": list(kb.d.get("categories", {}).keys()),
        "iqa_available": iqa_available(),
    }


@app.get("/api/models")
def get_models():
    reg = Registry()
    kb = Knowledge()
    models_list = []
    for mid, m in sorted(reg.data.get("models", {}).items()):
        speed = kb.d.get("speed", {}).get(mid)
        models_list.append({
            "id": mid,
            "status": m.get("status", "unknown"),
            "domain": m.get("domain", "live"),
            "speed": m.get("speed", "slow"),
            "licence": m.get("licence", "unknown"),
            "scale": m.get("scale", 4),
            "handles": m.get("handles", []),
            "trials": m.get("trials", 0),
            "wins": m.get("wins", 0),
            "measured_speed": f"{speed:.3f} s/MP" if speed else "-",
            "notes": m.get("notes", ""),
        })

    return {
        "models": models_list,
        "watchlist": reg.data.get("watchlist", []),
        "weights": kb.d.get("judge_weights", {}),
    }


@app.post("/api/models/update")
def trigger_update_check():
    reg = Registry()
    events = reg.check_updates(force=True, allow_engine=True)
    return {"status": "ok", "events": events}


@app.get("/api/videos")
def list_available_videos():
    videos = []
    # Test directory
    for f in TEST_DIR.glob("*"):
        if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
            videos.append({
                "name": f.name,
                "path": str(f.resolve()),
                "group": "Test Folder",
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })
    # Uploads directory
    for f in UPLOADS_DIR.glob("*"):
        if f.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi", ".webm"):
            videos.append({
                "name": f.name,
                "path": str(f.resolve()),
                "group": "Uploaded",
                "size_mb": round(f.stat().st_size / (1024 * 1024), 2),
            })
    return {"videos": videos}


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    dest = UPLOADS_DIR / file.filename
    with open(dest, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {
        "status": "success",
        "filename": file.filename,
        "path": str(dest.resolve()),
        "size_mb": round(dest.stat().st_size / (1024 * 1024), 2),
    }


class ProbeRequest(BaseModel):
    video_path: str


@app.post("/api/probe")
def probe_and_analyze_video(req: ProbeRequest):
    src = Path(req.video_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {req.video_path}")

    try:
        info = probe(src)
        prof = analyze(src, info, analysis_filters)
        feas = feasibility(prof, 2160)
        geo_4k = geometry(info, prof, (3840, 2160), "scale", 4, True)

        return {
            "info": {
                "width": info["width"],
                "height": info["height"],
                "codec": info["codec"],
                "fps": float(info["fps"]),
                "duration": info["duration"],
                "duration_fmt": fmt_time(info["duration"]),
                "bitrate_kbps": round(info["bitrate"] / 1000, 1),
                "audio_tracks": len(info.get("audio", [])),
            },
            "profile": {
                "category": prof["category"],
                "content": prof["content"],
                "anime_score": round(prof["anime_score"], 2),
                "effective_short": round(prof["effective_short"], 1),
                "compression": prof["compression"],
                "noise": prof["noise"],
                "grain": prof["grain"],
                "motion": prof["motion"],
                "metrics": {
                    "sigma": round(prof["metrics"]["sigma"], 2),
                    "temporal_sigma": round(prof["metrics"]["temporal_sigma"], 2),
                    "dark_share": round(prof["metrics"]["dark_share"], 2),
                    "banding_score": round(prof["metrics"]["banding_score"], 2),
                    "face_count": prof["metrics"]["face_count"],
                },
                "diagnosis": prof["diagnosis"],
            },
            "feasibility": {
                "factor": round(feas["factor"], 2),
                "risk": feas["risk"],
            },
            "geometry_4k": {
                "display": geo_4k["disp"],
                "ai_in": geo_4k["ai_in"],
                "out": geo_4k["out"],
                "prescale": round(geo_4k["prescale"], 2),
            },
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


class StartJobRequest(BaseModel):
    input_path: str
    action_type: str = "run"  # "run", "tournament", "analyze", "sample_compare"
    target: str = "4k"        # "4k", "dci4k", "1440p", "1080p", or "WxH"
    mode: str = "scale"       # "scale", "fill", "pad"
    fallback: str = "auto"    # "auto", "never", "ask"
    candidates: int = 3
    trials: int = 1
    samples: int = 4
    model: Optional[str] = None
    allow_nc: bool = False
    quick: bool = False
    compare: bool = True
    codec: str = "x265"
    crf: int = 16
    start: float = 0.0
    duration: Optional[float] = None


@app.post("/api/jobs")
def create_job(req: StartJobRequest):
    src = Path(req.input_path)
    if not src.exists():
        raise HTTPException(status_code=404, detail=f"Video file not found: {req.input_path}")

    job_id = f"job-{uuid.uuid4().hex[:8]}"

    # Build command line for upscale.py
    cmd = [sys.executable, "upscale.py"]

    if req.action_type == "analyze":
        cmd += ["analyze", str(src)]
    else:
        cmd += [str(src)]

        if req.action_type == "tournament":
            cmd += ["--plan-only"]
        elif req.action_type == "sample_compare":
            start_sec = req.start or 0.0
            dur_sec = req.duration or 10.0
            cmd += ["--start", str(start_sec), "--duration", str(dur_sec), "--compare"]
        else:
            # Full run
            if req.compare:
                cmd += ["--compare"]
            if req.start > 0:
                cmd += ["--start", str(req.start)]
            if req.duration and req.duration > 0:
                cmd += ["--duration", str(req.duration)]

        cmd += ["--target", req.target]
        cmd += ["--mode", req.mode]
        cmd += ["--fallback", req.fallback]
        cmd += ["--candidates", str(req.candidates)]
        cmd += ["--trials", str(req.trials)]
        cmd += ["--samples", str(req.samples)]
        cmd += ["--codec", req.codec]
        cmd += ["--crf", str(req.crf)]

        if req.model and req.model.strip():
            cmd += ["--model", req.model.strip()]
        if req.allow_nc:
            cmd += ["--allow-nc"]
        if req.quick:
            cmd += ["--quick"]

    job = Job(job_id, req.action_type, cmd, str(src), req.model_dump())
    with jobs_lock:
        jobs[job_id] = job

    # Launch background thread
    t = threading.Thread(target=run_job_worker, args=(job,), daemon=True)
    t.start()

    return {"job_id": job_id, "status": "started"}


@app.get("/api/jobs")
def list_jobs():
    with jobs_lock:
        return {"jobs": [j.to_dict() for j in reversed(list(jobs.values()))]}


@app.get("/api/jobs/{job_id}")
def get_job_status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    data = job.to_dict()
    data["logs"] = job.logs[-200:]
    return data


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job.process and job.status == "running":
        job.status = "cancelled"
        try:
            job.process.terminate()
        except Exception:
            try:
                job.process.kill()
            except Exception:
                pass
        return {"status": "cancelled"}
    return {"status": "not running"}


@app.get("/api/jobs/{job_id}/stream")
async def stream_job_logs(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    queue = asyncio.Queue()
    # Send existing logs
    for l in job.logs:
        queue.put_nowait(l)
    job.subscribers.append(queue)

    async def event_generator():
        try:
            while True:
                line = await queue.get()
                if line is None:
                    # Job finished
                    yield f"event: done\ndata: {json.dumps({'status': job.status, 'run_id': job.run_id})}\n\n"
                    break
                yield f"event: log\ndata: {line}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            if queue in job.subscribers:
                job.subscribers.remove(queue)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/runs")
def list_runs():
    runs = []
    if not RUNS_DIR.exists():
        return {"runs": []}

    for d in sorted(RUNS_DIR.glob("*"), reverse=True):
        if not d.is_dir():
            continue
        rep_path = d / "report.json"
        sheet_path = d / "tournament_sheet.png"
        data = load_json(rep_path, {})
        has_sheet = sheet_path.exists()

        winner = "Unknown"
        best_score = 0
        if "tournament" in data and "best" in data["tournament"]:
            best = data["tournament"]["best"]
            winner = f"{best.get('id', '')} {best.get('label', '')}"
            best_score = best.get("score", 0)
        elif "final_strategy" in data:
            winner = data["final_strategy"].get("model", "Default")

        runs.append({
            "id": d.name,
            "source": data.get("source", "Unknown"),
            "category": data.get("category", data.get("profile", {}).get("category", "Unknown")),
            "decision": data.get("decision", "Unknown"),
            "winner": winner,
            "best_score": round(best_score, 1) if best_score else None,
            "has_sheet": has_sheet,
            "output": data.get("output"),
            "render_seconds": data.get("render_seconds"),
            "total_seconds": data.get("total_seconds"),
        })
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run_details(run_id: str):
    run_dir = RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail="Run not found")

    rep_json = load_json(run_dir / "report.json", {})
    rep_md_path = run_dir / "report.md"
    rep_md = rep_md_path.read_text(encoding="utf-8") if rep_md_path.exists() else ""
    has_sheet = (run_dir / "tournament_sheet.png").exists()

    return {
        "id": run_id,
        "data": rep_json,
        "markdown": rep_md,
        "has_sheet": has_sheet,
    }


@app.get("/api/runs/{run_id}/sheet")
def get_tournament_sheet(run_id: str):
    sheet = RUNS_DIR / run_id / "tournament_sheet.png"
    if not sheet.exists():
        raise HTTPException(status_code=404, detail="Tournament sheet not found")
    return FileResponse(sheet, media_type="image/png")


class FeedbackRequest(BaseModel):
    run_id: str
    preferred: Optional[str] = None
    rating: Optional[int] = None


@app.post("/api/feedback")
def submit_feedback(req: FeedbackRequest):
    run_dir = RUNS_DIR / req.run_id
    rep = load_json(run_dir / "report.json", None)
    if not rep:
        raise HTTPException(status_code=404, detail="Run report not found")

    kb = Knowledge()
    reg = Registry()
    events = kb.record_feedback(rep, req.preferred, req.rating, reg)
    return {"status": "success", "events": events}


@app.get("/api/media")
def stream_media(path: str):
    p = Path(path).resolve()
    if not p.exists():
        raise HTTPException(status_code=404, detail="Media file not found")
    # Determine mime
    ext = p.suffix.lower()
    mimes = {
        ".mp4": "video/mp4",
        ".webm": "video/webm",
        ".mkv": "video/x-matroska",
        ".mov": "video/quicktime",
        ".png": "image/png",
        ".jpg": "image/jpeg",
    }
    return FileResponse(p, media_type=mimes.get(ext, "application/octet-stream"))


# ---------------------------------------------------------------------------
# Telegram Bot & Invite Codes API
# ---------------------------------------------------------------------------
class GenerateCodeRequest(BaseModel):
    note: str = ""
    max_uses: int = 1
    is_admin: bool = False


class RevokeCodeRequest(BaseModel):
    code: str


class RevokeUserRequest(BaseModel):
    user_id: str


@app.get("/api/telegram/status")
def get_telegram_status():
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    return {
        "configured": bool(token),
        "master_code": auth_manager.get_master_code(),
        "total_codes": len(auth_manager.list_codes()),
        "total_users": len(auth_manager.list_users()),
    }


@app.get("/api/telegram/codes")
def get_telegram_codes():
    return {
        "master_code": auth_manager.get_master_code(),
        "codes": auth_manager.list_codes(),
    }


@app.post("/api/telegram/codes")
def create_telegram_code(req: GenerateCodeRequest):
    code = auth_manager.generate_code(
        note=req.note, max_uses=req.max_uses, is_admin=req.is_admin, created_by="web_dashboard"
    )
    return {"status": "success", "code": code}


@app.post("/api/telegram/codes/revoke")
def revoke_telegram_code(req: RevokeCodeRequest):
    success = auth_manager.revoke_code(req.code)
    return {"status": "success" if success else "not_found"}


@app.get("/api/telegram/users")
def get_telegram_users():
    return {"users": auth_manager.list_users()}


@app.post("/api/telegram/users/revoke")
def revoke_telegram_user(req: RevokeUserRequest):
    success = auth_manager.revoke_user(req.user_id)
    return {"status": "success" if success else "not_found"}


# ---------------------------------------------------------------------------
# UI Entry Point
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def index():
    html_path = ROOT / "web_ui.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Upscale Pro 4K Web Server Running</h1>")


def start_server(host: str = "127.0.0.1", port: int = 8000):
    print(f"Starting Upscale Pro 4K Web Server on http://{host}:{port} ...")
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    start_server(host=args.host, port=args.port)
