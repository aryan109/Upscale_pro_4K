"""Upscale Pro 4K - Telegram Bot Service.

Provides remote AI video upscaling through Telegram with:
- Secret invite code authentication (role-based: Admin vs Authorized User)
- Real-time system live status checks
- Video & document file ingestion (with automatic fake HD & resolution analysis)
- Interactive quality & pipeline selection (4K UHD, 1080p FHD, 10s Sample Comparison)
- Background queue with Vulkan GPU engine acceleration
- Delivery of results strictly in Telegram file/document format
"""
from __future__ import annotations

import argparse
import asyncio
import os
import re
import shutil
import subprocess
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from telegram import (
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Update,
)
from telegram.constants import ChatAction, ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from engine.analyze import analyze, feasibility
from engine.common import DATA_DIR, ROOT, RUNS_DIR, fmt_time, log
from engine.probe import probe
from engine.registry import Registry
from engine.strategy import analysis_filters, geometry
from engine.telegram_auth import auth_manager

load_dotenv()

TELEGRAM_UPLOADS_DIR = ROOT / "uploads" / "telegram"
TELEGRAM_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# Async job queue to prevent GPU memory contention
job_queue = asyncio.Queue()
active_job_info: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_bot_token() -> Optional[str]:
    # Check environment or .env
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip().strip('"').strip("'")
    bot_id = (os.getenv("TELEGRAM_BOT_ID") or os.getenv("TELEGRAM_ADMIN_CHAT_ID", "")).strip().strip('"').strip("'")

    if not token:
        env_path = ROOT / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    token = line.split("=", 1)[1].strip().strip('"').strip("'")
                elif line.startswith("TELEGRAM_BOT_ID="):
                    bot_id = line.split("=", 1)[1].strip().strip('"').strip("'")

    if token:
        # If token does not contain ':', but bot_id exists and is numeric, combine them
        if ":" not in token and bot_id and bot_id.isdigit():
            token = f"{bot_id}:{token}"
        return token
    return None


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-\.]+", "_", name)


# ---------------------------------------------------------------------------
# Command Handlers
# ---------------------------------------------------------------------------
async def start_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return

    user_id = user.id
    if not auth_manager.is_user_authorized(user_id):
        await update.message.reply_text(
            "🔒 *Upscale Pro 4K — Private Access*\n\n"
            "This AI video upscaling system is invite-only.\n"
            "Please send your *secret access code* to activate your account.\n\n"
            "_(If you received a code, simply reply with it here!)_",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    is_admin = auth_manager.is_user_admin(user_id)
    role_str = "⭐ Administrator" if is_admin else "✨ Authorized User"
    msg = (
        f"👋 Welcome *{user.first_name}* ({role_str})!\n\n"
        "🎮 *Upscale Pro 4K System is Live & Ready.*\n\n"
        "📹 *How to use:*\n"
        "1. Send any video or video document here.\n"
        "2. The AI will analyze true detail and compression.\n"
        "3. Choose your target (4K UHD, 1080p, or 10s Sample).\n"
        "4. Your upscaled video will be returned as an uncompressed file!\n\n"
        "Commands:\n"
        "• `/status` - Check GPU engine & queue status\n"
        "• `/help` - Usage tips & instructions\n"
    )
    if is_admin:
        msg += (
            "\n👑 *Admin Commands:*\n"
            "• `/newcode [note]` - Generate a new invite code\n"
            "• `/listcodes` - List all invite codes\n"
            "• `/users` - List authorized users\n"
            "• `/revoke <code>` - Revoke an access code\n"
        )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


async def help_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_authorized(user.id):
        await start_handler(update, context)
        return

    text = (
        "💡 *Upscale Pro 4K Telegram Guide*\n\n"
        "• *Video Uploads:* Send videos either as Video clips or as Documents. We recommend sending as a *File / Document* to avoid Telegram's re-compression.\n"
        "• *File Size Notice:* Telegram Bot API allows downloading files up to *20MB*. For larger videos, use our Local Web Studio at `http://127.0.0.1:8000`.\n"
        "• *File Delivery:* All upscaled videos are returned strictly in *Document file format* so full 4K visual quality and metadata are preserved.\n"
        "• *A/B Tournament:* The engine automatically tests multiple models against the hardest scenes of your video to ensure the sharpest result."
    )
    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN)


async def status_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_authorized(user.id):
        await start_handler(update, context)
        return

    reg = Registry()
    qsize = job_queue.qsize()
    busy_text = f"🔄 Processing a video ({qsize} in queue)" if active_job_info else "🟢 Idle (Ready for new jobs)"

    msg = (
        "📊 *Upscale Pro 4K System Status*\n\n"
        f"• *Status:* 🟢 *ONLINE & LIVE*\n"
        f"• *Engine:* `Upscayl Vulkan NCNN` (GPU Accelerated)\n"
        f"• *Engine Version:* `{reg.data.get('engine', {}).get('tag', 'vulkan')}`\n"
        f"• *Installed Models:* `{len(reg.data.get('models', {}))}` models\n"
        f"• *Queue Status:* {busy_text}\n"
        f"• *Local Server:* `http://127.0.0.1:8000`"
    )
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Admin Code Management Commands
# ---------------------------------------------------------------------------
async def newcode_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_admin(user.id):
        await update.message.reply_text("⛔ This command is restricted to administrators.")
        return

    note = " ".join(context.args) if context.args else "General Invite"
    code = auth_manager.generate_code(note=note, max_uses=1, is_admin=False, created_by=f"@{user.username or user.id}")

    reply = (
        "🎟️ *New Invite Code Generated!*\n\n"
        f"Code: `{code}`\n"
        f"Note: `{note}`\n"
        "Max Uses: `1`\n\n"
        "📋 *Shareable message:*\n"
        f"Here is your invite code for Upscale Pro 4K:\n`{code}`\n"
        "Send this code to the bot to activate your access!"
    )
    await update.message.reply_text(reply, parse_mode=ParseMode.MARKDOWN)


async def listcodes_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_admin(user.id):
        await update.message.reply_text("⛔ This command is restricted to administrators.")
        return

    codes = auth_manager.list_codes()
    if not codes:
        await update.message.reply_text("No invite codes generated yet. Use `/newcode` to generate one.")
        return

    lines = ["📑 *Active & Past Invite Codes:*"]
    for c in codes:
        status_icon = "❌ Revoked" if c.get("revoked") else ("✅ Used" if c.get("times_used", 0) >= c.get("max_uses", 1) else "🟢 Active")
        lines.append(f"• `{c['code']}` — {status_icon} ({c.get('note', 'No note')}) [Used: {c.get('times_used')}/{c.get('max_uses')}]")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def users_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_admin(user.id):
        await update.message.reply_text("⛔ This command is restricted to administrators.")
        return

    user_list = auth_manager.list_users()
    if not user_list:
        await update.message.reply_text("No authorized users registered yet.")
        return

    lines = ["👥 *Authorized Users List:*"]
    for u in user_list:
        role = "⭐ Admin" if u.get("is_admin") else "User"
        uname = f"@{u['username']}" if u.get("username") else u.get("first_name", "Unknown")
        lines.append(f"• ID: `{u['user_id']}` ({uname}) — {role} [Auth: {u.get('authorized_at', '-')}]")

    await update.message.reply_text("\n".join(lines), parse_mode=ParseMode.MARKDOWN)


async def revoke_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user or not auth_manager.is_user_admin(user.id):
        await update.message.reply_text("⛔ This command is restricted to administrators.")
        return

    if not context.args:
        await update.message.reply_text("Usage: `/revoke <UPSCALE-CODE or USER_ID>`", parse_mode=ParseMode.MARKDOWN)
        return

    target = context.args[0].strip()
    if target.upper().startswith("UPSCALE-"):
        if auth_manager.revoke_code(target):
            await update.message.reply_text(f"✅ Code `{target}` has been revoked.", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"Code `{target}` not found.", parse_mode=ParseMode.MARKDOWN)
    else:
        if auth_manager.revoke_user(target):
            await update.message.reply_text(f"✅ User ID `{target}` access has been revoked.", parse_mode=ParseMode.MARKDOWN)
        else:
            await update.message.reply_text(f"User ID `{target}` not found in authorized records.", parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Message & Text Authentication Handler
# ---------------------------------------------------------------------------
async def text_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return

    text = update.message.text.strip() if update.message and update.message.text else ""
    user_id = user.id

    # If user is not authorized, check if text is an access code
    if not auth_manager.is_user_authorized(user_id):
        ok, msg, is_admin = auth_manager.verify_and_authorize(
            user_id=user_id,
            username=user.username,
            first_name=user.first_name,
            code_entered=text,
        )
        if ok:
            role_text = "⭐ Administrator" if is_admin else "✨ Authorized User"
            await update.message.reply_text(
                f"🎉 *{msg}*\n\n"
                f"Your account is now verified as *{role_text}*.\n\n"
                "You can now send any video or video document to upscale to 4K!",
                parse_mode=ParseMode.MARKDOWN,
            )
        else:
            await update.message.reply_text(
                f"❌ {msg}\n\n"
                "Please enter a valid invite code to unlock the service.",
            )
        return

    # Authorized user sent random text
    await update.message.reply_text(
        "📹 Send me a video file (or document) and I will analyze and upscale it to 4K for you!\n"
        "Use `/status` to inspect system availability.",
        parse_mode=ParseMode.MARKDOWN,
    )


# ---------------------------------------------------------------------------
# Video Ingestion & Processing
# ---------------------------------------------------------------------------
async def media_file_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user = update.effective_user
    if not user:
        return

    user_id = user.id
    if not auth_manager.is_user_authorized(user_id):
        await update.message.reply_text(
            "🔒 *Access Restricted*\n\n"
            "Please send your invite code first to activate your account.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    message = update.message
    file_obj = None
    file_name = "video.mp4"
    file_size = 0

    if message.video:
        file_obj = message.video
        file_name = file_obj.file_name or f"video_{file_obj.file_unique_id}.mp4"
        file_size = file_obj.file_size or 0
    elif message.document:
        doc = message.document
        ext = Path(doc.file_name or "").suffix.lower()
        if ext in (".mp4", ".mov", ".mkv", ".avi", ".webm", ".flv", ".m4v") or (doc.mime_type and doc.mime_type.startswith("video/")):
            file_obj = doc
            file_name = doc.file_name or f"document_{doc.file_unique_id}.mp4"
            file_size = doc.file_size or 0
        else:
            await message.reply_text("⚠️ Please send a video file (MP4, MOV, MKV, WebM, etc.).")
            return

    if not file_obj:
        return

    # Telegram Bot API 20MB download limit check
    max_download_bytes = 20 * 1024 * 1024
    if file_size > max_download_bytes:
        size_mb = round(file_size / (1024 * 1024), 1)
        await message.reply_text(
            f"⚠️ *File Size Limit Exceeded*\n\n"
            f"Telegram Bot API limits incoming file downloads to *20MB* for bots.\n"
            f"Your file is *{size_mb} MB*.\n\n"
            "💡 *Tips:*\n"
            "• You can send a shorter clip or trimmed sample under 20MB.\n"
            "• For large full-length video files, upload directly through our Local Web Studio at `http://127.0.0.1:8000`.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    status_msg = await message.reply_text("📥 *Downloading file from Telegram...*", parse_mode=ParseMode.MARKDOWN)

    # Download file
    try:
        tg_file = await context.bot.get_file(file_obj.file_id)
        local_name = f"{int(time.time())}_{sanitize_filename(file_name)}"
        local_path = TELEGRAM_UPLOADS_DIR / local_name

        await tg_file.download_to_drive(custom_path=local_path)
    except Exception as e:
        await status_msg.edit_text(f"❌ Failed to download file from Telegram: {e}")
        return

    # Probe & Analyze Video
    await status_msg.edit_text("🔍 *Analyzing video metrics & 4K feasibility...*", parse_mode=ParseMode.MARKDOWN)

    try:
        info = probe(local_path)
        prof = analyze(local_path, info, analysis_filters)
        feas = feasibility(prof, 2160)

        # Build interactive card
        risk_emojis = {"low": "🟢", "medium": "🟡", "high": "🟠", "extreme": "🔴"}
        risk_icon = risk_emojis.get(feas["risk"], "⚪")

        diag_text = (
            f"🎬 *Video Diagnostic Completed*\n\n"
            f"• *File:* `{file_name}`\n"
            f"• *Source:* `{info['width']}x{info['height']}` @ `{float(info['fps']):.2f} fps` ({fmt_time(info['duration'])})\n"
            f"• *True Detail:* `~{prof['effective_short']:.0f}p` "
            f"({'⚠️ Fake HD' if (info['height'] - prof['effective_short']) > 100 else 'Genuine resolution'})\n"
            f"• *Content:* `{prof['content']}` (Category: `{prof['category']}`)\n"
            f"• *4K Feasibility:* {risk_icon} *{feas['risk'].upper()} RISK* (`{feas['factor']:.1f}x` enlargement)\n\n"
            "👇 *Select Upscaling Mode:*"
        )

        # Keyboard with options
        keyboard = [
            [
                InlineKeyboardButton("🏆 4K UHD Master", callback_data=f"up:4k:{local_path.name}"),
                InlineKeyboardButton("🎬 1080p Full HD", callback_data=f"up:1080p:{local_path.name}"),
            ],
            [
                InlineKeyboardButton("🔬 10s Sample Comparison", callback_data=f"up:sample:{local_path.name}"),
                InlineKeyboardButton("⚡ Fast 4K (Top Pick)", callback_data=f"up:quick:{local_path.name}"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await status_msg.edit_text(diag_text, reply_markup=reply_markup, parse_mode=ParseMode.MARKDOWN)

    except Exception as e:
        await status_msg.edit_text(f"❌ Analysis failed: {e}")


# ---------------------------------------------------------------------------
# Callback Query Handler (Button Click)
# ---------------------------------------------------------------------------
async def callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    data = query.data
    if not data or not data.startswith("up:"):
        return

    parts = data.split(":", 2)
    mode_choice = parts[1]
    filename = parts[2]
    file_path = TELEGRAM_UPLOADS_DIR / filename

    if not file_path.exists():
        await query.edit_message_text("❌ Source file is no longer available on server.")
        return

    # Enqueue job
    job = {
        "chat_id": query.message.chat_id,
        "message_id": query.message.message_id,
        "file_path": file_path,
        "mode_choice": mode_choice,
        "filename": filename,
    }
    await job_queue.put(job)
    qsize = job_queue.qsize()

    wait_text = f"⏳ *Added to processing queue* (Position: {qsize})...\nThe GPU will start rendering shortly!"
    await query.edit_message_text(wait_text, parse_mode=ParseMode.MARKDOWN)


# ---------------------------------------------------------------------------
# Background Worker: Video Upscaling Queue
# ---------------------------------------------------------------------------
async def queue_worker(application: Application):
    global active_job_info
    log("[Telegram Bot] Video processing queue worker started.")

    while True:
        job = await job_queue.get()
        active_job_info = job
        chat_id = job["chat_id"]
        msg_id = job["message_id"]
        src = job["file_path"]
        mode = job["mode_choice"]

        log(f"[Telegram Bot] Processing job: {src.name} [mode={mode}] for chat {chat_id}")

        try:
            # Inform user processing started
            await application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="⚙️ *AI Upscaling in progress...*\n\n"
                     "1️⃣ Profiling source scenes\n"
                     "2️⃣ A/B Model Tournament (evaluating sharpest candidate)\n"
                     "3️⃣ Vulkan GPU Neural Render\n\n"
                     "_(Please wait, this may take a few moments...)_",
                parse_mode=ParseMode.MARKDOWN,
            )

            # Build command
            cmd = [sys.executable, "upscale.py", str(src)]

            is_sample = False
            if mode == "sample":
                cmd += ["--start", "0", "--duration", "10", "--compare", "--target", "4k"]
                is_sample = True
            elif mode == "1080p":
                cmd += ["--target", "1080p", "--fallback", "never"]
            elif mode == "quick":
                cmd += ["--quick", "--target", "4k", "--fallback", "auto"]
            else:
                # 4k
                cmd += ["--target", "4k", "--fallback", "auto"]

            # Run process
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=str(ROOT),
            )

            output_file: Optional[Path] = None
            compare_file: Optional[Path] = None
            winner_info = "AI Pipeline"

            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").strip()
                if not decoded:
                    continue

                if "Output:" in decoded:
                    m_out = re.search(r"Output:\s*([^\r\n]+)", decoded)
                    if m_out:
                        output_file = Path(m_out.group(1).strip())
                if "Comparison video:" in decoded:
                    m_cmp = re.search(r"Comparison video:\s*([^\r\n]+)", decoded)
                    if m_cmp:
                        compare_file = Path(m_cmp.group(1).strip())
                if "WINNER" in decoded:
                    winner_info = decoded.replace("WINNER", "").strip()

            await proc.wait()

            if proc.returncode != 0 or not output_file or not output_file.exists():
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"❌ Upscaling failed with process return code {proc.returncode}.",
                )
                continue

            # Return the upscaled video in Document/File format
            await application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="📤 *Render complete! Uploading uncompressed 4K file to Telegram...*",
                parse_mode=ParseMode.MARKDOWN,
            )

            # Check output file size for Telegram 50MB ceiling
            file_bytes = output_file.stat().st_size
            file_mb = round(file_bytes / (1024 * 1024), 2)
            caption = f"✨ Upscaled Result ({winner_info})\nResolution: 4K UHD\nDelivered in uncompressed file format."

            # Send typing action
            await application.bot.send_chat_action(chat_id=chat_id, action=ChatAction.UPLOAD_DOCUMENT)

            if file_bytes <= 50 * 1024 * 1024:
                # Send directly as document file
                with open(output_file, "rb") as f:
                    await application.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=output_file.name,
                        caption=caption,
                        read_timeout=300,
                        write_timeout=300,
                    )
            else:
                # If master > 50MB, transcode a Telegram-friendly compact delivery document
                compact_file = output_file.with_name(f"{output_file.stem}_tg.mp4")
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"📦 Master file is {file_mb} MB (exceeds Telegram 50MB limit).\n"
                         "Generating high-efficiency compact file for Telegram delivery...",
                )
                # Optimize to fit under 48MB
                sub_cmd = [
                    "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                    "-i", str(output_file),
                    "-c:v", "libx265", "-crf", "22", "-preset", "faster",
                    "-c:a", "copy",
                    str(compact_file),
                ]
                subprocess.run(sub_cmd, check=False)
                send_path = compact_file if compact_file.exists() and compact_file.stat().st_size <= 50 * 1024 * 1024 else output_file

                with open(send_path, "rb") as f:
                    await application.bot.send_document(
                        chat_id=chat_id,
                        document=f,
                        filename=output_file.name,
                        caption=f"{caption}\n(Master size: {file_mb}MB)",
                        read_timeout=300,
                        write_timeout=300,
                    )

            # If comparison video exists, send it too as document
            if compare_file and compare_file.exists() and compare_file.stat().st_size <= 50 * 1024 * 1024:
                with open(compare_file, "rb") as cf:
                    await application.bot.send_document(
                        chat_id=chat_id,
                        document=cf,
                        filename=compare_file.name,
                        caption="🔬 Split-Screen Comparison (Original vs AI Upscaled)",
                        read_timeout=300,
                        write_timeout=300,
                    )

            await application.bot.edit_message_text(
                chat_id=chat_id,
                message_id=msg_id,
                text="✅ *All files delivered successfully!*",
                parse_mode=ParseMode.MARKDOWN,
            )

        except Exception as e:
            log(f"[Telegram Bot] Job error: {e}")
            try:
                await application.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=f"❌ Error during upscaling: {str(e)[:200]}",
                )
            except Exception:
                pass
        finally:
            active_job_info = None
            job_queue.task_done()


# ---------------------------------------------------------------------------
# Main Bot Application Runner
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Upscale Pro 4K Telegram Bot")
    parser.add_argument("--token", help="Telegram Bot Token")
    parser.add_argument("--gen-code", help="Generate a new access code with note and exit")
    parser.add_argument("--list-codes", action="store_true", help="List all generated access codes and exit")
    parser.add_argument("--master-code", action="store_true", help="Print the master admin access code and exit")
    args = parser.parse_args()

    # CLI helpers
    if args.master_code:
        print(f"Master Admin Access Code: {auth_manager.get_master_code()}")
        return

    if args.gen_code:
        code = auth_manager.generate_code(note=args.gen_code, max_uses=1)
        print(f"Generated Code: {code} (Note: {args.gen_code})")
        return

    if args.list_codes:
        codes = auth_manager.list_codes()
        print(f"Total codes: {len(codes)}")
        for c in codes:
            print(f"  {c['code']} - uses: {c.get('times_used')}/{c.get('max_uses')} - note: {c.get('note')}")
        return

    token = args.token or get_bot_token()
    if not token:
        print("=" * 70)
        print(" [!] TELEGRAM BOT TOKEN REQUIRED")
        print("=" * 70)
        print("To connect Upscale Pro 4K to Telegram:")
        print("1. Open Telegram and message @BotFather")
        print("2. Send /newbot and follow the steps to get your HTTP API token")
        print("3. Add the token to your .env file:")
        print("     TELEGRAM_BOT_TOKEN=123456789:ABCdefGhIJKlmNoPQRstuVWXyz")
        print("   OR run:")
        print("     py -3.11 telegram_bot.py --token YOUR_TOKEN")
        print("=" * 70)
        print(f"\nYour Master Admin Access Code is: {auth_manager.get_master_code()}\n")
        return

    print("=" * 70)
    print("      🚀 Starting Upscale Pro 4K Telegram Bot Service")
    print("=" * 70)
    print(f"Master Admin Access Code: {auth_manager.get_master_code()}")
    print("Share this code (or generate new ones via /newcode) with authorized users.")
    print("=" * 70)

    # Initialize Application
    application = ApplicationBuilder().token(token).build()

    # Register Handlers
    application.add_handler(CommandHandler("start", start_handler))
    application.add_handler(CommandHandler("help", help_handler))
    application.add_handler(CommandHandler("status", status_handler))
    application.add_handler(CommandHandler("newcode", newcode_handler))
    application.add_handler(CommandHandler("gencode", newcode_handler))
    application.add_handler(CommandHandler("listcodes", listcodes_handler))
    application.add_handler(CommandHandler("users", users_handler))
    application.add_handler(CommandHandler("revoke", revoke_handler))

    application.add_handler(CallbackQueryHandler(callback_handler))
    application.add_handler(MessageHandler(filters.PHOTO | filters.VIDEO | filters.Document.ALL, media_file_handler))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_handler))

    # Start queue worker on post-init
    async def post_init(app: Application) -> None:
        asyncio.create_task(queue_worker(app))

    application.post_init = post_init

    # Run polling
    log("Connecting to Telegram Bot API...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
