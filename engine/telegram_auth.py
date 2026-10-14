"""Telegram Access Control & Secret Invite Code Manager for Upscale Pro 4K."""
from __future__ import annotations

import json
import os
import secrets
import string
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from .common import DATA_DIR, load_json, log, save_json

load_dotenv()

AUTH_FILE = DATA_DIR / "telegram_auth.json"


def _gen_random_code(prefix: str = "UPSCALE-", length: int = 6) -> str:
    alphabet = string.ascii_uppercase + string.digits
    # Avoid confusing characters like O and 0, I and 1
    safe_chars = [c for c in alphabet if c not in "OI01"]
    token = "".join(secrets.choice(safe_chars) for _ in range(length))
    return f"{prefix}{token}"


class TelegramAuthManager:
    def __init__(self, auth_path: Path = AUTH_FILE):
        self.path = auth_path
        self.data: Dict[str, Any] = self._load_or_init()

    def _load_or_init(self) -> Dict[str, Any]:
        default = {
            "master_code": _gen_random_code(prefix="UPSCALE-MASTER-", length=8),
            "admin_users": [],
            "valid_codes": {},
            "authorized_users": {},
        }
        loaded = load_json(self.path, None)
        if not loaded or not isinstance(loaded, dict):
            # First initialization
            save_json(self.path, default)
            log(f"[Telegram Auth] Initialized new auth database. Master Code: {default['master_code']}")
            loaded = default

        # Ensure all required keys exist
        for k, v in default.items():
            if k not in loaded:
                loaded[k] = v

        # Check if an admin chat ID is pre-configured in environment
        admin_env = os.getenv("TELEGRAM_ADMIN_CHAT_ID", "").strip()
        if admin_env and admin_env not in [str(u) for u in loaded.get("admin_users", [])]:
            loaded.setdefault("admin_users", []).append(admin_env)
            loaded.setdefault("authorized_users", {})[admin_env] = {
                "username": "Owner",
                "first_name": "Admin",
                "authorized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code_used": "ENV_PREAUTHORIZED",
                "is_admin": True,
            }
            save_json(self.path, loaded)

        return loaded

    def save(self) -> None:
        save_json(self.path, self.data)

    def get_master_code(self) -> str:
        return self.data.get("master_code", "")

    def is_user_authorized(self, user_id: int) -> bool:
        str_id = str(user_id)
        if str_id in self.data.get("authorized_users", {}):
            return True
        if str_id in [str(u) for u in self.data.get("admin_users", [])]:
            return True
        return False

    def is_user_admin(self, user_id: int) -> bool:
        str_id = str(user_id)
        if str_id in [str(u) for u in self.data.get("admin_users", [])]:
            return True
        user_info = self.data.get("authorized_users", {}).get(str_id, {})
        return bool(user_info.get("is_admin", False))

    def verify_and_authorize(
        self, user_id: int, username: Optional[str], first_name: Optional[str], code_entered: str
    ) -> Tuple[bool, str, bool]:
        """Verifies code and registers user.

        Returns: (success: bool, message: str, is_admin: bool)
        """
        code = code_entered.strip().upper()
        str_id = str(user_id)

        # Check Master Code
        if code == self.data.get("master_code", "").upper():
            if str_id not in self.data["admin_users"]:
                self.data["admin_users"].append(str_id)
            self.data["authorized_users"][str_id] = {
                "username": username or "",
                "first_name": first_name or "",
                "authorized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "code_used": code,
                "is_admin": True,
            }
            self.save()
            return True, "Master Access Code verified! You have full Administrator access.", True

        # Check standard generated codes
        codes_dict = self.data.get("valid_codes", {})
        if code not in codes_dict:
            return False, "Invalid access code. Please check for typos or ask the administrator.", False

        info = codes_dict[code]
        if info.get("revoked", False):
            return False, "This access code has been revoked by the administrator.", False

        max_uses = info.get("max_uses", 1)
        times_used = info.get("times_used", 0)
        if max_uses > 0 and times_used >= max_uses:
            return False, "This access code has already reached its maximum usage limit.", False

        # Valid code
        info["times_used"] = times_used + 1
        info.setdefault("used_by", []).append(str_id)
        is_admin = bool(info.get("is_admin", False))

        if is_admin and str_id not in self.data["admin_users"]:
            self.data["admin_users"].append(str_id)

        self.data["authorized_users"][str_id] = {
            "username": username or "",
            "first_name": first_name or "",
            "authorized_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "code_used": code,
            "is_admin": is_admin,
            "note": info.get("note", ""),
        }
        self.save()
        role = "Administrator" if is_admin else "Authorized User"
        return True, f"Code verified successfully! Welcome to Upscale Pro 4K as {role}.", is_admin

    def generate_code(
        self, note: str = "", max_uses: int = 1, is_admin: bool = False, created_by: str = "admin"
    ) -> str:
        """Generates a new unique access code and stores it."""
        prefix = "UPSCALE-ADMIN-" if is_admin else "UPSCALE-"
        code = _gen_random_code(prefix=prefix, length=6)
        while code in self.data.get("valid_codes", {}):
            code = _gen_random_code(prefix=prefix, length=6)

        self.data.setdefault("valid_codes", {})[code] = {
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "created_by": created_by,
            "note": note,
            "max_uses": max_uses,
            "times_used": 0,
            "is_admin": is_admin,
            "revoked": False,
            "used_by": [],
        }
        self.save()
        return code

    def list_codes(self) -> List[Dict[str, Any]]:
        result = []
        for code, info in self.data.get("valid_codes", {}).items():
            result.append({"code": code, **info})
        return result

    def list_users(self) -> List[Dict[str, Any]]:
        result = []
        for uid, info in self.data.get("authorized_users", {}).items():
            result.append({"user_id": uid, **info})
        return result

    def revoke_code(self, code: str) -> bool:
        c = code.strip().upper()
        if c in self.data.get("valid_codes", {}):
            self.data["valid_codes"][c]["revoked"] = True
            self.save()
            return True
        return False

    def revoke_user(self, user_id: int | str) -> bool:
        str_id = str(user_id)
        changed = False
        if str_id in self.data.get("authorized_users", {}):
            del self.data["authorized_users"][str_id]
            changed = True
        if str_id in self.data.get("admin_users", []):
            self.data["admin_users"].remove(str_id)
            changed = True
        if changed:
            self.save()
        return changed


# Global singleton instance
auth_manager = TelegramAuthManager()
