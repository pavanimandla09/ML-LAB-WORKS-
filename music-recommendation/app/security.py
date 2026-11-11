"""Password hashing and signed access-token helpers."""
from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import bcrypt
import jwt
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


JWT_SECRET = os.getenv("JWT_SECRET", "local-development-secret-change-before-deploy")
JWT_ALGORITHM = "HS256"
TOKEN_HOURS = 12


def hash_password(password: str) -> bytes:
    encoded = password.encode("utf-8")
    if len(encoded) > 72:
        raise ValueError("Passwords must be at most 72 UTF-8 bytes.")
    return bcrypt.hashpw(encoded, bcrypt.gensalt(rounds=12))


def verify_password(password: str, password_hash: bytes) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash)
    except (ValueError, TypeError):
        return False


def create_access_token(user_id: str, username: str) -> str:
    expires_at = datetime.now(timezone.utc) + timedelta(hours=TOKEN_HOURS)
    payload: dict[str, Any] = {
        "sub": user_id,
        "username": username,
        "exp": expires_at,
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    return jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
