from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import secrets

import jwt

from app.config import ACCESS_TOKEN_MINUTES, APP_SECRET, REFRESH_TOKEN_DAYS

ALGO = "HS256"


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return digest, salt


def verify_password(password: str, digest: str, salt: str) -> bool:
    check, _ = hash_password(password, salt)
    return hmac.compare_digest(check, digest)


def create_access_token(subject: str, role: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_MINUTES)
    return jwt.encode({"sub": subject, "role": role, "type": "access", "exp": exp}, APP_SECRET, algorithm=ALGO)


def create_refresh_token(subject: str) -> str:
    exp = datetime.now(timezone.utc) + timedelta(days=REFRESH_TOKEN_DAYS)
    jti = secrets.token_hex(16)
    return jwt.encode({"sub": subject, "type": "refresh", "jti": jti, "exp": exp}, APP_SECRET, algorithm=ALGO)


def decode_token(token: str) -> dict:
    return jwt.decode(token, APP_SECRET, algorithms=[ALGO])
