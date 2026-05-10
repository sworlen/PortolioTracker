from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import os
import secrets

import jwt

SECRET_KEY = os.getenv("APP_SECRET", "dev-secret-change-me")
ALGO = "HS256"


def hash_password(password: str, salt: str | None = None) -> tuple[str, str]:
    salt = salt or secrets.token_hex(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 100_000).hex()
    return digest, salt


def verify_password(password: str, digest: str, salt: str) -> bool:
    check, _ = hash_password(password, salt)
    return hmac.compare_digest(check, digest)


def create_access_token(subject: str, minutes: int = 60) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=minutes)
    return jwt.encode({"sub": subject, "exp": exp}, SECRET_KEY, algorithm=ALGO)


def decode_token(token: str) -> dict:
    return jwt.decode(token, SECRET_KEY, algorithms=[ALGO])
