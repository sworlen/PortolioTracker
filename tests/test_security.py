import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from app.security import create_access_token, create_refresh_token, decode_token, hash_password, verify_password


def test_password_roundtrip():
    digest, salt = hash_password("abc123")
    assert verify_password("abc123", digest, salt)


def test_tokens_decode():
    at = create_access_token("1", "user")
    rt = create_refresh_token("1")
    assert decode_token(at)["type"] == "access"
    assert decode_token(rt)["type"] == "refresh"
