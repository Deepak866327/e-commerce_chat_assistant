"""Local operations auth, redaction, rate limiting and retrieved-content checks."""

from dataclasses import dataclass
from hashlib import pbkdf2_hmac, sha256
import hmac
import os
import re
import secrets
import time


EMAIL = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d ()-]{7,}\d)(?!\w)")
SECRET = re.compile(r"(?i)(?:bearer\s+|(?:api[_ -]?key|password|token)\s*[:=]\s*)\S+")
IDENTIFIER = re.compile(r"(?i)\b(order|account|card|customer)(?:[ _-]*(?:id|number|no))?\s*[:#=-]\s*[\w-]{4,}")
INJECTION = re.compile(r"(?i)ignore\s+(?:all\s+)?(?:previous|system|prior)\s+instructions|"
                       r"(?:reveal|print|show)\s+(?:the\s+)?(?:system\s+prompt|api\s*key|credentials)|"
                       r"<\|(?:im_start|system)|\[INST\]|you are now\s+(?:a|an)\s+")


def redact(text):
    text = EMAIL.sub("[EMAIL]", str(text))
    text = SECRET.sub("[SECRET]", text)
    text = IDENTIFIER.sub("[IDENTIFIER]", text)
    return PHONE.sub("[PHONE]", text)


def redact_value(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: redact_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_value(v) for v in value]
    return value


def suspicious(text):
    return bool(INJECTION.search(text))


def password_hash(password, salt=None):
    if len(password) < 12:
        raise ValueError("Use at least 12 characters")
    salt = salt or secrets.token_hex(16)
    digest = pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), 600_000).hex()
    return f"pbkdf2_sha256$600000${salt}${digest}"


@dataclass(frozen=True)
class Principal:
    name: str
    role: str

    def require(self, permission):
        permissions = {
            "admin": {"read", "ingest", "review", "publish", "evaluate", "dataset", "delete", "activate", "backup"},
            "reviewer": {"read", "review", "dataset"},
            "evaluator": {"read", "evaluate"},
        }
        if permission not in permissions.get(self.role, set()):
            raise PermissionError(f"Role {self.role} cannot {permission}")


def authenticate(role, password):
    if role not in {"admin", "reviewer", "evaluator"}:
        raise PermissionError("Unknown role")
    value = os.getenv("SHOPASSIST_" + role.upper() + "_PASSWORD_HASH", "")
    try:
        algorithm, iterations, salt, expected = value.split("$")
        if algorithm != "pbkdf2_sha256" or int(iterations) != 600_000:
            raise ValueError()
        candidate = pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(iterations)).hex()
        if not hmac.compare_digest(candidate, expected):
            raise ValueError()
    except (ValueError, TypeError):
        raise PermissionError("Authentication failed; configure the role password hash") from None
    return Principal(role, role)


def rate_limit(store, subject, limit):
    minute = int(time.time() // 60)
    digest = sha256(subject.encode()).hexdigest()
    with store.connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute("DELETE FROM rate_limits WHERE minute<?", (minute - 2,))
        count = conn.execute("SELECT count FROM rate_limits WHERE subject=? AND minute=?", (digest, minute)).fetchone()
        if count and count[0] >= limit:
            raise PermissionError("Too many requests; please wait a minute")
        conn.execute("INSERT INTO rate_limits VALUES(?,?,1) ON CONFLICT(subject,minute) DO UPDATE SET count=count+1", (digest, minute))
