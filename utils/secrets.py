from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse, urlunparse

SENSITIVE_PARAM_KEYS = {
    "password",
    "passwd",
    "secret",
    "token",
    "api_key",
    "apikey",
    "access_token",
    "auth",
    "sasl_password",
    "private_key",
    "key",
}

URL_CRED_REGEX = re.compile(r"([a-zA-Z0-9\+\-\.]+://)([^:]+):([^@]+)@")
BEARER_REGEX = re.compile(r"(Bearer\s+)[A-Za-z0-9\-\._~\+\/]+=*", re.IGNORECASE)
GENERIC_KEY_REGEX = re.compile(
    r'(?i)(api[_-]?key|secret|password|passwd|token|auth|sasl[_-]?password)\s*[:=]\s*["\']?([^"\'\s&]+)["\']?'
)


class SecretString:
    def __init__(self, value: str):
        self._raw = str(value) if value is not None else ""

    def expose(self) -> str:
        return self._raw

    def __repr__(self) -> str:
        return "<SecretString [REDACTED]>"

    def __str__(self) -> str:
        return mask_secret(self._raw)

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, SecretString):
            return self._raw == other._raw
        if isinstance(other, str):
            return self._raw == other
        return False

    def __len__(self) -> int:
        return len(self._raw)

    def __bool__(self) -> bool:
        return bool(self._raw)


def mask_secret(value: str | None, visible_chars: int = 4) -> str:
    if not value:
        return ""
    val_str = str(value)
    length = len(val_str)
    if length <= visible_chars:
        return "*" * length
    return f"{'*' * (length - visible_chars)}{val_str[-visible_chars:]}"


def sanitize_url(url_str: str | None) -> str:
    if not url_str:
        return ""
    try:
        parsed = urlparse(url_str)
        if parsed.password:
            user = parsed.username or ""
            netloc = f"{user}:****@{parsed.hostname}"
            if parsed.port:
                netloc += f":{parsed.port}"
            sanitized = parsed._replace(netloc=netloc)
            return urlunparse(sanitized)
        return sanitize_text(url_str)
    except Exception:
        return sanitize_text(url_str)


def sanitize_text(text: str | None) -> str:
    if not text:
        return ""
    s = URL_CRED_REGEX.sub(r"\1\2:****@", str(text))
    s = BEARER_REGEX.sub(r"\1[REDACTED]", s)
    s = GENERIC_KEY_REGEX.sub(r"\1=****", s)
    return s


def sanitize_config(config_dict: dict[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for k, v in config_dict.items():
        k_lower = k.lower()
        if any(sens in k_lower for sens in SENSITIVE_PARAM_KEYS):
            if isinstance(v, str):
                sanitized[k] = mask_secret(v)
            elif isinstance(v, SecretString):
                sanitized[k] = str(v)
            else:
                sanitized[k] = "[REDACTED]"
        elif isinstance(v, str) and ("://" in v or "Bearer " in v):
            sanitized[k] = sanitize_url(v)
        elif isinstance(v, dict):
            sanitized[k] = sanitize_config(v)
        else:
            sanitized[k] = v
    return sanitized
