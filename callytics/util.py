"""Small shared helpers. Kept dependency-free so every layer may import it."""

from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import UTC, datetime
from uuid import UUID

_UUID7_LAST_MS = 0
_UUID7_SEQ = 0


def utcnow() -> datetime:
    return datetime.now(UTC)


def uuid7() -> UUID:
    """Time-ordered UUID (RFC 9562 v7).

    Time ordering matters here: lead and activity ids are used as pagination
    cursors on the timeline, so random v4 ids would force an extra sort key.
    """
    global _UUID7_LAST_MS, _UUID7_SEQ
    ms = int(time.time() * 1000)
    if ms == _UUID7_LAST_MS:
        _UUID7_SEQ += 1
    else:
        _UUID7_LAST_MS, _UUID7_SEQ = ms, 0
    rand = int.from_bytes(os.urandom(8), "big")
    value = (ms & 0xFFFFFFFFFFFF) << 80
    value |= 0x7 << 76
    value |= (_UUID7_SEQ & 0xFFF) << 64
    value |= (0b10 << 62) | (rand & 0x3FFFFFFFFFFFFFFF)
    return UUID(int=value)


# --- phone normalisation -------------------------------------------------
# Every lead in this business is reachable by phone, and phone is the join key
# between the CRM, the telephony vendor and FinEcho recordings. Normalisation
# has to be identical in all three or dedupe silently fails.

_DIGITS = re.compile(r"\D+")
DEFAULT_COUNTRY_CODE = "91"


def normalize_phone(raw: str | None, *, country_code: str = DEFAULT_COUNTRY_CODE) -> str | None:
    """Return E.164-without-plus (e.g. ``919876543210``) or None if unusable.

    Accepts the shapes actually seen in the LeadSquared export: ``+91 98765
    43210``, ``098765-43210``, ``9876543210``, ``0091 9876543210``.
    """
    if not raw:
        return None
    digits = _DIGITS.sub("", raw)
    if not digits:
        return None
    if digits.startswith("00"):
        digits = digits[2:]
    if len(digits) == 10:
        digits = country_code + digits
    elif len(digits) == 11 and digits.startswith("0"):
        digits = country_code + digits[1:]
    if not (10 <= len(digits) <= 15):
        return None
    return digits


def phone_variants(phone: str) -> list[str]:
    """Lookup variants for matching against systems that store phones loosely."""
    normalized = normalize_phone(phone)
    if not normalized:
        return []
    out = [normalized, f"+{normalized}"]
    if normalized.startswith(DEFAULT_COUNTRY_CODE) and len(normalized) == 12:
        local = normalized[len(DEFAULT_COUNTRY_CODE) :]
        out += [local, f"0{local}"]
    return list(dict.fromkeys(out))


def normalize_email(raw: str | None) -> str | None:
    if not raw:
        return None
    value = raw.strip().lower()
    return value if "@" in value and "." in value.split("@")[-1] else None


def normalize_name(raw: str | None) -> str:
    """Casefold + strip accents/punctuation, for fuzzy identity comparison."""
    if not raw:
        return ""
    decomposed = unicodedata.normalize("NFKD", raw)
    stripped = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"[^a-z0-9 ]+", " ", stripped.lower()).strip()


def slugify(raw: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", normalize_name(raw)).strip("_")
