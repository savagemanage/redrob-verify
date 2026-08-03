"""PDF resume parsing and deterministic public-profile handle extraction."""

from __future__ import annotations

import io
import re
from collections.abc import Iterable
from typing import Any
from urllib.parse import unquote, urlparse

import pdfplumber

SOURCES = ("github", "leetcode", "codeforces", "stackoverflow")
_URL_PATTERNS = {
    "github": r"(?:https?://)?(?:www\.)?github\.com/([^/?#\s]+)",
    "leetcode": r"(?:https?://)?(?:www\.)?leetcode\.com/(?:u/|@)?([^/?#\s]+)",
    "codeforces": r"(?:https?://)?(?:www\.)?codeforces\.com/profile/([^/?#\s]+)",
    "stackoverflow": r"(?:https?://)?(?:www\.)?stackoverflow\.com/users/(?:\d+/)?([^/?#\s]+)",
}
_LABEL_PATTERNS = {
    source: re.compile(rf"\b{source}\b\s*[:@]\s*([A-Za-z0-9_.-]+)", re.IGNORECASE)
    for source in SOURCES
}


def _walk_annotations(value: Any) -> Iterable[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_annotations(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_annotations(child)


def _annotation_uri(annotation: dict[str, Any]) -> str | None:
    uri = annotation.get("URI") or annotation.get("uri")
    if isinstance(uri, bytes):
        return uri.decode("utf-8", errors="ignore")
    if uri:
        return str(uri)
    action = annotation.get("A") or annotation.get("a")
    if not isinstance(action, dict):
        return None
    uri = action.get("URI") or action.get("uri")
    if isinstance(uri, bytes):
        return uri.decode("utf-8", errors="ignore")
    return str(uri) if uri else None


def parse_pdf(content: bytes) -> tuple[str, list[str]]:
    """Extract PDF text and hyperlink targets, raising when pdfplumber cannot parse."""
    text_parts: list[str] = []
    urls: list[str] = []
    with pdfplumber.open(io.BytesIO(content)) as document:
        for page in document.pages:
            text_parts.append(page.extract_text() or "")
            for annotation in page.annots or []:
                for candidate in _walk_annotations(annotation):
                    if uri := _annotation_uri(candidate):
                        urls.append(uri)
    return "\n".join(text_parts), sorted(set(urls))


def ocr_fallback_reason(error: Exception) -> dict[str, str]:
    """Expose why fallback did not run; OCR is intentionally not invoked on success."""
    return {
        "status": "unavailable",
        "reason": f"pdfplumber failed; OCR fallback is not configured: {error}",
    }


def extract_handles(text: str, urls: Iterable[str]) -> dict[str, str]:
    """Return at most one normalized handle per supported public platform."""
    candidates = "\n".join([text, *urls])
    handles: dict[str, str] = {}
    for source, pattern in _URL_PATTERNS.items():
        match = re.search(pattern, candidates, re.IGNORECASE)
        if not match:
            match = _LABEL_PATTERNS[source].search(text)
        if match:
            handle = unquote(match.group(1)).strip().strip(".,;:)]}>")
            if handle:
                handles[source] = handle
    return handles


def extract_name(text: str) -> str | None:
    """Best-effort name field used only for transparent discrepancy reporting."""
    for line in text.splitlines()[:8]:
        cleaned = " ".join(line.split())
        if 2 <= len(cleaned) <= 80 and not re.search(
            r"[@:/]|github|leetcode|codeforces|stackoverflow", cleaned, re.I
        ):
            return cleaned
    return None


def profile_url(source: str, handle: str) -> str:
    bases = {
        "github": "https://github.com/",
        "leetcode": "https://leetcode.com/u/",
        "codeforces": "https://codeforces.com/profile/",
        "stackoverflow": "https://stackoverflow.com/users/",
    }
    return f"{bases[source]}{handle}" if source in bases else urlparse(handle).geturl()
