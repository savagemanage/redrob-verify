"""Shared /v1/meta payload helpers for all module services."""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
STARTED_AT = datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path | None = None) -> str:
    root = repo or REPO_ROOT
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_meta(
    *,
    service: str,
    version: str = "0.1.0",
    backend: str,
    model_sha256: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "service": service,
        "version": version,
        "backend": backend,
        "model_sha256": model_sha256 or "none",
        "git_commit": git_commit(),
        "started_at": STARTED_AT,
    }
    if extra:
        payload.update(extra)
    return payload
