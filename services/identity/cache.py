"""Small TTL disk cache for public-profile lookups."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any


class DiskCache:
    def __init__(self, directory: Path, ttl_seconds: float) -> None:
        self.directory = directory
        self.ttl_seconds = ttl_seconds
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, source: str, handle: str) -> Path:
        digest = hashlib.sha256(f"{source}:{handle.casefold()}".encode()).hexdigest()
        return self.directory / f"{digest}.json"

    def get(self, source: str, handle: str | None) -> dict[str, Any] | None:
        if not handle:
            return None
        path = self._path(source, handle)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - float(payload["stored_at"]) <= self.ttl_seconds:
                result = dict(payload["result"])
                result["cached"] = True
                return result
        except (OSError, ValueError, KeyError, TypeError):
            return None
        return None

    def put(self, source: str, handle: str | None, result: dict[str, Any]) -> None:
        if not handle or result.get("status") not in {"ok", "not_found"}:
            return
        self._path(source, handle).write_text(
            json.dumps({"stored_at": time.time(), "result": result}, sort_keys=True),
            encoding="utf-8",
        )
