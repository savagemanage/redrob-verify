"""Common adapter contract for public identity sources."""

from __future__ import annotations

import asyncio
import os
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class SourceResult:
    source: str
    handle: str | None
    status: str
    data: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cached: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class SourceAdapter(Protocol):
    name: str

    async def fetch(self, handle: str | None) -> SourceResult:
        """Return a result instead of raising for expected source failures."""


async def maybe_inject_delay() -> None:
    """Test hook: IDENTITY_SOURCE_DELAY_SEC artificial per-source sleep (parallelism proof)."""
    raw = os.getenv("IDENTITY_SOURCE_DELAY_SEC", "").strip()
    if not raw:
        return
    await asyncio.sleep(float(raw))
