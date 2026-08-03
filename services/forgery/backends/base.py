"""Forgery detector protocol — continuous score in [0, 1], higher = more forged."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable


@runtime_checkable
class ForgeryDetector(Protocol):
    name: str
    status: str
    weights_path: Path

    def score(self, content: bytes) -> float:
        """Return a continuous forgery score in [0, 1]."""
        ...
