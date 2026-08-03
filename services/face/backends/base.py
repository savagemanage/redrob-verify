"""EmbeddingBackend protocol — keep model-zoo imports out of the rest of the stack."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingBackend(Protocol):
    name: str
    license: str

    def embed(self, img: np.ndarray) -> np.ndarray:
        """Return an L2-normalised embedding vector for an aligned face crop (RGB or BGR HxWxC)."""
        ...
