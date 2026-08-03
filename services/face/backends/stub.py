"""Deterministic hash-based fake embeddings for plumbing tests (no model weights)."""

from __future__ import annotations

import hashlib

import numpy as np


class StubEmbeddingBackend:
    """License: N/A (synthetic). Embeddings are deterministic functions of pixel bytes."""

    name = "stub"
    license = "N/A (synthetic, no model weights)"

    def __init__(self, dim: int = 128, seed: int = 42) -> None:
        self.dim = int(dim)
        self.seed = int(seed)

    def embed(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 2:
            payload = img.tobytes()
        else:
            payload = np.ascontiguousarray(img).tobytes()
        digest = hashlib.sha256(f"{self.seed}:".encode() + payload).digest()
        # Expand digest into dim floats in [-1, 1]
        raw = np.frombuffer(
            (digest * ((self.dim * 4) // len(digest) + 1))[: self.dim * 4],
            dtype=np.uint8,
        ).astype(np.float64)
        vec = (raw[: self.dim] / 255.0) * 2.0 - 1.0
        # Mix in a few spatial moments so tiny crops still vary slightly with layout
        if img.size:
            flat = img.astype(np.float64).ravel()
            vec = vec + 0.01 * np.array(
                [flat.mean(), flat.std(), flat.min(), flat.max()]
                + [0.0] * max(0, self.dim - 4),
                dtype=np.float64,
            )[: self.dim]
        norm = np.linalg.norm(vec)
        if norm < 1e-12:
            vec = np.zeros(self.dim, dtype=np.float64)
            vec[0] = 1.0
            return vec
        return (vec / norm).astype(np.float64)
