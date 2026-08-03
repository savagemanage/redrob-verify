"""OCR backend protocol — keep model-zoo imports out of call sites."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class OcrBackend(Protocol):
    name: str
    license: str
    model_version: str

    def extract(self, image: np.ndarray) -> dict[str, Any]:
        """Return structured OCR result.

        Required keys:
          fields: {name: {value, bbox, confidence}}
            — if bbox cannot be attached, value MUST be null (no hallucination)
          text: concatenated page text (optional diagnostic; not used for TC1 gate)
          script: detected/hint script or None
          reason: optional backend error string
        """
        ...

    def model_sha256(self) -> str:
        ...
