"""OCR backend factory — mirror face backends/factory.py pattern."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.ocr.backends.base import OcrBackend
from services.ocr.backends.paddleocr_classic import PaddleOcrClassicBackend
from services.ocr.backends.paddleocr_vl import PaddleOcrVlBackend


class StubOcrBackend:
    name = "stub"
    license = "n/a"
    model_version = "stub"

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def model_sha256(self) -> str:
        return "none"

    def extract(self, image: Any) -> dict[str, Any]:
        del image
        return {
            "fields": {},
            "text": "",
            "script": None,
            "reason": self.reason,
        }


def create_backend(ocr_cfg: dict[str, Any], *, repo_root: Path | None = None) -> OcrBackend:
    name = str(ocr_cfg.get("backend") or "paddleocr_vl").strip().lower()
    lang = str(ocr_cfg.get("lang") or "en")
    try:
        if name in {"paddleocr_classic", "classic", "paddleocr"}:
            return PaddleOcrClassicBackend(lang=lang)
        if name in {"paddleocr_vl", "vl"}:
            vl_cfg = ocr_cfg.get("vl") or {}
            model_dir = vl_cfg.get("model_dir")
            path = Path(str(model_dir)) if model_dir else None
            if path is not None and repo_root is not None and not path.is_absolute():
                path = repo_root / path
            return PaddleOcrVlBackend(model_dir=path)
        raise ValueError(f"unknown ocr backend '{name}'. Supported: paddleocr_classic, paddleocr_vl")
    except (ImportError, OSError, RuntimeError, ValueError) as error:
        reason = f"{name}_unavailable: {type(error).__name__}: {error}"
        print(f"OCR backend fallback to stub: {reason}", flush=True)
        return StubOcrBackend(reason)


__all__ = [
    "OcrBackend",
    "StubOcrBackend",
    "PaddleOcrClassicBackend",
    "PaddleOcrVlBackend",
    "create_backend",
]
