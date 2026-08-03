"""OCR backend selection, document classification, and field extraction."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.ocr.backends.factory import create_backend
from services.ocr.preprocess import preprocess_image


def load_image(source: str | bytes) -> np.ndarray:
    """Decode file path, data URL, raw base64, or uploaded bytes."""
    if isinstance(source, bytes):
        raw = source
    else:
        value = source.strip()
        # Prefer filesystem path only for short path-like strings. Never pass
        # multi-KB base64 into Path.stat() (OSError: File name too long).
        if len(value) < 4096 and not value.startswith("data:") and ("/" in value or "\\" in value or value.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"))):
            candidate = Path(value)
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            if is_file:
                image = cv2.imread(str(candidate), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"failed to read image: {candidate}")
                return image
        if value.startswith("data:") and "base64," in value:
            value = value.split("base64,", 1)[1]
        try:
            raw = base64.b64decode(value, validate=False)
        except Exception as error:
            raise ValueError("image must be a readable path or base64 payload") from error

    image = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("failed to decode image bytes")
    return image


def classify_doc_type(text: str, image: np.ndarray) -> str:
    """Use stable keyword/layout clues; returns generic_document when unsure."""
    upper = text.upper()
    if any(term in upper for term in ("DEGREE", "UNIVERSITY", "BACHELOR", "MASTER")):
        return "degree_certificate"
    if any(term in upper for term in ("EMPLOYMENT", "EXPERIENCE CERTIFICATE", "WORKED AS")):
        return "employment_certificate"
    if any(term in upper for term in ("TRANSCRIPT", "GRADE", "GPA", "SEMESTER")):
        return "transcript"
    if any(term in upper for term in ("PASSPORT", "IDENTITY CARD", "DATE OF BIRTH", "NATIONALITY")):
        return "identity_document"
    aspect_ratio = image.shape[1] / max(image.shape[0], 1)
    return "identity_document" if aspect_ratio > 1.4 else "generic_document"


def _field_values_for_doc_type(fields: dict[str, Any]) -> dict[str, Any]:
    """Pass through structured fields; only retain entries with bbox-backed values."""
    out: dict[str, Any] = {}
    for name, payload in (fields or {}).items():
        if isinstance(payload, dict):
            out[name] = {
                "value": payload.get("value"),
                "bbox": payload.get("bbox"),
                "confidence": payload.get("confidence"),
            }
        else:
            # legacy string field — treat as no-bbox → null (hallucination defense)
            out[str(name)] = {"value": None, "bbox": None, "confidence": None}
    return out


class OcrPipeline:
    def __init__(
        self,
        preprocess_cfg: dict[str, Any] | None = None,
        backend: Any | None = None,
        *,
        ocr_cfg: dict[str, Any] | None = None,
        repo_root: Path | None = None,
        lang: str = "en",
    ) -> None:
        self.preprocess_cfg = preprocess_cfg or {}
        self.ocr_cfg = dict(ocr_cfg or {})
        if "lang" not in self.ocr_cfg:
            self.ocr_cfg["lang"] = lang
        self.lang = str(self.ocr_cfg.get("lang") or lang)
        self.backend = backend or create_backend(self.ocr_cfg, repo_root=repo_root)

    def extract(
        self,
        source: str | bytes,
        requested_doc_type: str | None = None,
        *,
        preprocess_override: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        preprocess_cfg = preprocess_override if preprocess_override is not None else self.preprocess_cfg
        try:
            original = load_image(source)
            image, applied = preprocess_image(original, preprocess_cfg)
        except ValueError as error:
            return {
                "fields": {},
                "text": "",
                "doc_type": requested_doc_type or "unknown",
                "script": None,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "quality": {"reason": f"image_load_failed: {error}"},
            }

        result = self.backend.extract(image)
        fields_raw = result.get("fields") if isinstance(result, dict) else {}
        text = str(result.get("text") or "") if isinstance(result, dict) else ""
        reason = result.get("reason") if isinstance(result, dict) else None
        script = result.get("script") if isinstance(result, dict) else None
        fields = _field_values_for_doc_type(fields_raw if isinstance(fields_raw, dict) else {})
        # Rebuild page text from bbox-backed non-null values only
        page_parts = [
            str(f["value"])
            for f in fields.values()
            if isinstance(f, dict) and f.get("bbox") is not None and f.get("value") is not None
        ]
        if page_parts:
            text = "\n".join(page_parts)
        doc_type = requested_doc_type or classify_doc_type(text, image)
        quality: dict[str, Any] = {
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
            "laplacian_variance": round(float(cv2.Laplacian(image, cv2.CV_64F).var()), 2),
            "preprocess_applied": applied,
            "backend": self.backend.name,
            "model_version": getattr(self.backend, "model_version", None),
        }
        if reason:
            quality["reason"] = reason
        return {
            "fields": fields,
            "text": text,
            "doc_type": doc_type,
            "script": script,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            "quality": quality,
        }
