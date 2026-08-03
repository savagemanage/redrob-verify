"""Detect → align → embed → cosine similarity pipeline."""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from services.face.backends.base import EmbeddingBackend
from services.face.backends.sface import SFaceBackend
from services.face.detect import align_face, detect_faces
from services.face.quality import assess_quality


def cosine_to_unit_interval(cosine: float) -> float:
    """Map cosine similarity from [-1, 1] to [0, 1]: similarity_01 = (cosine + 1) / 2."""
    return float(np.clip((cosine + 1.0) / 2.0, 0.0, 1.0))


def load_bgr(source: str | bytes | np.ndarray, *, data_root: Path | None = None) -> np.ndarray:
    """Load image from ndarray, filesystem path, or base64 string."""
    if isinstance(source, np.ndarray):
        img = source
        if img.ndim == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return img

    if isinstance(source, (bytes, bytearray)):
        arr = np.frombuffer(source, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("failed to decode image bytes")
        return img

    text = str(source).strip()
    if text.startswith("data:") and "base64," in text:
        text = text.split("base64,", 1)[1]
    path_candidate = Path(text)
    if data_root is not None and not path_candidate.is_file():
        alt = data_root / text
        if alt.is_file():
            path_candidate = alt
    if path_candidate.is_file():
        img = cv2.imread(str(path_candidate), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"failed to read image: {path_candidate}")
        return img

    try:
        raw = base64.b64decode(text, validate=False)
    except Exception as e:
        raise ValueError(f"img is neither a readable path nor base64: {e}") from e
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("failed to decode base64 image")
    return img


class FacePipeline:
    def __init__(
        self,
        backend: EmbeddingBackend | SFaceBackend,
        *,
        quality_cfg: dict[str, Any] | None = None,
        detect_cfg: dict[str, Any] | None = None,
        align_size: tuple[int, int] = (112, 112),
    ) -> None:
        self.backend = backend
        self.quality_cfg = quality_cfg or {}
        q = self.quality_cfg
        d = detect_cfg or {}
        self.min_face_side_px = float(q.get("min_face_side_px", 40))
        self.min_laplacian_var = float(q.get("min_laplacian_var", 50.0))
        self.max_yaw_deg = float(q.get("max_yaw_deg", 30.0))
        self.max_pitch_deg = float(q.get("max_pitch_deg", 25.0))
        self.fullframe_fallback = bool(d.get("fullframe_fallback", False))
        self.align_size = align_size

    def _process_one_legacy(
        self, bgr: np.ndarray
    ) -> tuple[np.ndarray | None, dict[str, Any], str | None]:
        dets = detect_faces(bgr, fullframe_fallback=self.fullframe_fallback)
        if not dets:
            return None, {}, "no_face_detected"
        det = dets[0]
        quality = assess_quality(
            bgr,
            det,
            min_face_side_px=self.min_face_side_px,
            min_laplacian_var=self.min_laplacian_var,
            max_yaw_deg=self.max_yaw_deg,
            max_pitch_deg=self.max_pitch_deg,
        )
        aligned = align_face(bgr, det.landmarks, out_size=self.align_size)
        emb = self.backend.embed(aligned)
        return emb, quality, None

    def compare(
        self,
        img_a: str | bytes | np.ndarray,
        img_b: str | bytes | np.ndarray,
        *,
        data_root: Path | None = None,
    ) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            a = load_bgr(img_a, data_root=data_root)
            b = load_bgr(img_b, data_root=data_root)
        except ValueError as e:
            latency_ms = int(round((time.perf_counter() - t0) * 1000))
            return {
                "similarity": None,
                "backend": self.backend.name,
                "latency_ms": latency_ms,
                "reason": f"image_load_failed: {e}",
                "quality": {},
            }

        if isinstance(self.backend, SFaceBackend):
            emb_a, qa, reason_a = self.backend.process(a, quality_cfg=self.quality_cfg)
            emb_b, qb, reason_b = self.backend.process(b, quality_cfg=self.quality_cfg)
        else:
            emb_a, qa, reason_a = self._process_one_legacy(a)
            emb_b, qb, reason_b = self._process_one_legacy(b)

        latency_ms = int(round((time.perf_counter() - t0) * 1000))
        quality = {"img_a": qa, "img_b": qb}

        if emb_a is None or emb_b is None:
            reasons = [r for r in (reason_a, reason_b) if r]
            return {
                "similarity": None,
                "backend": self.backend.name,
                "latency_ms": latency_ms,
                "reason": ";".join(reasons) or "no_face_detected",
                "quality": quality,
            }

        if isinstance(self.backend, SFaceBackend):
            cosine = self.backend.match_cosine(emb_a, emb_b)
        else:
            cosine = float(np.dot(emb_a, emb_b))
        similarity = cosine_to_unit_interval(cosine)
        return {
            "similarity": similarity,
            "backend": self.backend.name,
            "latency_ms": latency_ms,
            "cosine": cosine,
            "quality": quality,
        }
