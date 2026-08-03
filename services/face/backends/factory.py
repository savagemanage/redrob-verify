"""Backend factory — select by config name without hardcoding model zoos at call sites."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from services.face.backends.base import EmbeddingBackend
from services.face.backends.sface import SFaceBackend
from services.face.backends.stub import StubEmbeddingBackend


def create_backend(
    face_cfg: dict[str, Any], *, repo_root: Path | None = None
) -> EmbeddingBackend | SFaceBackend:
    name = str(face_cfg.get("backend", "stub")).strip().lower()
    if name == "stub":
        stub_cfg = face_cfg.get("stub") or {}
        return StubEmbeddingBackend(
            dim=int(stub_cfg.get("dim", 128)),
            seed=int(stub_cfg.get("seed", face_cfg.get("seed", 42))),
        )
    if name == "sface":
        sface_cfg = face_cfg.get("sface") or {}
        yunet = Path(str(sface_cfg.get("yunet_path", "models/face/face_detection_yunet_2023mar.onnx")))
        sface = Path(str(sface_cfg.get("sface_path", "models/face/face_recognition_sface_2021dec.onnx")))
        if repo_root is not None:
            if not yunet.is_absolute():
                yunet = repo_root / yunet
            if not sface.is_absolute():
                sface = repo_root / sface
        return SFaceBackend(yunet_path=yunet, sface_path=sface)
    raise ValueError(f"unknown face backend '{name}'. Supported: stub, sface")


__all__ = [
    "EmbeddingBackend",
    "StubEmbeddingBackend",
    "SFaceBackend",
    "create_backend",
]
