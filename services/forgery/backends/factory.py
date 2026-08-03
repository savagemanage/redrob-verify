"""Select forgery detector by config / FORGERY_BACKEND env."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from services.forgery.backends.base import ForgeryDetector
from services.forgery.backends.forgery_net import ForgeryNetDetector
from services.forgery.backends.trufor import TruForDetector


def _resolve_path(raw: str | Path, repo_root: Path) -> Path:
    path = Path(raw)
    return path if path.is_absolute() else repo_root / path


def _resolve_trufor_src(forgery_cfg: dict[str, Any], repo_root: Path) -> Path:
    env = os.environ.get("TRUFOR_SRC", "").strip()
    if env:
        return Path(env)
    cfg = forgery_cfg.get("trufor_src")
    if cfg:
        return _resolve_path(str(cfg), repo_root)
    # Host default used during fine-tune / offline eval.
    return Path.home() / "TruFor" / "test_docker" / "src"


def create_detector(
    forgery_cfg: dict[str, Any],
    *,
    repo_root: Path,
    device: torch.device,
) -> ForgeryDetector:
    name = (
        os.environ.get("FORGERY_BACKEND") or str(forgery_cfg.get("backend", "forgery_net"))
    ).strip().lower()
    weights = _resolve_path(
        str(forgery_cfg.get("weights_path", "models/forgery/best.pth")),
        repo_root,
    )

    if name in {"forgery_net", "forgerynet"}:
        return ForgeryNetDetector(
            weights,
            device,
            image_size=int(forgery_cfg.get("image_size", 224)),
        )
    if name == "trufor":
        return TruForDetector(
            weights,
            device,
            trufor_src=_resolve_trufor_src(forgery_cfg, repo_root),
            image_size=int(forgery_cfg.get("image_size", 512)),
        )
    raise ValueError(f"unknown forgery backend '{name}'. Supported: forgery_net, trufor")


__all__ = ["ForgeryDetector", "create_detector"]
