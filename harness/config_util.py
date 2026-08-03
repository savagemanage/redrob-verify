"""Shared config / path helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.yaml"

# Dataset dirs under data_root (data/1_ocr, data/2_forgery, …)
DS_OCR = "1_ocr"
DS_FORGERY = "2_forgery"
DS_FACE = "3_face"
DS_RESUME = "4_resume"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"invalid config: {cfg_path}")
    return cfg


def resolve_data_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["data_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    return root


def resolve_results_root(cfg: dict[str, Any]) -> Path:
    root = Path(cfg["results_root"])
    if not root.is_absolute():
        root = REPO_ROOT / root
    root.mkdir(parents=True, exist_ok=True)
    return root
