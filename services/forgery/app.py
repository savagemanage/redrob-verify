"""FastAPI forgery score service. It returns a raw continuous score only."""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from services.common.meta import build_meta, sha256_file
from services.forgery.backends import create_detector

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict[str, Any]:
    with (REPO_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


CFG = _load_config()
FORGERY_CFG = dict(CFG.get("forgery") or {})
if os.getenv("FORGERY_BACKEND"):
    FORGERY_CFG["backend"] = os.environ["FORGERY_BACKEND"]
if os.getenv("FORGERY_WEIGHTS"):
    FORGERY_CFG["weights_path"] = os.environ["FORGERY_WEIGHTS"]


def _select_device() -> torch.device:
    """Prefer CUDA only when a kernel can actually run (Blackwell needs cu128+)."""
    if not torch.cuda.is_available():
        return torch.device("cpu")
    try:
        # zeros() can succeed while conv kernels are missing for sm_120.
        x = torch.randn(1, 3, 8, 8, device="cuda")
        w = torch.randn(4, 3, 3, 3, device="cuda")
        y = torch.nn.functional.conv2d(x, w, padding=1)
        torch.cuda.synchronize()
        del x, w, y
        return torch.device("cuda")
    except RuntimeError as error:
        print(f"forgery: CUDA unusable ({error}); falling back to CPU", flush=True)
        return torch.device("cpu")


DEVICE = _select_device()
torch.manual_seed(int(CFG.get("seed", 42)))
DETECTOR = create_detector(FORGERY_CFG, repo_root=REPO_ROOT, device=DEVICE)
WEIGHTS_PATH = DETECTOR.weights_path

app = FastAPI(title="redrob-verify forgery", version="0.1.0")


class DetectBody(BaseModel):
    image: str = Field(..., description="filesystem path or base64-encoded image")


def _decode_json_image(value: str) -> bytes:
    candidate = Path(value)
    if candidate.is_file():
        return candidate.read_bytes()
    encoded = value.split(",", 1)[-1] if value.startswith("data:") else value
    try:
        return base64.b64decode(encoded, validate=True)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="image must be a readable path or base64 image") from exc


def detect(content: bytes) -> dict[str, Any]:
    start = time.perf_counter()
    score = DETECTOR.score(content)
    return {
        "score": max(0.0, min(1.0, score)),
        "evidence": [],
        "latency_ms": round((time.perf_counter() - start) * 1000),
        "backend": DETECTOR.name,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE.type,
        "backend": DETECTOR.name,
        "model_status": DETECTOR.status,
        "weights_path": str(WEIGHTS_PATH.relative_to(REPO_ROOT))
        if WEIGHTS_PATH.is_relative_to(REPO_ROOT)
        else str(WEIGHTS_PATH),
    }


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    return build_meta(
        service="forgery",
        backend=f"{DETECTOR.name}:{DETECTOR.status}",
        model_sha256=sha256_file(WEIGHTS_PATH),
        extra={"device": DEVICE.type},
    )


@app.post("/v1/forgery/detect")
async def forgery_detect(request: Request) -> dict[str, Any]:
    """Accept JSON ``{image: path|base64}`` or multipart ``file``."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = DetectBody.model_validate(await request.json())
        except ValueError as exc:
            raise HTTPException(status_code=422, detail="invalid JSON body") from exc
        return detect(_decode_json_image(body.image))

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="multipart request requires file")
    return detect(await upload.read())
