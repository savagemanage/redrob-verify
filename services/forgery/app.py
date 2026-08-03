"""FastAPI forgery score service. It returns a raw continuous score only."""

from __future__ import annotations

import base64
import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from fastapi import FastAPI, HTTPException, Request
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from services.common.meta import build_meta, sha256_file
from services.forgery.model import ForgeryNet

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_config() -> dict[str, Any]:
    with (REPO_ROOT / "config.yaml").open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


CFG = _load_config()
FORGERY_CFG = dict(CFG.get("forgery") or {})
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(int(CFG.get("seed", 42)))
MODEL = ForgeryNet().to(DEVICE).eval()
WEIGHTS_PATH = REPO_ROOT / str(FORGERY_CFG.get("weights_path", "models/forgery/best.pth"))
MODEL_STATUS = "untrained"
if WEIGHTS_PATH.is_file():
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=True)
    state = checkpoint.get("model_state", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    MODEL.load_state_dict(state)
    MODEL_STATUS = "checkpoint"

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


def _tensor_from_image(content: bytes) -> torch.Tensor:
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB").resize((224, 224), Image.Resampling.BILINEAR)
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=422, detail="unable to decode image") from exc
    pixels = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(DEVICE)


def detect(content: bytes) -> dict[str, Any]:
    start = time.perf_counter()
    image = _tensor_from_image(content)
    with torch.inference_mode():
        score = float(torch.sigmoid(MODEL(image)).item())
    return {
        "score": max(0.0, min(1.0, score)),
        "evidence": [],
        "latency_ms": round((time.perf_counter() - start) * 1000),
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "device": DEVICE.type,
        "model_status": MODEL_STATUS,
        "weights_path": str(WEIGHTS_PATH.relative_to(REPO_ROOT)),
    }


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    return build_meta(
        service="forgery",
        backend=f"forgery_net:{MODEL_STATUS}",
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
