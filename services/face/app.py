"""FastAPI face comparison service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from services.common.meta import build_meta, sha256_file
from services.face.backends.factory import create_backend
from services.face.pipeline import FacePipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_cfg() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


CFG = _load_cfg()
FACE_CFG = dict(CFG.get("face") or {})
FACE_CFG.setdefault("seed", CFG.get("seed", 42))
if __import__("os").getenv("FACE_BACKEND"):
    FACE_CFG["backend"] = __import__("os").environ["FACE_BACKEND"]
BACKEND = create_backend(FACE_CFG, repo_root=REPO_ROOT)
PIPELINE = FacePipeline(
    BACKEND,
    quality_cfg=FACE_CFG.get("quality") or {},
    detect_cfg=FACE_CFG.get("detect") or {},
)

app = FastAPI(title="redrob-verify face", version="0.1.0")


class CompareBody(BaseModel):
    img_a: str = Field(..., description="Filesystem path or base64 image")
    img_b: str = Field(..., description="Filesystem path or base64 image")


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": BACKEND.name,
        "license": BACKEND.license,
    }


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    model_sha = "none"
    if BACKEND.name == "sface":
        sface_cfg = FACE_CFG.get("sface") or {}
        path = REPO_ROOT / str(
            sface_cfg.get("sface_path", "models/face/face_recognition_sface_2021dec.onnx")
        )
        model_sha = sha256_file(path) or "missing"
    return build_meta(
        service="face",
        backend=BACKEND.name,
        model_sha256=model_sha,
        extra={"license": BACKEND.license},
    )


@app.post("/v1/face/compare")
async def face_compare(request: Request) -> JSONResponse:
    """Accept JSON `{img_a, img_b}` (path|base64) or multipart files `img_a`/`img_b`."""
    ctype = (request.headers.get("content-type") or "").lower()
    if "application/json" in ctype:
        payload = await request.json()
        body = CompareBody.model_validate(payload)
        return JSONResponse(PIPELINE.compare(body.img_a, body.img_b))

    form = await request.form()
    fa = form.get("img_a")
    fb = form.get("img_b")
    if fa is None or fb is None:
        return JSONResponse(
            {
                "similarity": None,
                "backend": BACKEND.name,
                "latency_ms": 0,
                "reason": "missing img_a or img_b",
                "quality": {},
            },
            status_code=400,
        )
    bytes_a = await fa.read() if hasattr(fa, "read") else bytes(fa)  # type: ignore[arg-type]
    bytes_b = await fb.read() if hasattr(fb, "read") else bytes(fb)  # type: ignore[arg-type]
    return JSONResponse(PIPELINE.compare(bytes_a, bytes_b))


def main() -> None:
    import uvicorn

    host = str(FACE_CFG.get("host", "127.0.0.1"))
    port = int(FACE_CFG.get("port", 8002))
    uvicorn.run("services.face.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
