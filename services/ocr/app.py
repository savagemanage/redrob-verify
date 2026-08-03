"""FastAPI service for document OCR."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, model_validator

from services.common.meta import build_meta
from services.ocr.pipeline import OcrPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    return config if isinstance(config, dict) else {}


CFG = _load_config()
OCR_CFG = dict(CFG.get("ocr") or {})
PIPELINE = OcrPipeline(
    dict(OCR_CFG.get("preprocess") or {}),
    ocr_cfg=OCR_CFG,
    repo_root=REPO_ROOT,
    lang=str(OCR_CFG.get("lang") or "en"),
)

app = FastAPI(title="redrob-verify OCR", version="0.1.0")


class OcrBody(BaseModel):
    path: str | None = None
    base64: str | None = None
    doc_type: str | None = None
    preprocess: dict[str, bool] | None = None

    @model_validator(mode="after")
    def requires_image(self) -> OcrBody:
        if not self.path and not self.base64:
            raise ValueError("one of path or base64 is required")
        if self.path and self.base64:
            raise ValueError("provide only one of path or base64")
        return self


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "backend": PIPELINE.backend.name,
        "model_version": getattr(PIPELINE.backend, "model_version", None),
        "preprocess": PIPELINE.preprocess_cfg,
    }


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    backend = PIPELINE.backend
    sha = backend.model_sha256() if hasattr(backend, "model_sha256") else "none"
    return build_meta(
        service="ocr",
        backend=backend.name,
        model_sha256=sha,
        extra={
            "model_version": getattr(backend, "model_version", None),
            "model_sha256": sha,
            "preprocess": PIPELINE.preprocess_cfg,
            "lang": getattr(PIPELINE, "lang", None),
            "backend_lang": getattr(backend, "lang", None),
            "pipeline_version": getattr(backend, "model_version", None),
        },
    )


@app.post("/v1/ocr/extract")
async def extract(request: Request) -> dict[str, Any]:
    """Accept JSON `{path|base64, doc_type?}` or multipart `file` plus `id`."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        try:
            body = OcrBody.model_validate(await request.json())
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return PIPELINE.extract(
            body.path or body.base64 or "",
            body.doc_type,
            preprocess_override=body.preprocess,
        )

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="multipart request requires file")
    content = await upload.read()
    doc_type = form.get("doc_type")
    return PIPELINE.extract(content, str(doc_type) if doc_type else None)


def main() -> None:
    import uvicorn

    uvicorn.run(
        "services.ocr.app:app",
        host=str(OCR_CFG.get("host", "127.0.0.1")),
        port=int(OCR_CFG.get("port", 8001)),
        reload=False,
    )


if __name__ == "__main__":
    main()
