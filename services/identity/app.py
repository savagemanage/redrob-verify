"""FastAPI endpoint for deterministic digital identity aggregation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request

from services.common.meta import build_meta
from services.identity.pipeline import IdentityPipeline

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "config.yaml"
PROFILE_KEYS = {"github", "leetcode", "codeforces", "stackoverflow"}


def _load_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return config if isinstance(config, dict) else {}


CFG = _load_config()
IDENTITY_CFG = dict(CFG.get("identity") or {})
PIPELINE = IdentityPipeline(IDENTITY_CFG)
app = FastAPI(title="redrob-verify Identity", version="0.1.0")


def _profiles(values: dict[str, Any]) -> dict[str, str]:
    return {
        key.removeprefix("profile_"): str(value).strip()
        for key, value in values.items()
        if key.startswith("profile_")
        and key.removeprefix("profile_") in PROFILE_KEYS
        and str(value).strip()
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    return {"status": "ok", "source_timeout_seconds": PIPELINE.timeout_seconds}


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    return build_meta(
        service="identity",
        backend="rule_score+public_apis",
        model_sha256="none",
        extra={
            "sources": ["github", "leetcode", "codeforces", "stackoverflow"],
            "source_timeout_seconds": PIPELINE.timeout_seconds,
        },
    )


@app.post("/v1/identity/aggregate")
async def aggregate(request: Request) -> dict[str, Any]:
    """Accept multipart PDF plus profile fields, or JSON `{path, profile_*}`."""
    content_type = (request.headers.get("content-type") or "").lower()
    if "application/json" in content_type:
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get("path"), str):
            raise HTTPException(status_code=422, detail="JSON request requires string path")
        return await PIPELINE.aggregate(None, body["path"], _profiles(body))

    form = await request.form()
    upload = form.get("file")
    if upload is None or not hasattr(upload, "read"):
        raise HTTPException(status_code=422, detail="multipart request requires PDF file")
    filename = str(getattr(upload, "filename", ""))
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=422, detail="file must be a PDF")
    form_values = {str(key): value for key, value in form.items() if isinstance(value, str)}
    return await PIPELINE.aggregate(await upload.read(), None, _profiles(form_values))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "services.identity.app:app",
        host=str(IDENTITY_CFG.get("host", "127.0.0.1")),
        port=int(IDENTITY_CFG.get("port", 8004)),
        reload=False,
    )


if __name__ == "__main__":
    main()
