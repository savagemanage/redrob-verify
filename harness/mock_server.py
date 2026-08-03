"""FastAPI mock server: four endpoints with seed-fixed random responses.

Forgery/face scores use overlapping Beta distributions from config.yaml mock.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config.yaml"


def _load_config() -> dict[str, Any]:
    with CONFIG_PATH.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg if isinstance(cfg, dict) else {}


def _load_manifest_index(rel_dir: str) -> dict[str, dict[str, Any]]:
    path = REPO_ROOT / "data" / rel_dir / "manifest.jsonl"
    index: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return index
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("//"):
                continue
            row = json.loads(line)
            index[str(row["id"])] = row
            for key in ("path", "img_a", "img_b"):
                if key in row:
                    name = Path(row[key]).name
                    # Prefer unique filenames (id-based); last write wins on collision
                    index[name] = row
                    index[Path(row[key]).stem] = row
    return index


CFG = _load_config()
SEED = int(CFG.get("seed", 42))
MOCK = CFG.get("mock") or {}
OCR_INDEX = _load_manifest_index("1_ocr")
FORGERY_INDEX = _load_manifest_index("2_forgery")
FACE_INDEX = _load_manifest_index("3_face")
RESUME_INDEX = _load_manifest_index("4_resume")

app = FastAPI(title="redrob-verify mock", version="0.1.0")


def _beta_params(section: str, key: str, default: tuple[float, float]) -> tuple[float, float]:
    raw = (MOCK.get(section) or {}).get(key, list(default))
    return float(raw[0]), float(raw[1])


def _rng_for(record_id: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{SEED}:{record_id}".encode()).digest()
    seed_int = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(seed_int)


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


@app.get("/health")
async def health() -> dict[str, Any]:
    import re

    id_re = re.compile(r"^(ocr|fg|fc|rs)_\d+$")
    return {
        "status": "ok",
        "seed": SEED,
        "counts": {
            "ocr": sum(1 for k in OCR_INDEX if id_re.match(k) and k.startswith("ocr_")),
            "forgery": sum(1 for k in FORGERY_INDEX if id_re.match(k) and k.startswith("fg_")),
            "face": sum(1 for k in FACE_INDEX if id_re.match(k) and k.startswith("fc_")),
            "resume": sum(1 for k in RESUME_INDEX if id_re.match(k) and k.startswith("rs_")),
        },
    }


@app.get("/v1/meta")
async def meta() -> dict[str, Any]:
    # Intentionally NOT a production module name — preflight must reject this.
    from datetime import datetime, timezone

    return {
        "service": "mock",
        "version": "0.1.0",
        "backend": "seeded_random",
        "model_sha256": "none",
        "git_commit": "unknown",
        "started_at": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/v1/ocr/extract")
async def ocr_extract(
    file: UploadFile = File(...),
    id: str = Form(""),
) -> JSONResponse:
    t0 = time.perf_counter()
    record_id = id or Path(file.filename or "unknown").stem
    row = OCR_INDEX.get(record_id) or OCR_INDEX.get(file.filename or "")
    rng = _rng_for(record_id)
    noise = float(MOCK.get("ocr_noise_rate", 0.08))

    if row and "gt_text" in row:
        gt = row["gt_text"]
        chars = list(gt)
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        for i in range(len(chars)):
            if rng.random() < noise:
                chars[i] = alphabet[int(rng.integers(0, len(alphabet)))]
        text = "".join(chars)
    else:
        text = f"mock-ocr-{record_id}"

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return JSONResponse(
        {
            "text": text,
            "fields": {"doc_id": record_id},
            "latency_ms": latency_ms,
        }
    )


@app.post("/v1/forgery/detect")
async def forgery_detect(
    file: UploadFile = File(...),
    id: str = Form(""),
) -> JSONResponse:
    t0 = time.perf_counter()
    record_id = id or Path(file.filename or "unknown").stem
    row = FORGERY_INDEX.get(record_id) or FORGERY_INDEX.get(file.filename or "")
    rng = _rng_for(record_id)

    label = int(row["label"]) if row and "label" in row else 0
    if label == 1:
        a, b = _beta_params("forgery", "label_1", (3.0, 2.0))
    else:
        a, b = _beta_params("forgery", "label_0", (2.0, 3.0))
    score = _clip01(float(rng.beta(a, b)))

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return JSONResponse(
        {
            "score": score,
            "evidence": [{"type": "mock", "label_hint": label}],
            "latency_ms": latency_ms,
        }
    )


@app.post("/v1/face/compare")
async def face_compare(
    img_a: UploadFile = File(...),
    img_b: UploadFile = File(...),
    id: str = Form(""),
) -> JSONResponse:
    t0 = time.perf_counter()
    record_id = id or Path(img_a.filename or "unknown").stem
    # strip _a suffix if present
    if record_id.endswith("_a"):
        maybe = record_id[:-2]
        if maybe in FACE_INDEX:
            record_id = maybe
    row = FACE_INDEX.get(record_id) or FACE_INDEX.get(img_a.filename or "")
    rng = _rng_for(record_id if row is None else str(row.get("id", record_id)))

    same = bool(row["same"]) if row and "same" in row else False
    if same:
        a, b = _beta_params("face", "same_true", (4.0, 2.0))
    else:
        a, b = _beta_params("face", "same_false", (2.0, 4.0))
    similarity = _clip01(float(rng.beta(a, b)))

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return JSONResponse(
        {
            "similarity": similarity,
            "latency_ms": latency_ms,
        }
    )


@app.post("/v1/identity/aggregate")
async def identity_aggregate(
    file: UploadFile = File(...),
    id: str = Form(""),
) -> JSONResponse:
    t0 = time.perf_counter()
    record_id = id or Path(file.filename or "unknown").stem
    rng = _rng_for(record_id)

    sleep_range = MOCK.get("identity_sleep_seconds", [0.01, 0.05])
    lo, hi = float(sleep_range[0]), float(sleep_range[1])
    time.sleep(float(rng.uniform(lo, hi)))

    consistency = int(rng.integers(40, 101))
    sources = ["github", "leetcode", "codeforces", "stackoverflow"]
    if record_id in RESUME_INDEX:
        sources = list(RESUME_INDEX[record_id].get("profiles", {}).keys()) or sources

    latency_ms = (time.perf_counter() - t0) * 1000.0
    return JSONResponse(
        {
            "consistency_score": consistency,
            "sources": sources,
            "discrepancies": [],
            "latency_ms": latency_ms,
        }
    )


def main() -> None:
    import uvicorn

    uvicorn.run("harness.mock_server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
