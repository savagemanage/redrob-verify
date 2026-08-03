"""One-image OCR smoke for GPU hosts.

Fails if /v1/meta is not a real OCR backend, the request times out, or text is empty.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

from harness.config_util import load_config, resolve_data_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke-test OCR on one MIDV image")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Override image path (default: first scan under data/1_ocr/img)",
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    endpoint = str((cfg.get("endpoints") or {}).get("ocr") or "").rstrip("/")
    if not endpoint:
        print("ERROR: endpoints.ocr missing", file=sys.stderr)
        sys.exit(2)

    meta = httpx.get(f"{endpoint}/v1/meta", timeout=30.0).json()
    backend = str(meta.get("backend") or "")
    service = str(meta.get("service") or "")
    print(f"meta service={service!r} backend={backend!r} model_version={meta.get('model_version')!r}")
    if service != "ocr":
        print(f"ERROR: expected service=ocr, got {service!r}", file=sys.stderr)
        sys.exit(1)
    if backend not in {"paddleocr_vl", "paddleocr_classic"}:
        print(
            f"ERROR: refusing non-Paddle backend {backend!r} (legacy mock risk)",
            file=sys.stderr,
        )
        sys.exit(1)

    data_root = resolve_data_root(cfg)
    image = args.image
    if image is None:
        img_dir = data_root / "1_ocr" / "img"
        candidates = sorted(img_dir.glob("scan_*.jpg")) + sorted(img_dir.glob("scan_*.jpeg"))
        if not candidates:
            candidates = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
        if not candidates:
            print(f"ERROR: no images under {img_dir} — run ./run.sh fetch-midv && ingest-midv", file=sys.stderr)
            sys.exit(2)
        image = candidates[0]
    if not image.is_file():
        print(f"ERROR: image not found: {image}", file=sys.stderr)
        sys.exit(2)

    print(f"POST {endpoint}/v1/ocr/extract  image={image}  timeout={args.timeout_seconds}s")
    t0 = time.perf_counter()
    with image.open("rb") as f:
        resp = httpx.post(
            f"{endpoint}/v1/ocr/extract",
            files={"file": (image.name, f, "application/octet-stream")},
            data={"id": image.stem},
            timeout=args.timeout_seconds,
        )
    ms = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    payload = resp.json()
    text = str(payload.get("text") or "")
    fields = payload.get("fields") if isinstance(payload.get("fields"), dict) else {}
    nonnull = sum(1 for v in fields.values() if isinstance(v, dict) and v.get("value"))
    reason = (payload.get("quality") or {}).get("reason") if isinstance(payload.get("quality"), dict) else None
    print(f"OK ms={ms:.0f} fields={len(fields)} nonnull={nonnull} text_len={len(text)} reason={reason!r}")
    print(f"TEXT_PREVIEW={text[:240]!r}")
    out = Path("results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "smoke_ocr.json").write_text(
        json.dumps(
            {
                "image": str(image),
                "ms": ms,
                "backend": backend,
                "text_len": len(text),
                "n_fields": len(fields),
                "n_nonnull": nonnull,
                "reason": reason,
                "text_preview": text[:500],
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    if not text.strip() and nonnull == 0:
        print("ERROR: empty OCR output — layout/hang regression", file=sys.stderr)
        sys.exit(1)
    print(f"wrote {out / 'smoke_ocr.json'}")


if __name__ == "__main__":
    try:
        main()
    except httpx.TimeoutException:
        print("ERROR: OCR request timed out (treat as hang)", file=sys.stderr)
        sys.exit(1)
