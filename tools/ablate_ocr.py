"""Measure cer_field for OCR preprocess ablation via the running OCR API.

Grid: none / deskew / perspective / bg_remove / all
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
from pathlib import Path
from typing import Any

import httpx

from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.eval_cer import field_cer
from harness.schema import load_ocr_manifest

ABLATIONS: list[tuple[str, dict[str, bool]]] = [
    ("none", {"deskew": False, "perspective": False, "bg_remove": False, "binarize": False, "sharpen": False}),
    ("deskew", {"deskew": True, "perspective": False, "bg_remove": False, "binarize": False, "sharpen": False}),
    ("perspective", {"deskew": False, "perspective": True, "bg_remove": False, "binarize": False, "sharpen": False}),
    ("bg_remove", {"deskew": False, "perspective": False, "bg_remove": True, "binarize": False, "sharpen": False}),
    (
        "all",
        {"deskew": True, "perspective": True, "bg_remove": True, "binarize": True, "sharpen": True},
    ),
]


async def _extract(
    client: httpx.AsyncClient,
    url: str,
    image_path: Path,
    doc_type: str,
    preprocess: dict[str, bool],
) -> dict[str, Any]:
    # Prefer path inside container mount: /app/data/...
    # Host path under data/ maps to /app/data/
    payload: dict[str, Any] = {"doc_type": doc_type, "preprocess": preprocess}
    # Send file bytes — works regardless of container path mapping
    files = {"file": (image_path.name, image_path.read_bytes(), "application/octet-stream")}
    # Multipart cannot easily send nested preprocess; use JSON+base64 for ablation
    import base64

    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    resp = await client.post(
        url,
        json={"base64": encoded, "doc_type": doc_type, "preprocess": preprocess},
        timeout=900.0,
    )
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError("OCR response not an object")
    return data


async def run(config_path: Path | None = None, *, limit: int | None = None) -> list[dict[str, Any]]:
    cfg = load_config(config_path)
    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    endpoint = str(cfg["endpoints"]["ocr"]).rstrip("/")
    url = f"{endpoint}/v1/ocr/extract"
    records = load_ocr_manifest(
        data_root / "1_ocr",
        expected=(cfg.get("expected_counts") or {}).get("1_ocr"),
    )
    if limit is not None:
        records = records[:limit]
    if not records:
        raise RuntimeError("OCR manifest is empty")

    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=900.0) as client:
        meta = (await client.get(f"{endpoint}/v1/meta")).json()
        backend = str(meta.get("backend") or "unknown")
        for name, flags in ABLATIONS:
            field_cers: list[float] = []
            latencies: list[int] = []
            print(f"ablation={name} n={len(records)} ...", flush=True)
            for i, record in enumerate(records, start=1):
                image = data_root / "1_ocr" / record.path
                response = await _extract(client, url, image, record.doc_type, flags)
                text = str(response.get("text") or "")
                hyp_fields = response.get("fields") if isinstance(response.get("fields"), dict) else {}
                fcer, _ = field_cer(hyp_fields, record.gt_fields, page_hyp=text)
                if fcer is not None:
                    field_cers.append(float(fcer))
                latencies.append(int(response.get("latency_ms") or 0))
                if i % 20 == 0 or i == len(records):
                    print(f"  {name}: {i}/{len(records)}", flush=True)
            cer_field = sum(field_cers) / len(field_cers) if field_cers else None
            rows.append(
                {
                    "ablation": name,
                    **flags,
                    "cer_field": cer_field,
                    "n": len(records),
                    "n_with_gt_fields": len(field_cers),
                    "mean_latency_ms": sum(latencies) / len(latencies) if latencies else 0.0,
                    "backend": backend,
                }
            )
            print(f"  -> cer_field={cer_field}", flush=True)

    results_root.mkdir(parents=True, exist_ok=True)
    output_csv = results_root / "ocr_ablation.csv"
    fieldnames = [
        "ablation",
        "deskew",
        "perspective",
        "bg_remove",
        "binarize",
        "sharpen",
        "cer_field",
        "n",
        "n_with_gt_fields",
        "mean_latency_ms",
        "backend",
    ]
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (results_root / "ocr_ablation.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    baseline = next((r["cer_field"] for r in rows if r["ablation"] == "none"), None)
    recommendations: dict[str, Any] = {"baseline_cer_field": baseline, "drop": [], "keep": []}
    if baseline is not None:
        for row in rows:
            if row["ablation"] == "none":
                continue
            val = row["cer_field"]
            if val is None or val >= baseline - 1e-6:
                recommendations["drop"].append(row["ablation"])
            else:
                recommendations["keep"].append(row["ablation"])
    (results_root / "ocr_ablation_recommend.json").write_text(
        json.dumps(recommendations, indent=2) + "\n", encoding="utf-8"
    )
    print(f"recommend: {recommendations}", flush=True)
    print(f"wrote {output_csv}", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Ablate OCR preprocessing via API")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    asyncio.run(run(args.config, limit=args.limit))


if __name__ == "__main__":
    main()
