"""TC1 — Document OCR field Character Error Rate evaluation.

cer_page is removed. Gate / report metric is cer_field only.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rapidfuzz.distance import Levenshtein

from harness.client import ApiClient
from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.eval_errors import EvalTransportError, require_fields
from harness.freeze import collect_freeze
from harness.schema import (
    country_for_doc_type,
    load_ocr_manifest,
    origin_distribution,
    script_for_doc_type,
    tta_valid_for_records,
)


def cer(hyp: str, ref: str) -> float:
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    return Levenshtein.distance(hyp, ref) / len(ref)


def _best_window_cer(hyp: str, ref: str) -> tuple[float, str]:
    """Min CER of ref vs any hyp window near len(ref). Used when structured fields absent."""
    if len(ref) == 0:
        return (0.0 if len(hyp) == 0 else 1.0), ""
    if not hyp:
        return 1.0, ""
    if ref in hyp:
        return 0.0, ref
    n = len(ref)
    best = 1.0
    best_span = ""
    for w in range(max(1, n - 2), n + 3):
        if w > len(hyp):
            continue
        for i in range(0, len(hyp) - w + 1):
            span = hyp[i : i + w]
            d = cer(span, ref)
            if d < best:
                best = d
                best_span = span
                if best == 0.0:
                    return best, best_span
    return best, best_span


def _unwrap_hyp_fields(hyp_fields: dict) -> dict[str, str]:
    """Normalize API fields to {name: value_str}; skip null/no-bbox values."""
    out: dict[str, str] = {}
    for key, val in (hyp_fields or {}).items():
        if isinstance(val, dict):
            if val.get("bbox") is None or val.get("value") is None:
                continue
            out[str(key)] = str(val["value"])
        elif val is not None:
            out[str(key)] = str(val)
    return out


def field_cer(
    hyp_fields: dict,
    gt_fields: dict,
    *,
    page_hyp: str = "",
) -> tuple[float | None, list[dict]]:
    """Micro-averaged CER over gt_fields; key match first, else best window in page text."""
    if not gt_fields:
        return None, []
    rows: list[dict] = []
    total_edits = 0.0
    total_n = 0
    hyp_fields = _unwrap_hyp_fields(hyp_fields)
    for key, ref in gt_fields.items():
        ref_s = str(ref)
        hyp_s = hyp_fields.get(str(key), "")
        source = "fields"
        if not hyp_s:
            for hk, hv in hyp_fields.items():
                if hk.lower() == str(key).lower():
                    hyp_s = str(hv)
                    break
        if not hyp_s and page_hyp:
            item, hyp_s = _best_window_cer(page_hyp, ref_s)
            source = "page_window"
        else:
            item = cer(hyp_s, ref_s)
        rows.append({"field": key, "cer": item, "gt": ref_s, "hyp": hyp_s, "match": source})
        total_edits += item * max(len(ref_s), 1)
        total_n += max(len(ref_s), 1)
    return (total_edits / total_n) if total_n else 0.0, rows


def _aggregate_field_cer(items: list[dict]) -> float | None:
    field_items = [it for it in items if it.get("cer_field") is not None]
    if not field_items:
        return None
    return sum(float(it["cer_field"]) for it in field_items) / len(field_items)


def _stratified_by_country(records: list[Any], max_n: int, *, seed: int) -> list[Any]:
    """Take up to max_n records with equal per-country quotas; mix scan/photo inside each."""
    by_country: dict[str, list[Any]] = defaultdict(list)
    for rec in records:
        country = country_for_doc_type(rec.doc_type) or "unknown"
        by_country[country].append(rec)
    countries = sorted(by_country.keys())
    if not countries:
        return []
    rng = random.Random(seed)
    base = max_n // len(countries)
    rem = max_n % len(countries)
    selected: list[Any] = []
    for i, country in enumerate(countries):
        quota = base + (1 if i < rem else 0)
        pool = list(by_country[country])
        scans = [r for r in pool if "scan" in getattr(r, "id", "")]
        photos = [r for r in pool if "photo" in getattr(r, "id", "")]
        other = [r for r in pool if r not in scans and r not in photos]
        rng.shuffle(scans)
        rng.shuffle(photos)
        rng.shuffle(other)
        take: list[Any] = []
        # Alternate scan/photo to avoid Albania-scan-only bias.
        while len(take) < quota and (scans or photos or other):
            if scans and (len(take) % 2 == 0 or not photos):
                take.append(scans.pop())
            elif photos:
                take.append(photos.pop())
            elif other:
                take.append(other.pop())
            elif scans:
                take.append(scans.pop())
            else:
                break
        selected.extend(take[:quota])
    rng.shuffle(selected)
    return selected[:max_n]


async def run(cfg_path: Path | None = None, *, backend_tag: str | None = None) -> dict:
    cfg = load_config(cfg_path)
    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    expected = (cfg.get("expected_counts") or {}).get("1_ocr")
    records = load_ocr_manifest(data_root / "1_ocr", expected=expected)
    max_n = (cfg.get("ocr") or {}).get("eval_max_samples")
    sample_mode = str((cfg.get("ocr") or {}).get("eval_sample_mode") or "stratified").lower()
    seed = int(cfg.get("seed") or 42)
    if max_n is not None and int(max_n) > 0 and len(records) > int(max_n):
        if sample_mode in {"legacy", "head", "prefix"}:
            scans = [r for r in records if "scan" in r.id]
            photos = [r for r in records if "photo" in r.id]
            other = [r for r in records if r not in scans and r not in photos]
            half = int(max_n) // 2
            records = scans[:half] + photos[: int(max_n) - min(half, len(scans))]
            if len(records) < int(max_n):
                records = (scans + photos + other)[: int(max_n)]
            print(
                f"TC1 subsample (legacy): {len(records)} / manifest "
                f"(eval_max_samples={max_n})",
                flush=True,
            )
        else:
            records = _stratified_by_country(records, int(max_n), seed=seed)
            by_c: dict[str, int] = defaultdict(int)
            by_s: dict[str, int] = defaultdict(int)
            for rec in records:
                c = country_for_doc_type(rec.doc_type) or "unknown"
                s = rec.script or script_for_doc_type(rec.doc_type) or "unknown"
                by_c[c] += 1
                by_s[str(s)] += 1
            print(
                f"TC1 subsample (stratified): {len(records)} / manifest "
                f"(eval_max_samples={max_n} seed={seed})",
                flush=True,
            )
            print(f"  by_country={dict(sorted(by_c.items()))}", flush=True)
            print(f"  by_script={dict(sorted(by_s.items()))}", flush=True)
    tta_valid = tta_valid_for_records(records)
    origins = origin_distribution(records)
    target = float(cfg["targets"]["tc1_cer_max"])

    per_item: list[dict] = []
    fatals: list[str] = []
    by_script_items: dict[str, list[dict]] = defaultdict(list)
    by_country_items: dict[str, list[dict]] = defaultdict(list)
    lang_used = str((cfg.get("ocr") or {}).get("lang") or "en")
    lang_by_country = dict((cfg.get("ocr") or {}).get("lang_by_country") or {})
    timeout_s = float(cfg.get("timeout_seconds", 30))
    n_total = len(records)
    progress_every = max(1, int((cfg.get("ocr") or {}).get("eval_progress_every") or 1))
    results_root.mkdir(parents=True, exist_ok=True)
    progress_path = results_root / "tc1_cer_progress.jsonl"
    # Truncate so a fresh run is easy to `tail -f`
    progress_path.write_text("", encoding="utf-8")

    print(
        f"TC1 starting: n={n_total} timeout={timeout_s}s "
        f"endpoint={cfg['endpoints'].get('ocr')} progress={progress_path}",
        flush=True,
    )

    async with ApiClient(
        cfg["endpoints"],
        timeout_seconds=timeout_s,
        retry_count=int(cfg.get("retry_count", 3)),
    ) as client:
        for idx, rec in enumerate(records, start=1):
            img = data_root / "1_ocr" / rec.path
            country = country_for_doc_type(rec.doc_type) or "unknown"
            script = rec.script or script_for_doc_type(rec.doc_type) or "unknown"
            print(f"[{idx}/{n_total}] {rec.id} country={country} …", flush=True)
            try:
                resp, latency_ms = await client.ocr_extract(
                    img, record_id=rec.id, doc_type=rec.doc_type
                )
                require_fields(resp, ["fields"], context=rec.id)
                hyp = str(resp.get("text") or "")
                hyp_fields = resp.get("fields") if isinstance(resp.get("fields"), dict) else {}
            except Exception as e:
                fatals.append(f"{rec.id}: {e}")
                print(f"[{idx}/{n_total}] {rec.id} FAIL {e}", flush=True)
                with progress_path.open("a", encoding="utf-8") as pf:
                    pf.write(
                        json.dumps(
                            {"i": idx, "n": n_total, "id": rec.id, "ok": False, "error": str(e)},
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                continue
            fcer, field_rows = field_cer(hyp_fields, rec.gt_fields, page_hyp=hyp)
            item_lang = lang_by_country.get(country) or lang_used
            item = {
                "id": rec.id,
                "cer_field": fcer,
                "field_rows": field_rows,
                "script": script,
                "country": country,
                "doc_type": rec.doc_type,
                "lang_code_used": item_lang,
                "hyp_text": hyp,
                "latency_ms": latency_ms,
            }
            per_item.append(item)
            by_script_items[str(script)].append(item)
            by_country_items[str(country)].append(item)
            running = _aggregate_field_cer(per_item)
            if idx == 1 or idx == n_total or idx % progress_every == 0:
                print(
                    f"[{idx}/{n_total}] {rec.id} ok cer_field={fcer} "
                    f"running={running} ms={latency_ms:.0f} text_len={len(hyp)}",
                    flush=True,
                )
            with progress_path.open("a", encoding="utf-8") as pf:
                pf.write(
                    json.dumps(
                        {
                            "i": idx,
                            "n": n_total,
                            "id": rec.id,
                            "ok": True,
                            "cer_field": fcer,
                            "running_cer_field": running,
                            "latency_ms": latency_ms,
                            "country": country,
                            "script": script,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    if fatals:
        print(f"FATAL: {len(fatals)} OCR call(s) failed — metrics not computed", file=sys.stderr)
        for line in fatals[:50]:
            print(f"  {line}", file=sys.stderr)
        if len(fatals) > 50:
            print(f"  ... and {len(fatals) - 50} more", file=sys.stderr)
        raise EvalTransportError(f"{len(fatals)} fatal OCR failures")

    cer_field_overall = _aggregate_field_cer(per_item)
    cer_field_by_script: dict[str, Any] = {}
    for script, items in sorted(by_script_items.items()):
        cer_field_by_script[script] = {
            "cer_field": _aggregate_field_cer(items),
            "n": len(items),
            "n_with_gt_fields": sum(1 for it in items if it.get("cer_field") is not None),
        }

    diagnostics_rows: list[dict[str, Any]] = []
    for country, items in sorted(by_country_items.items()):
        script = items[0].get("script") if items else "unknown"
        row_lang = lang_by_country.get(country) or lang_used
        diagnostics_rows.append(
            {
                "country": country,
                "script": script,
                "n_samples": len(items),
                "cer_field": _aggregate_field_cer(items),
                "lang_code_used": row_lang,
            }
        )

    field_items = [it for it in per_item if it.get("cer_field") is not None]
    passed = cer_field_overall is not None and cer_field_overall < target
    backend_name = backend_tag or str((cfg.get("ocr") or {}).get("backend") or "unknown")

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    diagnostics = {
        "timestamp": ts,
        "n": len(records),
        "cer_field_overall": cer_field_overall,
        "backend": backend_name,
        "lang_default": lang_used,
        "lang_by_country": lang_by_country,
        "by_country": diagnostics_rows,
        "cer_field_by_script": cer_field_by_script,
    }
    diag_dir = data_root / "1_ocr"
    diag_dir.mkdir(parents=True, exist_ok=True)
    diag_path = diag_dir / f"diagnostics_by_country_{ts}.json"
    diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    payload = {
        "tc": "TC1",
        "metric": "cer_field",
        "value": cer_field_overall,
        "cer_field": cer_field_overall,
        "cer_field_by_script": cer_field_by_script,
        "cer_field_by_country": {r["country"]: r for r in diagnostics_rows},
        "diagnostics_path": str(diag_path),
        "backend": backend_name,
        "target_max": target,
        "passed": passed,
        "n": len(records),
        "n_with_gt_fields": len(field_items),
        "tta_valid": tta_valid,
        "origin_distribution": origins,
        "ocr_lang_note": "classic: lang per country; VL-1.6 multilingual (lang noted for audit)",
        "per_item": per_item,
        "freeze": collect_freeze(cfg),
    }

    suffix = f"_{backend_name}" if backend_tag else ""
    out = results_root / f"tc1_cer{suffix}.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    # Canonical path for report when this is the active backend
    if not backend_tag:
        (results_root / "tc1_cer.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    status = "PASS" if passed else "FAIL"
    print(f"TC1 cer_field = {cer_field_overall}  (target < {target})  [{status}]  backend={backend_name}")
    for script, row in cer_field_by_script.items():
        print(f"  script={script}: cer_field={row['cer_field']}  n={row['n']}")
    for row in diagnostics_rows:
        print(
            f"  country={row['country']} script={row['script']}: "
            f"cer_field={row['cer_field']}  n={row['n_samples']}  lang={row['lang_code_used']}"
        )
    print(f"wrote {out}")
    print(f"wrote {diag_path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TC1 cer_field")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--backend-tag", type=str, default=None, help="Write tc1_cer_<tag>.json")
    args = parser.parse_args()
    try:
        payload = asyncio.run(run(args.config, backend_tag=args.backend_tag))
    except EvalTransportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    sys.exit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
