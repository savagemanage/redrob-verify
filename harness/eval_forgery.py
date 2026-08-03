"""TC2 / TC3 — Forgery detection with in-domain vs cross-domain sweeps.

Design (DS-2):
  in_domain   — MIDV authentic + our ``gen_forgery.py`` forgeries
  cross_domain — MIDV authentic + FMIDV forgeries (independent generator)

Cross-domain must jointly satisfy TC2/TC3 targets for a real pass.
In-domain is reported for diagnostics (memorisation check).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from harness.client import ApiClient
from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.eval_errors import EvalTransportError, require_number
from harness.freeze import collect_freeze
from harness.schema import ForgeryRecord, load_forgery_manifest, origin_distribution, tta_valid_for_records
from harness.sweep import (
    f1,
    fpr,
    precision,
    sweep_thresholds,
    tpr,
    warn_thin_margins,
    write_sweep_csv,
)


def _domain_of(rec: ForgeryRecord) -> str:
    return rec.eval_domain or "in_domain"


def _sweep_domain(
    records: list[ForgeryRecord],
    by_id: dict[str, tuple[float, float]],
    *,
    tpr_min: float,
    f1_min: float,
    domain: str,
    results_root: Path,
) -> dict[str, Any]:
    scores: list[float] = []
    labels: list[int] = []
    raw_rows: list[dict] = []
    for rec in records:
        score, latency_ms = by_id[rec.id]
        scores.append(score)
        labels.append(int(rec.label))
        raw_rows.append(
            {
                "id": rec.id,
                "label": rec.label,
                "score": score,
                "origin": rec.origin,
                "fabrication": rec.fabrication,
                "tamper": rec.tamper,
                "eval_domain": domain,
                "generator": rec.generator,
                "latency_ms": latency_ms,
            }
        )

    if not scores:
        return {
            "domain": domain,
            "n": 0,
            "infeasible": True,
            "tc2_tpr": None,
            "tc3_f1": None,
            "warning": f"no scored rows for {domain}",
            "raw": [],
            "sweep": [],
        }

    sweep = sweep_thresholds(
        scores,
        labels,
        metric_fns={"tpr": tpr, "fpr": fpr, "precision": precision, "f1": f1},
        targets={"tpr": tpr_min, "f1": f1_min},
    )
    csv_path = results_root / f"tc2_tc3_sweep_{domain}.csv"
    write_sweep_csv(csv_path, sweep.points, ["tpr", "fpr", "precision", "f1"])

    print(f"\n=== {domain} (n={len(records)}) ===")
    print("threshold  TPR       FPR       Precision  F1")
    for p in sweep.points:
        if abs(round(p.threshold * 100)) % 10 == 0 or p.threshold in (
            sweep.t_min,
            sweep.t_max,
            sweep.recommended,
        ):
            print(
                f"  {p.threshold:4.2f}    "
                f"{p.metrics['tpr']:.4f}    "
                f"{p.metrics['fpr']:.4f}    "
                f"{p.metrics['precision']:.4f}    "
                f"{p.metrics['f1']:.4f}"
            )

    # ROC gate feel: must pass above (FPR=0.348, TPR=0.88)
    roc_gate = {"fpr_max": 0.348, "tpr_min": 0.88}
    roc_pass = any(
        p.metrics["fpr"] <= roc_gate["fpr_max"] and p.metrics["tpr"] >= roc_gate["tpr_min"]
        for p in sweep.points
    )
    # best-effort: maximise TPR-FPR (Youden) among points
    best_pt = max(sweep.points, key=lambda p: p.metrics["tpr"] - p.metrics["fpr"])
    best_effort = {
        "threshold": best_pt.threshold,
        "tpr": best_pt.metrics["tpr"],
        "fpr": best_pt.metrics["fpr"],
        "precision": best_pt.metrics["precision"],
        "f1": best_pt.metrics["f1"],
        "criterion": "max(TPR-FPR)",
        "roc_gate": roc_gate,
        "roc_gate_pass": roc_pass,
    }

    if sweep.infeasible:
        print(f"INFEASIBLE [{domain}]: TPR>={tpr_min} AND F1>={f1_min} not jointly met")
        print(
            f"best-effort t={best_effort['threshold']:.2f}  "
            f"TPR={best_effort['tpr']:.4f}  FPR={best_effort['fpr']:.4f}  "
            f"F1={best_effort['f1']:.4f}  roc_gate_pass={roc_pass}"
        )
        return {
            "domain": domain,
            "n": len(records),
            "origin_distribution": origin_distribution(records),
            "raw": raw_rows,
            "sweep": [{"threshold": p.threshold, **p.metrics} for p in sweep.points],
            "feasible_interval": None,
            "recommended_threshold": None,
            "recommended_metrics": None,
            "recommended_margins": None,
            "min_normalized_margin": sweep.min_normalized_margin,
            "infeasible": True,
            "tc2_tpr": None,
            "tc3_f1": None,
            "best_effort": best_effort,
            "sweep_csv": str(csv_path),
        }

    rec_tpr = sweep.recommended_metrics["tpr"]  # type: ignore[index]
    rec_f1 = sweep.recommended_metrics["f1"]  # type: ignore[index]
    margins = sweep.recommended_margins
    print(
        f"Feasible [{domain}]: [{sweep.t_min:.2f}, {sweep.t_max:.2f}]  "
        f"t*={sweep.recommended:.2f}  TPR={rec_tpr:.4f}  F1={rec_f1:.4f}"
    )
    warn_thin_margins(margins)
    return {
        "domain": domain,
        "n": len(records),
        "origin_distribution": origin_distribution(records),
        "raw": raw_rows,
        "sweep": [{"threshold": p.threshold, **p.metrics} for p in sweep.points],
        "feasible_interval": [sweep.t_min, sweep.t_max],
        "recommended_threshold": sweep.recommended,
        "recommended_metrics": {
            "tpr": rec_tpr,
            "precision": sweep.recommended_metrics["precision"],  # type: ignore[index]
            "f1": rec_f1,
        },
        "recommended_margins": margins,
        "min_normalized_margin": sweep.min_normalized_margin,
        "infeasible": False,
        "tc2_tpr": rec_tpr,
        "tc3_f1": rec_f1,
        "sweep_csv": str(csv_path),
    }


async def run(cfg_path: Path | None = None) -> dict:
    cfg = load_config(cfg_path)
    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    expected = (cfg.get("expected_counts") or {}).get("2_forgery")
    records = load_forgery_manifest(data_root / "2_forgery", expected=expected)
    tta_valid = tta_valid_for_records(records)
    origins = origin_distribution(records)

    tpr_min = float(cfg["targets"]["tc2_tpr_min"])
    f1_min = float(cfg["targets"]["tc3_f1_min"])

    fatals: list[str] = []
    sem = asyncio.Semaphore(32)

    async with ApiClient(
        cfg["endpoints"],
        timeout_seconds=float(cfg.get("timeout_seconds", 30)),
        retry_count=int(cfg.get("retry_count", 3)),
    ) as client:

        async def one(rec: ForgeryRecord):
            async with sem:
                img = data_root / "2_forgery" / rec.path
                resp, latency_ms = await client.forgery_detect(img, record_id=rec.id)
                score = require_number(resp, "score", context=rec.id)
                return rec, score, latency_ms

        results = []
        pending = [asyncio.create_task(one(rec)) for rec in records]
        done = 0
        for coro in asyncio.as_completed(pending):
            try:
                rec, score, latency_ms = await coro
                results.append((rec, score, latency_ms))
            except Exception as e:
                fatals.append(str(e))
            done += 1
            if done % 100 == 0 or done == len(records):
                print(f"  forgery inference {done}/{len(records)}", flush=True)

    if fatals:
        print(f"FATAL: {len(fatals)} forgery call(s) failed — metrics not computed", file=sys.stderr)
        for line in fatals[:50]:
            print(f"  {line}", file=sys.stderr)
        raise EvalTransportError(f"{len(fatals)} fatal forgery failures")

    by_id = {rec.id: (score, latency_ms) for rec, score, latency_ms in results}

    by_domain: dict[str, list[ForgeryRecord]] = {}
    for rec in records:
        by_domain.setdefault(_domain_of(rec), []).append(rec)

    has_explicit_split = any(r.eval_domain for r in records)
    domains = sorted(by_domain.keys())
    domain_results: dict[str, Any] = {}
    for domain in domains:
        domain_results[domain] = _sweep_domain(
            by_domain[domain],
            by_id,
            tpr_min=tpr_min,
            f1_min=f1_min,
            domain=domain,
            results_root=results_root,
        )

    # Gate: cross-domain is the real pass when present; else legacy single bucket
    if "cross_domain" in domain_results:
        gate = domain_results["cross_domain"]
        gate_name = "cross_domain"
    else:
        gate = domain_results.get("in_domain") or next(iter(domain_results.values()))
        gate_name = gate["domain"]
        if has_explicit_split is False and len(domains) == 1:
            print(
                "NOTE: no eval_domain=cross_domain rows — reporting single bucket "
                "(add FMIDV via tools/build_ds2_splits.py for true pass gate)",
                flush=True,
            )

    infeasible = bool(gate.get("infeasible"))
    # Also require cross_domain to exist for a "true" pass when config asks
    require_cross = bool((cfg.get("forgery") or {}).get("require_cross_domain", True))
    missing_cross = require_cross and "cross_domain" not in domain_results
    if missing_cross:
        infeasible = True
        print(
            "INFEASIBLE: cross-domain (FMIDV) eval set missing — "
            "in-domain alone cannot prove independence from gen_forgery artifacts",
            flush=True,
        )

    # Backward-compatible top-level TC2/TC3 fields = gate metrics
    payload = {
        "tc": ["TC2", "TC3"],
        "n": len(records),
        "tta_valid": tta_valid,
        "origin_distribution": origins,
        "targets": {"tpr_min": tpr_min, "f1_min": f1_min},
        "eval_design": {
            "in_domain": "MIDV authentic + gen_forgery.py forgeries",
            "cross_domain": "MIDV authentic + FMIDV forgeries (3rd-party)",
            "pass_gate": "cross_domain",
            "require_cross_domain": require_cross,
            "fmidv_tamper": "copy_move only; splice/inpaint remain gen_forgery",
        },
        "domains": domain_results,
        "gate_domain": gate_name,
        "missing_cross_domain": missing_cross,
        "raw": [row for d in domain_results.values() for row in d.get("raw") or []],
        "sweep": gate.get("sweep") or [],
        "feasible_interval": gate.get("feasible_interval"),
        "recommended_threshold": gate.get("recommended_threshold"),
        "recommended_metrics": gate.get("recommended_metrics"),
        "recommended_margins": gate.get("recommended_margins"),
        "min_normalized_margin": gate.get("min_normalized_margin"),
        "infeasible": infeasible,
        "tc2_tpr": gate.get("tc2_tpr"),
        "tc3_f1": gate.get("tc3_f1"),
        "in_domain": domain_results.get("in_domain"),
        "cross_domain": domain_results.get("cross_domain"),
        "freeze": collect_freeze(cfg),
    }

    # Legacy combined CSV = gate sweep
    csv_path = results_root / "tc2_tc3_sweep.csv"
    if gate.get("sweep"):
        lines = ["threshold,tpr,precision,f1\n"]
        for p in gate["sweep"]:
            lines.append(
                f"{p['threshold']:.2f},{p['tpr']:.6f},{p['precision']:.6f},{p['f1']:.6f}\n"
            )
        csv_path.write_text("".join(lines), encoding="utf-8")

    out = results_root / "tc2_tc3_forgery.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {csv_path}")
    if domain_results.get("in_domain"):
        print(
            f"in-domain:  "
            f"TPR={domain_results['in_domain'].get('tc2_tpr')}  "
            f"F1={domain_results['in_domain'].get('tc3_f1')}  "
            f"infeasible={domain_results['in_domain'].get('infeasible')}"
        )
    if domain_results.get("cross_domain"):
        print(
            f"cross-domain: "
            f"TPR={domain_results['cross_domain'].get('tc2_tpr')}  "
            f"F1={domain_results['cross_domain'].get('tc3_f1')}  "
            f"infeasible={domain_results['cross_domain'].get('infeasible')}"
        )
    print(f"PASS GATE = {gate_name}  infeasible={infeasible}")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TC2/TC3 forgery (in/cross domain)")
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()
    try:
        payload = asyncio.run(run(args.config))
    except EvalTransportError as e:
        print(str(e), file=sys.stderr)
        sys.exit(1)
    sys.exit(1 if payload["infeasible"] else 0)


if __name__ == "__main__":
    main()
