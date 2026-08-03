"""TC4 / TC5 — Face comparison evaluation with threshold sweep.

Does NOT apply a fixed verdict. Collects raw similarity scores, sweeps
0.00–1.00 @ 0.01, finds the joint-feasible interval for sensitivity and accuracy.
t* maximises the worst-case normalised margin among targets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from harness.client import ApiClient
from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.eval_errors import EvalTransportError, require_fields
from harness.freeze import collect_freeze
from harness.schema import load_face_manifest, origin_distribution, tta_valid_for_records
from harness.sweep import (
    accuracy,
    sweep_thresholds,
    tpr,
    warn_thin_margins,
    write_sweep_csv,
)


async def run(cfg_path: Path | None = None) -> dict:
    cfg = load_config(cfg_path)
    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    expected = (cfg.get("expected_counts") or {}).get("3_face")
    records = load_face_manifest(data_root / "3_face", expected=expected)
    tta_valid = tta_valid_for_records(records)
    origins = origin_distribution(records)
    under_variation = any(
        (getattr(r, "pair_warning", None) or "").startswith("under_variation")
        or (getattr(r, "pair_warning", None) == "single_image_augment_forbidden")
        for r in records
    )
    if any(getattr(r, "capture_a", None) and getattr(r, "capture_b", None) for r in records):
        # MIDV-style photo/scan/video cross pairs still under-vary for field thresholds
        under_variation = under_variation or any(
            {r.capture_a, r.capture_b} <= {"photo", "scan", "video_frame"}
            for r in records
            if r.same and r.capture_a and r.capture_b
        )

    sens_min = float(cfg["targets"]["tc4_sensitivity_min"])
    acc_min = float(cfg["targets"]["tc5_accuracy_min"])

    scores: list[float] = []
    labels: list[int] = []
    raw_rows: list[dict] = []
    fatals: list[str] = []
    null_count = 0

    sem = asyncio.Semaphore(4)

    async with ApiClient(
        cfg["endpoints"],
        timeout_seconds=float(cfg.get("timeout_seconds", 30)),
        retry_count=max(5, int(cfg.get("retry_count", 3))),
    ) as client:

        async def one(rec):
            async with sem:
                a = data_root / "3_face" / rec.img_a
                b = data_root / "3_face" / rec.img_b
                resp, latency_ms = await client.face_compare(a, b, record_id=rec.id)
                require_fields(resp, ["similarity"], context=rec.id)
                sim = resp["similarity"]
                if sim is not None:
                    try:
                        sim = float(sim)
                    except (TypeError, ValueError) as e:
                        raise EvalTransportError(
                            f"{rec.id}: similarity is not numeric: {resp['similarity']!r}"
                        ) from e
                return rec, sim, latency_ms, resp.get("reason"), resp.get("quality"), resp.get("backend")

        results = []
        pending = [asyncio.create_task(one(rec)) for rec in records]
        done = 0
        for coro in asyncio.as_completed(pending):
            try:
                item = await coro
                results.append(item)
            except Exception as e:
                fatals.append(str(e))
            done += 1
            if done % 100 == 0 or done == len(records):
                print(f"  face inference {done}/{len(records)}", flush=True)

    if fatals:
        print(f"FATAL: {len(fatals)} face call(s) failed — metrics not computed", file=sys.stderr)
        for line in fatals[:50]:
            print(f"  {line}", file=sys.stderr)
        raise EvalTransportError(f"{len(fatals)} fatal face failures")

    by_id = {
        rec.id: (sim, latency_ms, reason, quality, backend)
        for rec, sim, latency_ms, reason, quality, backend in results
    }
    for rec in records:
        sim, latency_ms, reason, quality, backend = by_id[rec.id]
        raw_rows.append(
            {
                "id": rec.id,
                "same": rec.same,
                "similarity": sim,
                "reason": reason,
                "quality": quality,
                "backend": backend,
                "latency_ms": latency_ms,
            }
        )
        if sim is None:
            null_count += 1
            continue
        scores.append(float(sim))
        labels.append(1 if rec.same else 0)

    if null_count:
        print(f"  null_count={null_count} (no-face / legitimate null similarity)", flush=True)
    if not scores:
        print("=" * 60)
        print("INFEASIBLE: no scored pairs (all similarity=null)")
        print("=" * 60)
        payload = {
            "tc": ["TC4", "TC5"],
            "n": len(records),
            "tta_valid": tta_valid,
            "origin_distribution": origins,
            "n_scored": 0,
            "null_count": null_count,
            "under_variation_warning": under_variation,
            "threshold_lock": "blocked" if under_variation else "ok",
            "warning": (
                "변이 과소, 임계값 확정 불가"
                if under_variation
                else None
            ),
            "targets": {"sensitivity_min": sens_min, "accuracy_min": acc_min},
            "raw": raw_rows,
            "infeasible": True,
            "tc4_sensitivity": None,
            "tc5_accuracy": None,
            "freeze": collect_freeze(cfg),
        }
        out = results_root / "tc4_tc5_face.json"
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"wrote {out}")
        return payload

    sweep = sweep_thresholds(
        scores,
        labels,
        metric_fns={"sensitivity": tpr, "accuracy": accuracy},
        targets={"sensitivity": sens_min, "accuracy": acc_min},
    )

    csv_path = results_root / "tc4_tc5_sweep.csv"
    write_sweep_csv(csv_path, sweep.points, ["sensitivity", "accuracy"])

    print("threshold  Sensitivity  Accuracy")
    for p in sweep.points:
        if abs(round(p.threshold * 100)) % 10 == 0 or p.threshold in (
            sweep.t_min,
            sweep.t_max,
            sweep.recommended,
        ):
            print(
                f"  {p.threshold:4.2f}    "
                f"{p.metrics['sensitivity']:.4f}      "
                f"{p.metrics['accuracy']:.4f}"
            )

    margins = None
    if under_variation:
        print("WARNING: 변이 과소, 임계값 확정 불가 (calibration-only — do not freeze)", flush=True)

    # Metrics gate vs threshold lock are separate: MIDV can clear TC4/TC5 numbers
    # while still blocking freeze / DS-3 trial lock.
    if sweep.infeasible:
        print()
        print("=" * 60)
        print("INFEASIBLE: no threshold simultaneously satisfies")
        print(f"  TC4 sensitivity >= {sens_min}  AND  TC5 accuracy >= {acc_min}")
        print("=" * 60)
        rec_sens = rec_acc = None
        recommended = None
        interval = None
        margins = None
        metrics_infeasible = True
    else:
        recommended = sweep.recommended
        interval = [sweep.t_min, sweep.t_max]
        rec_sens = sweep.recommended_metrics["sensitivity"]  # type: ignore[index]
        rec_acc = sweep.recommended_metrics["accuracy"]  # type: ignore[index]
        margins = sweep.recommended_margins
        metrics_infeasible = False
        print()
        if under_variation:
            print("=" * 60)
            print("THRESHOLD LOCK BLOCKED: 변이 과소, 임계값 확정 불가")
            print("=" * 60)
            print("(metrics below are feel/calibration — not DS-3 trial lock)")
        print(f"Feasible interval: [{sweep.t_min:.2f}, {sweep.t_max:.2f}]")
        print(f"Recommended threshold t*: {recommended:.2f}  (maximin normalised margin)")
        print(
            f"  TC4 sensitivity = {rec_sens:.6f}  (target >= {sens_min})  "
            f"margin={margins['sensitivity']:.6f}"  # type: ignore[index]
        )
        print(
            f"  TC5 accuracy    = {rec_acc:.6f}  (target >= {acc_min})  "
            f"margin={margins['accuracy']:.6f}"  # type: ignore[index]
        )
        print(f"  min normalised margin = {sweep.min_normalized_margin:.6f}")
        warn_thin_margins(margins)

    payload = {
        "tc": ["TC4", "TC5"],
        "n": len(records),
        "tta_valid": tta_valid,
        "origin_distribution": origins,
        "n_scored": len(scores),
        "null_count": null_count,
        "under_variation_warning": under_variation,
        "threshold_lock": "blocked" if under_variation else "ok",
        "warning": ("변이 과소, 임계값 확정 불가" if under_variation else None),
        "targets": {"sensitivity_min": sens_min, "accuracy_min": acc_min},
        "raw": raw_rows,
        "sweep": [{"threshold": p.threshold, **p.metrics} for p in sweep.points],
        "feasible_interval": interval,
        "recommended_threshold": recommended,
        "recommended_metrics": (
            None
            if recommended is None
            else {"sensitivity": rec_sens, "accuracy": rec_acc}
        ),
        "recommended_margins": margins,
        "min_normalized_margin": None if recommended is None else sweep.min_normalized_margin,
        # Exit/report gate: joint metric targets only. Freeze still respects threshold_lock.
        "infeasible": metrics_infeasible,
        "tc4_sensitivity": rec_sens,
        "tc5_accuracy": rec_acc,
        "freeze": collect_freeze(cfg),
    }

    out = results_root / "tc4_tc5_face.json"
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {csv_path}")

    if rec_sens is not None and rec_acc is not None:
        print(f"TC4 sensitivity = {rec_sens:.6f}  (target >= {sens_min})")
        print(f"TC5 accuracy    = {rec_acc:.6f}  (target >= {acc_min})")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TC4/TC5 face with threshold sweep")
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
