#!/usr/bin/env python3
"""Calibrate face thresholds on 3_face using the same sweep logic as eval_face.

Writes results/calib_face_{backend}.json so each embedding backend is calibrated separately.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.config_util import load_config, resolve_data_root, resolve_results_root  # noqa: E402
from harness.schema import load_face_manifest  # noqa: E402
from harness.sweep import (  # noqa: E402
    accuracy,
    sweep_thresholds,
    tpr,
    warn_thin_margins,
    write_sweep_csv,
)
from services.face.backends.factory import create_backend  # noqa: E402
from services.face.pipeline import FacePipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate face similarity thresholds")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="Override face.backend (stub | sface)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    face_cfg = dict(cfg.get("face") or {})
    face_cfg.setdefault("seed", cfg.get("seed", 42))
    if args.backend:
        face_cfg["backend"] = args.backend

    backend = create_backend(face_cfg, repo_root=REPO_ROOT)
    pipeline = FacePipeline(
        backend,
        quality_cfg=face_cfg.get("quality") or {},
        detect_cfg=face_cfg.get("detect") or {},
    )

    data_root = resolve_data_root(cfg)
    results_root = resolve_results_root(cfg)
    expected = (cfg.get("expected_counts") or {}).get("3_face")
    records = load_face_manifest(data_root / "3_face", expected=expected)

    sens_min = float(cfg["targets"]["tc4_sensitivity_min"])
    acc_min = float(cfg["targets"]["tc5_accuracy_min"])

    scores: list[float] = []
    labels: list[int] = []
    raw: list[dict] = []
    skipped = 0

    for i, rec in enumerate(records, start=1):
        a = data_root / "3_face" / rec.img_a
        b = data_root / "3_face" / rec.img_b
        out = pipeline.compare(str(a), str(b))
        sim = out.get("similarity")
        row = {
            "id": rec.id,
            "same": rec.same,
            "similarity": sim,
            "reason": out.get("reason"),
            "quality": out.get("quality"),
            "latency_ms": out.get("latency_ms"),
        }
        raw.append(row)
        if sim is None:
            skipped += 1
        else:
            scores.append(float(sim))
            labels.append(1 if rec.same else 0)
        if i % 100 == 0 or i == len(records):
            print(f"  calibrate {backend.name} {i}/{len(records)}", flush=True)

    payload: dict = {
        "backend": backend.name,
        "license": backend.license,
        "n": len(records),
        "n_scored": len(scores),
        "n_skipped_null": skipped,
        "targets": {"sensitivity_min": sens_min, "accuracy_min": acc_min},
        "raw": raw,
    }

    if len(scores) == 0:
        payload["infeasible"] = True
        payload["reason"] = "no scored pairs (all similarity=null)"
        print("INFEASIBLE: no scored pairs")
    else:
        sweep = sweep_thresholds(
            scores,
            labels,
            metric_fns={"sensitivity": tpr, "accuracy": accuracy},
            targets={"sensitivity": sens_min, "accuracy": acc_min},
        )
        csv_path = results_root / f"calib_face_{backend.name}_sweep.csv"
        write_sweep_csv(csv_path, sweep.points, ["sensitivity", "accuracy"])
        payload.update(
            {
                "sweep": [{"threshold": p.threshold, **p.metrics} for p in sweep.points],
                "feasible_interval": (
                    None if sweep.infeasible else [sweep.t_min, sweep.t_max]
                ),
                "recommended_threshold": sweep.recommended,
                "recommended_metrics": sweep.recommended_metrics,
                "recommended_margins": sweep.recommended_margins,
                "min_normalized_margin": sweep.min_normalized_margin,
                "infeasible": sweep.infeasible,
                "sweep_csv": str(csv_path),
            }
        )
        if sweep.infeasible:
            print("=" * 60)
            print("INFEASIBLE: no threshold jointly satisfies TC4/TC5 targets")
            print("=" * 60)
        else:
            print(
                f"Feasible [{sweep.t_min:.2f}, {sweep.t_max:.2f}]  "
                f"t*={sweep.recommended:.2f}  margins={sweep.recommended_margins}"
            )
            warn_thin_margins(sweep.recommended_margins)

    out_path = results_root / f"calib_face_{backend.name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {out_path}")
    sys.exit(1 if payload.get("infeasible") else 0)


if __name__ == "__main__":
    main()
