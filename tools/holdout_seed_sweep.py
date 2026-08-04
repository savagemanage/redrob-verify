#!/usr/bin/env python3
"""Multi-seed document-disjoint holdout: split → regen → train → eval.

Judges stability of TC2/TC3 near the 0.88 / 0.79 gates by reporting
per-seed TPR/F1 and the **minimum** across seeds.

Host orchestration (GPU). Example::

    # Already have seed 42 results; run four more:
    python tools/holdout_seed_sweep.py --seeds 7,13,99,123 --epochs 40

    # Include prior seed-42 row without re-running:
    python tools/holdout_seed_sweep.py --seeds 7,13,99,123 \\
        --prior-json results/holdout_seed_42.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from harness.config_util import REPO_ROOT

RESULTS = REPO_ROOT / "results"
LOGS = REPO_ROOT / "logs"
MODELS = REPO_ROOT / "models" / "forgery"


def _run(cmd: list[str], *, log: Path, env: dict | None = None, check: bool = True) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"+ {' '.join(cmd)}", flush=True)
    with log.open("a", encoding="utf-8") as fh:
        fh.write(f"\n>>> {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env or os.environ.copy(),
            stdout=fh,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)} (see {log})")
    return proc.returncode


def _extract_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n": data.get("n"),
        "infeasible": bool(data.get("infeasible")),
        "feasible_interval": data.get("feasible_interval"),
        "recommended_threshold": data.get("recommended_threshold"),
        "tc2_tpr": data.get("tc2_tpr"),
        "tc3_f1": data.get("tc3_f1"),
        "recommended_metrics": data.get("recommended_metrics"),
        "recommended_margins": data.get("recommended_margins"),
    }


def run_one_seed(seed: int, *, epochs: int, train_n: int, eval_n: int) -> dict:
    t0 = time.time()
    log = LOGS / f"holdout_seed_{seed}.log"
    if log.exists():
        log.unlink()
    print(f"\n======== seed={seed} ========", flush=True)

    _run(
        [
            sys.executable,
            "tools/split_forgery_holdout.py",
            "--seed",
            str(seed),
            "--train-n",
            str(train_n),
            "--eval-n",
            str(eval_n),
            "--regenerate-train",
            "--rebuild-eval",
            "--no-backup",
        ],
        log=log,
    )

    cache = REPO_ROOT / "data" / "_cache" / f"forgery_320_seed{seed}"
    if cache.exists():
        shutil.rmtree(cache, ignore_errors=True)
    ckpt = MODELS / f"forgerynet_holdout_seed{seed}.pth"

    train_cmd = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "--name",
        f"forgery-train-seed{seed}",
        "-v",
        f"{REPO_ROOT}:/workspace",
        "-w",
        "/workspace",
        "-e",
        "PYTHONPATH=/workspace",
        "-e",
        "PYTHONUNBUFFERED=1",
        "redrob-verify-forgery",
        "python",
        "-m",
        "services.forgery.train",
        "--config",
        "config.yaml",
        "--epochs",
        str(epochs),
        "--batch-size",
        "32",
        "--image-size",
        "320",
        "--output",
        str(Path("models/forgery") / ckpt.name),
        "--cache-dir",
        str(Path("data/_cache") / f"forgery_320_seed{seed}"),
    ]
    _run(train_cmd, log=log)

    train_meta = json.loads((RESULTS / "forgery_train.json").read_text(encoding="utf-8"))
    # Promote to serving path + recreate
    serving = MODELS / "forgerynet_apache.pth"
    subprocess.run(["sudo", "cp", str(ckpt), str(serving)], check=True, cwd=str(REPO_ROOT))
    subprocess.run(["sudo", "cp", str(ckpt), str(MODELS / "best.pth")], check=True, cwd=str(REPO_ROOT))
    _run(["docker", "compose", "up", "-d", "--force-recreate", "forgery"], log=log)
    time.sleep(5)

    eval_out = RESULTS / f"tc2_tc3_forgery_seed{seed}.json"
    _run(
        [
            sys.executable,
            "-m",
            "harness.eval_forgery",
            "--config",
            "config.yaml",
        ],
        log=log,
        check=False,  # infeasible → exit 1; still record metrics
    )
    default_eval = RESULTS / "tc2_tc3_forgery.json"
    if not default_eval.exists():
        raise RuntimeError(f"missing {default_eval} after eval")
    shutil.copy2(default_eval, eval_out)

    metrics = _extract_metrics(eval_out)
    row = {
        "seed": seed,
        "train_n": train_n,
        "eval_n": eval_n,
        "epochs": epochs,
        "checkpoint": str(ckpt),
        "best_auc": train_meta.get("best_auc"),
        "best_joint_interval_width": train_meta.get("best_joint_interval_width"),
        "elapsed_s": round(time.time() - t0, 1),
        **metrics,
    }
    out = RESULTS / f"holdout_seed_{seed}.json"
    out.write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(row, indent=2), flush=True)
    return row


def summarize(rows: list[dict]) -> dict:
    tprs = [float(r["tc2_tpr"]) for r in rows if r.get("tc2_tpr") is not None]
    f1s = [float(r["tc3_f1"]) for r in rows if r.get("tc3_f1") is not None]
    infeas = [bool(r.get("infeasible")) for r in rows]
    summary = {
        "n_seeds": len(rows),
        "seeds": [r["seed"] for r in rows],
        "tpr": {"values": tprs, "min": min(tprs) if tprs else None, "max": max(tprs) if tprs else None,
                "mean": sum(tprs) / len(tprs) if tprs else None},
        "f1": {"values": f1s, "min": min(f1s) if f1s else None, "max": max(f1s) if f1s else None,
               "mean": sum(f1s) / len(f1s) if f1s else None},
        "any_infeasible": any(infeas),
        "gates": {"tpr_min_target": 0.88, "f1_min_target": 0.79},
        "pass_by_minimum": (
            bool(tprs)
            and bool(f1s)
            and not any(infeas)
            and min(tprs) >= 0.88
            and min(f1s) >= 0.79
        ),
        "rows": rows,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=str, default="7,13,99,123", help="comma-separated seeds to run")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--train-n", type=int, default=400)
    parser.add_argument("--eval-n", type=int, default=100)
    parser.add_argument(
        "--prior-json",
        type=Path,
        nargs="*",
        default=[],
        help="existing holdout_seed_*.json rows to include in the summary",
    )
    parser.add_argument(
        "--summary-out",
        type=Path,
        default=RESULTS / "holdout_seed_sweep.json",
    )
    args = parser.parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    rows: list[dict] = []
    for path in args.prior_json:
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    for seed in seeds:
        rows.append(
            run_one_seed(seed, epochs=args.epochs, train_n=args.train_n, eval_n=args.eval_n)
        )
        # incremental summary
        summary = summarize(rows)
        args.summary_out.parent.mkdir(parents=True, exist_ok=True)
        args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(
            f"so_far min_TPR={summary['tpr']['min']} min_F1={summary['f1']['min']} "
            f"pass_by_min={summary['pass_by_minimum']}",
            flush=True,
        )
    summary = summarize(rows)
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print("\n=== FINAL (min-based) ===", flush=True)
    print(json.dumps({k: summary[k] for k in ("n_seeds", "seeds", "tpr", "f1", "any_infeasible", "pass_by_minimum")}, indent=2))
    print(f"wrote {args.summary_out}", flush=True)
    raise SystemExit(0 if summary["pass_by_minimum"] else 2)


if __name__ == "__main__":
    main()
