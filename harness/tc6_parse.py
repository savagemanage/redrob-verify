"""Parse JMeter JTL (CSV) results into TC6 JSON summary."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from harness.config_util import load_config, resolve_data_root, resolve_results_root
from harness.freeze import collect_freeze
from harness.schema import load_resume_manifest, origin_distribution, tta_valid_for_records


def parse_jtl(jtl_path: Path) -> dict:
    """Parse JMeter CSV JTL into latency list (seconds) and summary stats."""
    latencies_ms: list[float] = []
    successes = 0
    failures = 0

    with jtl_path.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Standard JMeter CSV: elapsed (ms), success (true/false)
            elapsed = row.get("elapsed") or row.get("Latency") or row.get("elapsed_ms")
            if elapsed is None:
                continue
            try:
                ms = float(elapsed)
            except ValueError:
                continue
            latencies_ms.append(ms)
            ok = str(row.get("success", "true")).lower() in ("true", "1", "yes")
            if ok:
                successes += 1
            else:
                failures += 1

    latencies_s = [ms / 1000.0 for ms in latencies_ms]
    n = len(latencies_s)
    mean_s = sum(latencies_s) / n if n else 0.0
    return {
        "n": n,
        "successes": successes,
        "failures": failures,
        "latencies_seconds": latencies_s,
        "mean_seconds": mean_s,
    }


def write_tc6_json(
    parsed: dict,
    *,
    target_max: float,
    out_path: Path,
    freeze: dict | None = None,
    tta_valid: bool = True,
    origin_distribution: dict[str, int] | None = None,
) -> dict:
    mean_s = float(parsed["mean_seconds"])
    payload = {
        "tc": "TC6",
        "metric": "mean_identity_seconds",
        "value": mean_s,
        "target_max": target_max,
        "passed": mean_s < target_max,
        "n": parsed["n"],
        "tta_valid": tta_valid,
        "origin_distribution": origin_distribution or {},
        "successes": parsed.get("successes"),
        "failures": parsed.get("failures"),
        "latencies_seconds": parsed["latencies_seconds"],
        "freeze": freeze or {},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse TC6 JMeter JTL → JSON")
    parser.add_argument("jtl", type=Path, help="Input .jtl (CSV) path")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSON path (default: results/tc6_identity.json)",
    )
    parser.add_argument("--config", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_root = resolve_results_root(cfg)
    data_root = resolve_data_root(cfg)
    out = args.output or (results_root / "tc6_identity.json")
    target = float(cfg["targets"]["tc6_seconds_max"])
    expected = (cfg.get("expected_counts") or {}).get("4_resume")
    resumes = load_resume_manifest(data_root / "4_resume", expected=expected)

    parsed = parse_jtl(args.jtl)
    payload = write_tc6_json(
        parsed,
        target_max=target,
        out_path=out,
        freeze=collect_freeze(cfg),
        tta_valid=tta_valid_for_records(resumes),
        origin_distribution=origin_distribution(resumes),
    )

    status = "PASS" if payload["passed"] else "FAIL"
    print(
        f"TC6 mean identity time = {payload['value']:.6f}s  "
        f"(target < {target})  n={payload['n']}  [{status}]"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
