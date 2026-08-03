#!/usr/bin/env python3
"""Full-scale *rendered* local data — never writes 1×1 placeholder pixels.

Delegates to:
  - tools/gen_dev_fixtures.py   → results/dev_fixtures
  - tools/gen_indian_docs.py    → results/indian_docs
  - tools/gen_resumes.py        → data/4_resume

Eval ``data_root``: ``data`` (``1_ocr`` … ``4_resume``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harness.config_util import REPO_ROOT, load_config
from tools.gen_dev_fixtures import generate as gen_fixtures
from tools.gen_indian_docs import generate as gen_indian


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate rendered local datasets (no 1×1 placeholders)"
    )
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "config.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--skip-resumes",
        action="store_true",
        help="Skip GitHub-anchored resume generation (needs network)",
    )
    parser.add_argument(
        "--n-resume",
        type=int,
        default=None,
        help="Resume count override (default: config expected_counts.4_resume)",
    )
    parser.add_argument(
        "--n-ocr-per-type",
        type=int,
        default=None,
        help="Indian docs per template type (4 types). Default: ceil(1_ocr/4)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.get("seed", 42))
    counts = cfg.get("expected_counts") or {}
    n_ocr = int(counts.get("1_ocr", 300))
    n_per_type = args.n_ocr_per_type
    if n_per_type is None:
        n_per_type = max(1, (n_ocr + 3) // 4)
    n_resume = int(args.n_resume if args.n_resume is not None else counts.get("4_resume", 100))

    print("=== gen_dev_fixtures (rendered) ===", flush=True)
    fixture_counts = gen_fixtures(
        seed,
        n_ocr=min(40, n_ocr),
        n_face=50,
        n_forgery=50,
        n_resume=min(20, n_resume),
    )
    print(json.dumps({"dev_fixtures": fixture_counts}, indent=2))

    print(f"=== gen_indian_docs n_per_type={n_per_type} ===", flush=True)
    indian = gen_indian(n_per_type=n_per_type, seed=seed)
    print(json.dumps({"indian_docs": indian}, indent=2))

    if args.skip_resumes:
        print("skip resumes (--skip-resumes)", flush=True)
    else:
        print(f"=== gen_resumes n={n_resume} ===", flush=True)
        from tools.gen_resumes import generate as gen_resumes

        resumes = gen_resumes(n=n_resume, seed=seed)
        print(json.dumps({"resumes": resumes}, indent=2))

    print("done. data_root for MIDV: data", flush=True)


if __name__ == "__main__":
    main()
