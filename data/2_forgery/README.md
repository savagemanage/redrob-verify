# TC2 / TC3 — forgery

Authentic scans + forged samples. See `tools/gen_forgery.py`, `tools/ingest_midv.py`,
`tools/build_ds2_splits.py`. Images gitignored; commit `manifest.jsonl` only if desired.

## Document-disjoint holdout

Default Apache ForgeryNet training should **not** reuse the same authentic document
IDs in train and eval:

```bash
./run.sh split-forgery-holdout --seed 42 --train-n 400 --eval-n 100 \
  --regenerate-train --rebuild-eval
```

Writes `holdout_split.json`, rebuilds `data/2_forgery_gen/` from train docs only, and
shrinks the eval manifest to **n=200** (100 auth + 100 forged). Update
`config.yaml` `expected_counts.2_forgery` accordingly.

Multi-seed stability: `tools/holdout_seed_sweep.py`. Leakage check:
`tools/check_holdout_leakage.py`.
