# data/

Harness datasets live here. **Images and archives are gitignored**; only
`manifest.jsonl`, `README.md`, and `meta.json` may be committed.

```
data/
  1_ocr/          # TC1 document OCR (MIDV-2020 ingest)
  2_forgery/      # TC2 / TC3 forgery detect
  3_face/         # TC4 / TC5 face pairs
  4_resume/       # TC6 identity resumes
```

`config.yaml` → `data_root: data`

## Populate

```bash
./run.sh bootstrap-gpu          # recommended on a fresh GPU host
# or:
./run.sh fetch-midv && ./run.sh ingest-midv
./run.sh gen-fixtures           # tiny smoke only (no MIDV)
```

Do not commit PII-bearing scans, field-collected photos, or FTP archives.
Archives download under `results/midv_archives/` (also gitignored).
