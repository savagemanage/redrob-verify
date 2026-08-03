# redrob-verify

Open evaluation harness and reference microservices for **document OCR**,
**forgery detection**, **face matching**, and **developer-identity aggregation**.

Each capability is an HTTP service with a shared preflight contract (`/v1/meta`).
A host-side harness runs reproducible metrics and writes JSON (+ optional HTML report).

## Features

| Module | Service port | Metric focus |
|--------|--------------|--------------|
| Document OCR | `:8001` | Field-level character error rate (`cer_field`) |
| Forgery detect | `:8003` | Score-based TPR / F1 (threshold sweep) |
| Face compare | `:8002` | Sensitivity / accuracy (threshold sweep) |
| Identity aggregate | `:8004` | End-to-end latency over public profiles |

- Dockerized stacks with **GPU** support for OCR and forgery training/inference
- Provenance-aware manifests (`origin`, freeze checks)
- Apache-2.0–oriented model choices (see `LICENSES.md`)

## Architecture

```
┌────────────┐     ┌─────┐  ┌──────┐  ┌─────────┐  ┌──────────┐
│  harness   │────▶│ OCR │  │ Face │  │ Forgery │  │ Identity │
│ eval_*.py  │     │8001 │  │ 8002 │  │  8003   │  │   8004   │
└────────────┘     └─────┘  └──────┘  └─────────┘  └──────────┘
       │                ▲
       │                │  optional NestJS gateway :8000
       ▼
  results/*.json  report.html
```

Default OCR backend: **PaddleOCR-VL-1.6**. Face: **OpenCV Zoo YuNet + SFace**.

## Requirements

- Python 3.11+ and [uv](https://docs.astral.sh/uv/)
- Docker with NVIDIA Container Toolkit (for GPU OCR / forgery)
- Optional: Java 11+ and [Apache JMeter](https://jmeter.apache.org/) 5.6+ (identity latency eval)

## Quick start

```bash
git clone https://github.com/savagemanage/redrob-verify.git
cd redrob-verify
chmod +x run.sh tools/bootstrap_gpu.sh tools/fetch_models.sh

# One-shot on a GPU machine: deps → models → MIDV data → compose up → OCR smoke
./run.sh bootstrap-gpu
```

If `data/` is already populated:

```bash
SKIP_MIDV=1 ./run.sh bootstrap-gpu
```

Manual steps:

```bash
cp .env.example .env   # optional GITHUB_TOKEN / OPENAI_API_KEY
./run.sh setup
./run.sh fetch-models
./run.sh fetch-midv && ./run.sh ingest-midv
./run.sh up
./run.sh smoke-ocr
./run.sh preflight
```

## Datasets

Images and archives are **not** in git (see `.gitignore`). Manifests and READMEs are.

| Path | Role | How to obtain |
|------|------|----------------|
| `data/1_ocr` | OCR eval | MIDV-2020 via `./run.sh fetch-midv` + `ingest-midv` |
| `data/2_forgery` | Forgery eval/train | MIDV authentic + `./run.sh gen-forgery` |
| `data/3_face` | Face pairs | Built during MIDV ingest / pair tools |
| `data/4_resume` | Identity latency | `./run.sh gen-resumes` |

MIDV-2020 author FTP: `ftp://smartengines.com/midv-2020`  
Citation: Bulatov et al., 2021, [arXiv:2107.00396](https://arxiv.org/abs/2107.00396).  
Follow each dataset’s `license.txt`. Details: `LICENSES.md`, `data/README.md`.

## Evaluation

```bash
./run.sh eval-cer        # OCR cer_field
./run.sh eval-forgery
./run.sh eval-face
./run.sh eval-tc6        # requires JMeter
./run.sh report          # → results/report.html (gitignored)
```

Targets and seeds live in `config.yaml`. For full OCR eval keep
`ocr.eval_max_samples: null`.

## Configuration

| File | Purpose |
|------|---------|
| `config.yaml` | Endpoints, metric targets, seeds, OCR backend |
| `.env` | Optional secrets (never commit; use `.env.example`) |
| `docker-compose.yml` | Per-service images (Paddle / Torch / OpenCV isolated) |

Useful commands: `./run.sh help`

## Development

```bash
./run.sh test
./run.sh freeze --strict   # provenance / count gates
```

## License

Code: **Apache License 2.0** — see `LICENSE`.

Third-party models and datasets (PaddleOCR-VL, OpenCV Zoo, MIDV-2020, …) have
their own terms; inventory and constraints are in `LICENSES.md`. Weights and
raw images are downloaded locally and must not be committed.

## Contributing

Issues and PRs are welcome. Please:

1. Do not commit `.env`, model weights, or personal document images.
2. Keep new model dependencies documented in `LICENSES.md`.
3. Prefer `./run.sh test` and `./run.sh smoke-ocr` before opening a PR that
   touches OCR or Docker.

## Disclaimer

This repository is a **research / evaluation harness**, not a turnkey production
KYC product. Validate licenses and data rights for your own deployment.
