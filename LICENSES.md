# Third-party licenses and data sources

This file records **allowed** and **forbidden** dependencies for this project.
Model weights and dataset archives are **not** committed; fetch them with the
tools below and keep them out of git.

## Face — OpenCV Zoo (Apache-2.0)

Production face backend: **YuNet** detector + **SFace** recognizer.

| File | Role | SHA-256 |
|------|------|---------|
| `models/face/face_detection_yunet_2023mar.onnx` | YuNet | 8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4 |
| `models/face/face_recognition_sface_2021dec.onnx` | SFace | 0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79 |

- https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx
- https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx

Fetch: `./tools/fetch_models.sh` / `./run.sh fetch-models`.

### Do not use (license / redistribution risk)

| Project | Issue |
|---------|--------|
| InsightFace pretrained packs (`buffalo_*`, `antelopev2`) | Non-commercial / paid commercial terms on weights |
| FaceNet pretrained weights (VGGFace2 / CASIA-WebFace lineage) | Academic-use training data constraints |
| CompreFace bundled weights | Redistribution clarity unresolved |
| Cloud face APIs as the sole backend | Not suitable for offline / on-prem eval |

## OCR — PaddleOCR-VL (Apache-2.0)

- **PaddleOCR-VL-1.6** + **ERNIE-4.5-0.3B** + **PP-DocLayoutV3** are treated as
  Apache-2.0 for the released artifacts used here. Confirm upstream notices when
  upgrading.
- Pin serving to **1.6** (`PaddlePaddle/PaddleOCR-VL-1.6`, `pipeline_version=v1.6`).
  Do not auto-float to “latest”.
- Expose `model_version` and `model_sha256` on `/v1/meta`.

Classic PaddleOCR remains available for A/B (`paddleocr_classic`).

## MIDV-2020

- Citation: Bulatov et al., 2021, [arXiv:2107.00396](https://arxiv.org/abs/2107.00396)
- Preferred download: **ftp://smartengines.com/midv-2020** (`tools/fetch_midv.py`)
- Follow `license.txt` in the archive (packaging text referenced as CC BY-SA 2.5;
  attribute Generated Photos / template sources as required by the dataset docs).

Stage-1 archives used by default: `scan_upright.tar`, `scan_rotated.tar`,
`photo.tar`, `templates.tar`.

## FMIDV (optional cross-domain forgeries)

- Portal: https://l3i-share.univ-lr.fr/2022FMIDV/FMIDV_v3.htm
- L3i-Share terms may require **permission for commercial activity**. Check the
  portal terms before production or commercial redistribution.
- Prefer author FTP for MIDV itself; use FMIDV only when you accept L3i terms.

## Forgery — backends

### Publishable path: ForgeryNet (Apache-oriented) — **default**

In-repo `services/forgery/model.py` + **torchvision ResNet-50 ImageNet** init
([torchvision](https://github.com/pytorch/vision) BSD-3 / redistributable weights) + our
FFT/HOG heads and upsample mask head trained on synthetic `gen_forgery` masks.

| Item | Notes |
|------|--------|
| Code license | Apache-2.0 (this repo) |
| Backbone init | torchvision `ResNet50_Weights.IMAGENET1K_V2` (BSD-3) |
| Fine-tune data | MIDV authentic (dataset terms) + synthetic gen_forgery (multi-seed/difficulty) |
| Default weights file | `models/forgery/forgerynet_apache.pth` (not in git) |
| Service | `forgery.backend: forgery_net`, `image_size: 320` (see `config.yaml`) |
| Harness | Cleared in-domain TC2/TC3 on n=1000 (joint feasible; see model card) |
| Hugging Face | **Allowed** — Hub model id `savagemanage/redrob-verify-forgery` (see `services/forgery/MODEL_CARD.md`); document ImageNet + MIDV + synthetic provenance |

### Research-only backend: TruFor (GRIP-UNINA, nonprofit-only)

Optional override: `FORGERY_BACKEND=trufor` / `forgery.backend: trufor` with a
checkpoint fine-tuned on our synthetic `gen_forgery` + MIDV authentic set.

| Item | Notes |
|------|--------|
| Upstream | [grip-unina/TruFor](https://github.com/grip-unina/TruFor) |
| License | GRIP-UNINA custom — **informational / nonprofit only**; industrial or profit-oriented use prohibited |
| Weights in git? | **No.** Pretrained `trufor.pth.tar` and fine-tuned `models/forgery/trufor_finetuned.pth.tar` stay local / private |
| Redistribution | Do **not** publish fine-tuned TruFor weights publicly (e.g. public Hugging Face). Cite GRIP-UNINA when reporting results that used TruFor |
| Status in this repo | Optional eval/research path that cleared TC2/TC3 on in-domain MIDV+gen_forgery. **Not** an Apache-2.0 production claim |

Code in this repository (FastAPI glue, harness, training scripts) remains Apache-2.0.
That does **not** re-license TruFor source or weights. Downstream users who enable
`backend: trufor` must obtain TruFor themselves and accept GRIP-UNINA terms.

### Do not treat as shippable product weights

| Artifact | Issue |
|----------|--------|
| TruFor pretrained / fine-tuned `.pth.tar` | Nonprofit-only upstream; keep out of public model hubs |
| Commercial “forgery API” outputs as training targets | Forbidden by project policy |

## Synthetic generators

- `tools/gen_forgery.py`, `tools/gen_indian_docs.py`, `tools/gen_resumes.py` produce
  synthetic or public-API-anchored samples. Do **not** commit real government-ID
  scans from Kaggle or similar dumps.

## Keeping this file honest

When adding a model or dataset:

1. Record license + URL here **before** wiring it into services.
2. Prefer Apache-2.0 / MIT / BSD / CC with clear commercial terms.
3. Never commit weight files or raw PII images.
