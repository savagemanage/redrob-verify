---
language:
  - en
license: apache-2.0
library_name: pytorch
tags:
  - image-classification
  - forgery-detection
  - document-forgery
  - resnet
  - redrob
  - redrob-verify
pipeline_tag: image-classification
---

# redrob-verify — forgery

Forgery detector weights for **[redrob-verify](https://github.com/savagemanage/redrob-verify)**,
the Redrob verification stack (document OCR, forgery, face match, identity).

- Hub: [`savagemanage/redrob-verify-forgery`](https://huggingface.co/savagemanage/redrob-verify-forgery)
- Code: Apache-2.0 in the GitHub repo (`services/forgery/`)

## Intended use

Score a document scan or photo for **tampering likelihood** in `[0, 1]`
(higher = more likely forged). Tuned for ID / KYC–style pages in the
redrob-verify harness (MIDV authentic + synthetic patches).

Not a face deepfake detector. Not a claim of FMIDV cross-domain pass unless you
run that split yourself.

## Model

| | |
|--|--|
| Architecture | ResNet-50 + FFT + HOG streams + upsample localization head |
| Backbone init | torchvision `ResNet50_Weights.IMAGENET1K_V2` (BSD-3) |
| Input | RGB **320×320** |
| Serving score | Image-level sigmoid (localization used at train time) |
| Code | `services/forgery/model.py` |

Files in this repo:

- `model.safetensors` — weights for Hub / `safetensors` loaders
- `forgerynet_apache.pth` — full training checkpoint (`model_state` + metadata); drop-in for redrob-verify `config.yaml`
- `config.json` — image size, recommended threshold, provenance pointers

## Training data (provenance)

| Source | Role | Terms |
|--------|------|--------|
| torchvision ResNet-50 ImageNet-1K V2 | Backbone init | BSD-3 / torchvision |
| MIDV-2020 authentic pages | Train negatives (JPEG-recompressed) + eval authentic | Follow MIDV / portal terms |
| `tools/gen_forgery.py` synthetic tampers | Train positives + masks | Synthetic; generated in-repo |

Weights are **not** derived from TruFor.

## Evaluation (in-domain)

On redrob-verify `data/2_forgery` (n=1000):

- Joint TC2/TC3 feasible ≈ **[0.13, 0.89]**
- Recommended threshold **0.45** → TPR ≈ **0.92**, F1 ≈ **0.82**

Protocol: `./run.sh eval-forgery` in the GitHub repo.

## Download & run

```bash
# From the redrob-verify checkout
./tools/fetch_models.sh   # pulls face + forgery from Hugging Face

# Or Hub only
huggingface-cli download savagemanage/redrob-verify-forgery \
  --local-dir models/forgery
```

Serve with `forgery.backend: forgery_net`, `image_size: 320`, and
`weights_path: models/forgery/forgerynet_apache.pth` (or load `model.safetensors`
via the same `ForgeryNet` class).

## Limitations

- Domain: MIDV + our synthetic generator; other scanners/tampers may need fine-tuning.
- Optional TruFor backend in the code repo is research-only (nonprofit upstream) and is **not** these weights.

## Citation

Cite redrob-verify and MIDV-2020 per their terms when reporting results.
