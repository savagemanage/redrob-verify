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
pipeline_tag: image-classification
---

# ForgeryNet (redrob-verify)

Document image forgery detector used by [redrob-verify](https://github.com/savagemanage/redrob-verify).

## Intended use

Score a document scan/photo for **tampering likelihood** in `[0, 1]` (higher = more likely forged).
Built for ID / KYC style documents in the redrob-verify harness (MIDV authentic + synthetic patches).

**Not** a general deepfake face detector. **Not** a claim of cross-domain FMIDV pass unless you evaluate that split yourself.

## Architecture

- ResNet-50 encoder (ImageNet-1K V2 init via torchvision)
- FFT magnitude + differentiable HOG auxiliary streams
- Image-level classifier + upsample localization head (mask BCE + Dice at train time)
- Default input size: **320×320** RGB

Code: `services/forgery/model.py` in the repo (Apache-2.0).

## Training data (provenance)

| Source | Role | Terms |
|--------|------|--------|
| torchvision ResNet50 ImageNet-1K V2 | Backbone init | BSD-3 / torchvision |
| MIDV-2020 authentic pages | Train negatives (JPEG-recompressed) + eval authentic | Follow MIDV / portal terms |
| `tools/gen_forgery.py` synthetic tampers | Train positives + masks (copy-move / splice / inpaint / text_replace; multi-seed & difficulty) | Synthetic; generated in-repo |

Weights are **not** derived from TruFor.

## Evaluation (in-domain harness)

On redrob-verify `data/2_forgery` (n=1000, 500 auth / 500 gen_forgery test-profile):

- Joint TC2/TC3 feasible interval ≈ **[0.13, 0.89]**
- Recommended threshold **0.45** → TPR ≈ **0.92**, F1 ≈ **0.82**
- Metrics and protocol: `./run.sh eval-forgery` (see repo `config.yaml` targets)

Cross-domain FMIDV is optional and off by default (`forgery.require_cross_domain: false`).

## How to use

```bash
# Train (GPU)
./run.sh train-forgery --epochs 50 --image-size 320 --loc-weight 2.0 --gpu-resident \
  --output models/forgery/forgerynet_apache.pth

# Serve
# config.yaml: forgery.backend=forgery_net, weights_path=models/forgery/forgerynet_apache.pth, image_size=320
# or: FORGERY_BACKEND=forgery_net
```

Checkpoint dict keys: `model_state`, plus train metadata (`epoch`, `auc`, `joint_interval_width`, …).

## Limitations

- Tuned to MIDV + our synthetic generator; other scanners / tampers may need fine-tuning.
- Localization head helps training; serving score is the **image-level** sigmoid.
- Do not confuse with the optional TruFor research backend (nonprofit-only upstream).

## Citation

If you use this checkpoint, cite redrob-verify and the MIDV-2020 dataset per their terms.
