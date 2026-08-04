---
language:
  - en
license: apache-2.0
library_name: opencv
tags:
  - face-detection
  - face-recognition
  - onnx
  - redrob
  - redrob-verify
  - yunet
  - sface
pipeline_tag: image-feature-extraction
---

# redrob-verify — face

Face detection + recognition ONNX weights used by
**[redrob-verify](https://github.com/savagemanage/redrob-verify)** (Redrob verification stack).

- Hub: [`savagemanage/redrob-verify-face`](https://huggingface.co/savagemanage/redrob-verify-face)
- Upstream: [OpenCV Zoo](https://github.com/opencv/opencv_zoo) (Apache-2.0)
- Code: `services/face/` in the GitHub repo

## Files

| File | Model | Role |
|------|--------|------|
| `face_detection_yunet_2023mar.onnx` | YuNet | Face detection |
| `face_recognition_sface_2021dec.onnx` | SFace | Face embedding / match |

These are the same Apache-2.0 Zoo artifacts redrob-verify fetches by default.
Republished here so a single Hugging Face download path can bootstrap the stack
without depending on GitHub `raw` URLs.

## Intended use

Detect faces and compare embeddings for the redrob-verify face service
(`face.backend: sface`). Typical use: selfie vs ID-document face for KYC-style flows.

## Download

```bash
# From the redrob-verify checkout (preferred)
./tools/fetch_models.sh

# Or Hub only
huggingface-cli download savagemanage/redrob-verify-face \
  --local-dir models/face
```

Place files under `models/face/` as named above (see `config.yaml` `face.sface.*`).

## License & attribution

Apache-2.0. Copyright OpenCV / OpenCV Zoo authors. Redrob redistributes unchanged
ONNX weights for convenient Hub access; see upstream Zoo notices.

## Citation

If you use these models, cite OpenCV Zoo (YuNet, SFace) and redrob-verify as appropriate.
