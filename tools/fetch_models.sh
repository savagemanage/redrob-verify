#!/usr/bin/env bash
# Fetch redrob-verify runtime weights from Hugging Face (public).
# Face: YuNet + SFace ONNX (OpenCV Zoo, Apache-2.0) via redrob-labs/redrob-verify-face
# Forgery: ForgeryNet checkpoint via redrob-labs/redrob-verify-forgery
#
# Fallback: OpenCV Zoo GitHub raw URLs for face only if Hub is unreachable.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FACE_OUT="$ROOT/models/face"
FORGERY_OUT="$ROOT/models/forgery"
mkdir -p "$FACE_OUT" "$FORGERY_OUT"

HF_FACE_REPO="${HF_FACE_REPO:-redrob-labs/redrob-verify-face}"
HF_FORGERY_REPO="${HF_FORGERY_REPO:-redrob-labs/redrob-verify-forgery}"

YUNET_NAME="face_detection_yunet_2023mar.onnx"
SFACE_NAME="face_recognition_sface_2021dec.onnx"
FORGERY_PTH="forgerynet_apache.pth"
FORGERY_ST="model.safetensors"

YUNET_ZOO="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_ZOO="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

resolve_python() {
  if [[ -x "$ROOT/.venv/bin/python" ]]; then
    echo "$ROOT/.venv/bin/python"
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v uv >/dev/null 2>&1; then
    echo "uv run python"
    return 0
  fi
  return 1
}

download_http() {
  local url="$1" dest="$2"
  if [[ -f "$dest" ]]; then
    echo "exists: $dest"
    return 0
  fi
  echo "fetching $url"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$dest" "$url"
  else
    wget -O "$dest" "$url"
  fi
}

hf_download_file() {
  local repo="$1" filename="$2" dest="$3"
  if [[ -f "$dest" ]]; then
    echo "exists: $dest"
    return 0
  fi
  local url="https://huggingface.co/${repo}/resolve/main/${filename}"
  echo "fetching HF ${repo}/${filename}"
  if command -v curl >/dev/null 2>&1; then
    curl -L --fail --retry 3 -o "$dest" "$url"
  else
    wget -O "$dest" "$url"
  fi
}

echo "==> face (${HF_FACE_REPO})"
if ! hf_download_file "$HF_FACE_REPO" "$YUNET_NAME" "$FACE_OUT/$YUNET_NAME"; then
  echo "HF face YuNet failed — falling back to OpenCV Zoo"
  rm -f "$FACE_OUT/$YUNET_NAME"
  download_http "$YUNET_ZOO" "$FACE_OUT/$YUNET_NAME"
fi
if ! hf_download_file "$HF_FACE_REPO" "$SFACE_NAME" "$FACE_OUT/$SFACE_NAME"; then
  echo "HF face SFace failed — falling back to OpenCV Zoo"
  rm -f "$FACE_OUT/$SFACE_NAME"
  download_http "$SFACE_ZOO" "$FACE_OUT/$SFACE_NAME"
fi

echo "==> forgery (${HF_FORGERY_REPO})"
hf_download_file "$HF_FORGERY_REPO" "$FORGERY_PTH" "$FORGERY_OUT/$FORGERY_PTH" || true
hf_download_file "$HF_FORGERY_REPO" "$FORGERY_ST" "$FORGERY_OUT/$FORGERY_ST" || true
hf_download_file "$HF_FORGERY_REPO" "config.json" "$FORGERY_OUT/config.json" || true

if [[ ! -f "$FORGERY_OUT/$FORGERY_PTH" && ! -f "$FORGERY_OUT/$FORGERY_ST" ]]; then
  echo "WARNING: forgery weights not downloaded. Train locally or check Hub access:" >&2
  echo "  https://huggingface.co/${HF_FORGERY_REPO}" >&2
fi

sha256_file() {
  local f="$1"
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum "$f" | awk '{print $1}'
  elif command -v shasum >/dev/null 2>&1; then
    shasum -a 256 "$f" | awk '{print $1}'
  else
    local py
    py="$(resolve_python)" || {
      echo "ERROR: need sha256sum or python to hash $f" >&2
      return 1
    }
    # shellcheck disable=SC2086
    $py -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" "$f"
  fi
}

YUNET_SHA="$(sha256_file "$FACE_OUT/$YUNET_NAME")"
SFACE_SHA="$(sha256_file "$FACE_OUT/$SFACE_NAME")"
echo "YuNet  SHA-256: $YUNET_SHA"
echo "SFace  SHA-256: $SFACE_SHA"
[[ -f "$FORGERY_OUT/$FORGERY_PTH" ]] && echo "Forgery PTH SHA-256: $(sha256_file "$FORGERY_OUT/$FORGERY_PTH")"
[[ -f "$FORGERY_OUT/$FORGERY_ST" ]] && echo "Forgery ST  SHA-256: $(sha256_file "$FORGERY_OUT/$FORGERY_ST")"

LIC="$ROOT/LICENSES.md"
if [[ -f "$LIC" ]]; then
  if PY="$(resolve_python)"; then
    # shellcheck disable=SC2086
    $PY - <<PY
from pathlib import Path
import re
lic = Path(r"""$LIC""")
text = lic.read_text(encoding="utf-8")
yunet = "$YUNET_SHA"
sface = "$SFACE_SHA"
text2 = re.sub(
    r"(\| \`models/face/face_detection_yunet_2023mar\.onnx\` \| YuNet[^|]*\| )[^|\n]+",
    r"\g<1>" + yunet + " ",
    text,
    count=1,
)
text2 = re.sub(
    r"(\| \`models/face/face_recognition_sface_2021dec\.onnx\` \| SFace[^|]*\| )[^|\n]+",
    r"\g<1>" + sface + " ",
    text2,
    count=1,
)
if text2 != text:
    lic.write_text(text2, encoding="utf-8")
    print(f"updated SHA-256 rows in {lic}")
else:
    print("LICENSES.md SHA rows unchanged (already filled or pattern miss)")
PY
  fi
fi

echo "done → $FACE_OUT , $FORGERY_OUT"
