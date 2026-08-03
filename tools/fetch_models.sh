#!/usr/bin/env bash
# Download OpenCV Zoo YuNet + SFace ONNX weights (Apache-2.0). Do not commit the files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/models/face"
mkdir -p "$OUT"

YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
YUNET="$OUT/face_detection_yunet_2023mar.onnx"
SFACE="$OUT/face_recognition_sface_2021dec.onnx"

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

download() {
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

download "$YUNET_URL" "$YUNET"
download "$SFACE_URL" "$SFACE"

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

YUNET_SHA="$(sha256_file "$YUNET")"
SFACE_SHA="$(sha256_file "$SFACE")"
echo "YuNet  SHA-256: $YUNET_SHA"
echo "SFace  SHA-256: $SFACE_SHA"

LIC="$ROOT/LICENSES.md"
if [[ -f "$LIC" ]]; then
  if ! PY="$(resolve_python)"; then
    echo "warning: no python — skip LICENSES.md SHA update (models already downloaded)"
  else
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

echo "done → $OUT"
