#!/usr/bin/env bash
# Single entrypoint for TTA / local evaluation.
# Usage: ./run.sh <command>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

export PYTHONIOENCODING=utf-8
export PYTHONUTF8=1
export PATH="${HOME}/.local/bin:/c/Python313/Scripts:${PATH:-}"

UV="${UV:-uv}"
CONFIG="${CONFIG:-config.yaml}"
COMPOSE="${COMPOSE:-docker compose}"

die() { echo "ERROR: $*" >&2; exit 1; }

need_uv() {
  command -v "$UV" >/dev/null 2>&1 || die "uv not found. Install: https://docs.astral.sh/uv/  (or: pip install uv)"
}

need_docker() {
  command -v docker >/dev/null 2>&1 || die "docker not found"
  docker compose version >/dev/null 2>&1 || die "docker compose not available"
}

# Print which process holds a host TCP port. Does not kill.
report_port_holder() {
  local port="$1"
  echo "Port $port is already in use. Holder:" >&2
  if command -v netstat >/dev/null 2>&1; then
    netstat -ano 2>/dev/null | grep -E "[:.]${port} .*LISTEN" >&2 || true
  fi
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null >&2 || true
  fi
}

# Return 0 if nothing is listening on TCP $1 on localhost.
port_is_free() {
  local port="$1"
  # Connect probe — reliable on Windows Git Bash where ss/lsof often miss listeners.
  local py=""
  if command -v "$UV" >/dev/null 2>&1; then
    py="$UV run python"
  elif command -v python >/dev/null 2>&1; then
    py="python"
  else
    return 0
  fi
  # shellcheck disable=SC2086
  if $py -c "
import socket
s = socket.socket()
s.settimeout(0.4)
try:
    r = s.connect_ex(('127.0.0.1', int('$port')))
finally:
    s.close()
raise SystemExit(0 if r != 0 else 1)
" 2>/dev/null; then
    return 0
  fi
  return 1
}

check_host_ports() {
  local ports=(8000 8001 8002 8003 8004)
  local busy=0
  local p
  for p in "${ports[@]}"; do
    if ! port_is_free "$p"; then
      report_port_holder "$p"
      busy=1
    fi
  done
  [[ $busy -eq 0 ]] || die "host port conflict — free the ports above, then retry (processes were NOT killed)"
}

run_preflight() {
  need_uv
  "$UV" run python -m harness.preflight --config "$CONFIG"
}

find_jmeter() {
  if [[ -n "${JMETER_HOME:-}" && -f "${JMETER_HOME}/bin/ApacheJMeter.jar" ]]; then
    echo "${JMETER_HOME}/bin/jmeter"
    return 0
  fi
  if [[ -n "${JMETER:-}" ]]; then
    echo "$JMETER"
    return 0
  fi
  if command -v jmeter >/dev/null 2>&1; then
    command -v jmeter
    return 0
  fi
  local candidates=(
    "$ROOT/tools/apache-jmeter/bin/jmeter"
    "$ROOT/tools/apache-jmeter/bin/jmeter.bat"
    "/c/apache-jmeter/bin/jmeter.bat"
    "/c/Program Files/apache-jmeter/bin/jmeter.bat"
  )
  local d
  for d in /c/apache-jmeter-* "$ROOT"/tools/apache-jmeter-*; do
    [[ -e "$d/bin/jmeter.bat" ]] && candidates+=("$d/bin/jmeter.bat")
    [[ -e "$d/bin/jmeter" ]] && candidates+=("$d/bin/jmeter")
    [[ -e "$d/bin/ApacheJMeter.jar" ]] && candidates+=("$d/bin/jmeter")
  done
  local c
  for c in "${candidates[@]}"; do
    local home
    home="$(dirname "$(dirname "$c")")"
    if [[ -f "$home/bin/ApacheJMeter.jar" ]]; then
      echo "$c"
      return 0
    fi
  done
  return 1
}

find_jmeter_home() {
  local bin
  bin="$(find_jmeter)" || return 1
  dirname "$(dirname "$bin")"
}

need_jmeter() {
  if ! JMETER_HOME="$(find_jmeter_home)"; then
    cat >&2 <<'EOF'
ERROR: Apache JMeter not found.

TC6 must be measured with JMeter (합의서). Python timing substitutes are not allowed.

Install (pick one):
  1) Download Apache JMeter 5.6+ from https://jmeter.apache.org/download_jmeter.cgi
     Unzip to tools/apache-jmeter/  (so tools/apache-jmeter/bin/ApacheJMeter.jar exists)
  2) Or: choco install jmeter
  3) Or set JMETER_HOME=/path/to/apache-jmeter

Requires Java 11+ (java -version).
Then re-run: ./run.sh eval-tc6
EOF
    exit 1
  fi
  [[ -f "$JMETER_HOME/bin/ApacheJMeter.jar" ]] || die "ApacheJMeter.jar missing under $JMETER_HOME/bin"
  export JMETER_HOME
  command -v java >/dev/null 2>&1 || die "java not found (JMeter requires Java 11+)"
}

cmd="${1:-}"
shift || true

case "$cmd" in
  setup)
    need_uv
    "$UV" sync
    ;;
  up)
    need_docker
    check_host_ports
    if [[ ! -f "$ROOT/models/face/face_recognition_sface_2021dec.onnx" ]]; then
      echo "Fetching OpenCV Zoo face models..."
      bash "$ROOT/tools/fetch_models.sh"
    fi
    $COMPOSE -f "$ROOT/docker-compose.yml" up -d --build
    echo "Waiting for /v1/meta on module ports..."
    # OCR first boot downloads PaddleOCR weights; allow generous warm-up.
    for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
      if "$UV" run python -m harness.preflight --config "$CONFIG"; then
        exit 0
      fi
      echo "preflight not ready yet (attempt $i/12); sleeping 10s..."
      sleep 10
    done
    die "containers up but preflight failed after warm-up"
    ;;
  down)
    need_docker
    $COMPOSE -f "$ROOT/docker-compose.yml" down
    ;;
  logs)
    need_docker
    $COMPOSE -f "$ROOT/docker-compose.yml" logs -f --tail=200 "$@"
    ;;
  ps)
    need_docker
    $COMPOSE -f "$ROOT/docker-compose.yml" ps "$@"
    ;;
  fetch-models)
    bash "$ROOT/tools/fetch_models.sh"
    ;;
  fetch-midv)
    need_uv
    "$UV" run python -m tools.fetch_midv "$@"
    ;;
  ingest-midv)
    need_uv
    "$UV" run python -m tools.ingest_midv "$@"
    ;;
  preflight)
    run_preflight
    ;;
  smoke-ocr)
    need_uv
    run_preflight
    "$UV" run python -m tools.smoke_ocr --config "$CONFIG" "$@"
    ;;
  bootstrap-gpu)
    # Fresh GPU host: deps → models → MIDV data → docker up → OCR smoke.
    # Skip steps: SKIP_MIDV=1 SKIP_UP=1 SKIP_SMOKE=1
    need_uv
    need_docker
    if [[ ! -f "$ROOT/.env" && -f "$ROOT/.env.example" ]]; then
      cp "$ROOT/.env.example" "$ROOT/.env"
      echo "created .env from .env.example (edit tokens if needed)"
    fi
    echo "==> setup (uv sync)"
    "$UV" sync
    echo "==> fetch face models"
    bash "$ROOT/tools/fetch_models.sh"
    if [[ "${SKIP_MIDV:-0}" != "1" ]]; then
      if [[ -f "$ROOT/data/1_ocr/manifest.jsonl" ]] && [[ -d "$ROOT/data/1_ocr/img" ]] \
        && "$UV" run python -c "from pathlib import Path; import sys; sys.exit(0 if any(Path('data/1_ocr/img').glob('*')) else 1)"; then
        echo "==> MIDV data already present under data/1_ocr — skip fetch/ingest"
      else
        echo "==> fetch MIDV-2020 (FTP smartengines.com)"
        "$UV" run python -m tools.fetch_midv
        echo "==> ingest MIDV → data/{1_ocr,2_forgery,3_face}"
        "$UV" run python -m tools.ingest_midv
      fi
    else
      echo "==> SKIP_MIDV=1"
    fi
    if [[ "${SKIP_UP:-0}" != "1" ]]; then
      echo "==> docker compose up (GPU)"
      check_host_ports
      $COMPOSE -f "$ROOT/docker-compose.yml" up -d --build
      echo "==> wait for preflight (OCR warm-up can take several minutes)"
      ready=0
      for i in $(seq 1 36); do
        if "$UV" run python -m harness.preflight --config "$CONFIG"; then
          ready=1
          break
        fi
        echo "preflight not ready (attempt $i/36); sleeping 10s..."
        sleep 10
      done
      [[ $ready -eq 1 ]] || die "containers up but preflight failed after warm-up"
      meta_backend="$("$UV" run python - <<'PY'
import json, urllib.request
print(json.load(urllib.request.urlopen("http://127.0.0.1:8001/v1/meta", timeout=10)).get("backend"))
PY
)"
      echo "OCR /v1/meta backend=$meta_backend"
      [[ "$meta_backend" == "paddleocr_vl" || "$meta_backend" == "paddleocr_classic" ]] \
        || die "unexpected OCR backend='$meta_backend' (refuse legacy mock)"
    else
      echo "==> SKIP_UP=1"
    fi
    if [[ "${SKIP_SMOKE:-0}" != "1" ]]; then
      echo "==> OCR smoke (one MIDV image, fail if hang/empty)"
      "$UV" run python -m tools.smoke_ocr --config "$CONFIG" --timeout-seconds 180
    else
      echo "==> SKIP_SMOKE=1"
    fi
    cat <<'EOF'
bootstrap-gpu OK.

Next:
  ./run.sh eval-cer          # TC1 full set (ocr.eval_max_samples must be null)
  ./run.sh eval-forgery
  ./run.sh eval-face
  ./run.sh logs ocr          # follow OCR container
EOF
    ;;
  gen-data)
    need_uv
    "$UV" run python tools/gen_mock_manifest.py --config "$CONFIG"
    ;;
  gen-fixtures)
    need_uv
    "$UV" run python tools/gen_dev_fixtures.py --config "$CONFIG" "$@"
    ;;
  mock)
    need_uv
    exec "$UV" run uvicorn harness.mock_server:app --host 127.0.0.1 --port 8000
    ;;
  ocr)
    need_uv
    OCR_HOST="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('ocr') or {}).get('host','127.0.0.1'))")"
    OCR_PORT="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('ocr') or {}).get('port',8001))")"
    exec "$UV" run uvicorn services.ocr.app:app --host "$OCR_HOST" --port "$OCR_PORT"
    ;;
  ablate-ocr)
    need_uv
    "$UV" run python tools/ablate_ocr.py --config "$CONFIG" "$@"
    ;;
  face)
    need_uv
    FACE_HOST="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('face') or {}).get('host','127.0.0.1'))")"
    FACE_PORT="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('face') or {}).get('port',8002))")"
    exec "$UV" run uvicorn services.face.app:app --host "$FACE_HOST" --port "$FACE_PORT"
    ;;
  forgery)
    need_uv
    FORGERY_HOST="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('forgery') or {}).get('host','127.0.0.1'))")"
    FORGERY_PORT="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('forgery') or {}).get('port',8003))")"
    exec "$UV" run uvicorn services.forgery.app:app --host "$FORGERY_HOST" --port "$FORGERY_PORT"
    ;;
  identity)
    need_uv
    IDENTITY_HOST="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('identity') or {}).get('host','127.0.0.1'))")"
    IDENTITY_PORT="$("$UV" run python -c "import yaml; print((yaml.safe_load(open('$CONFIG',encoding='utf-8')).get('identity') or {}).get('port',8004))")"
    exec "$UV" run uvicorn services.identity.app:app --host "$IDENTITY_HOST" --port "$IDENTITY_PORT"
    ;;
  gen-forgery)
    need_uv
    "$UV" run python tools/gen_forgery.py "$@"
    ;;
  gen-indian-docs)
    need_uv
    "$UV" run python tools/gen_indian_docs.py "$@"
    ;;
  gen-resumes)
    need_uv
    "$UV" run python tools/gen_resumes.py "$@"
    ;;
  build-midv-face-pairs)
    need_uv
    "$UV" run python tools/build_midv_face_pairs.py "$@"
    ;;
  build-face-pairs)
    need_uv
    "$UV" run python tools/build_face_pairs.py "$@"
    ;;
  build-ds2-splits)
    need_uv
    "$UV" run python tools/build_ds2_splits.py "$@"
    ;;
  train-forgery)
    need_uv
    "$UV" run python -m services.forgery.train --config "$CONFIG" "$@"
    ;;
  calibrate-face)
    need_uv
    "$UV" run python tools/calibrate_face.py --config "$CONFIG" "$@"
    ;;
  test-face)
    need_uv
    "$UV" run pytest tests/test_face_backend_swap.py -q
    ;;
  test)
    need_uv
    "$UV" run pytest tests/ -q
    ;;
  freeze)
    need_uv
    "$UV" run python -m harness.freeze --config "$CONFIG" "$@"
    ;;
  eval-cer)
    need_uv
    run_preflight
    "$UV" run python -m harness.eval_cer --config "$CONFIG"
    ;;
  eval-forgery)
    need_uv
    run_preflight
    "$UV" run python -m harness.eval_forgery --config "$CONFIG"
    ;;
  eval-face)
    need_uv
    run_preflight
    "$UV" run python -m harness.eval_face --config "$CONFIG"
    ;;
  eval-tc6)
    need_uv
    need_jmeter
    run_preflight
    mkdir -p results
    rm -f results/tc6.jtl results/jmeter.log results/tc6_identity.json
    IDENTITY_URL="$("$UV" run python -c "import yaml; e=yaml.safe_load(open('$CONFIG',encoding='utf-8'))['endpoints']['identity']; print(e.rstrip('/'))")"
    IDENTITY_HOST="${IDENTITY_HOST:-$("$UV" run python -c "from urllib.parse import urlparse; u=urlparse('$IDENTITY_URL'); print(u.hostname or '127.0.0.1')")}"
    IDENTITY_PORT="${IDENTITY_PORT:-$("$UV" run python -c "from urllib.parse import urlparse; u=urlparse('$IDENTITY_URL'); print(u.port or 80)")}"
    echo "Using JMeter home: $JMETER_HOME"
    echo "TC6 target: http://${IDENTITY_HOST}:${IDENTITY_PORT}"
    java -jar "$JMETER_HOME/bin/ApacheJMeter.jar" -n \
      -t harness/tc6.jmx \
      -l results/tc6.jtl \
      -j results/jmeter.log \
      -JRESUME_CSV=data/4_resume/jmeter_resumes.csv \
      -JIDENTITY_HOST="$IDENTITY_HOST" \
      -JIDENTITY_PORT="$IDENTITY_PORT"
    [[ -f results/tc6.jtl ]] || die "JMeter did not produce results/tc6.jtl"
    "$UV" run python -m harness.tc6_parse results/tc6.jtl -o results/tc6_identity.json --config "$CONFIG"
    ;;
  report)
    need_uv
    "$UV" run python -m harness.report --config "$CONFIG"
    ;;
  eval-all)
    need_uv
    need_jmeter
    run_preflight
    ec=0
    set +e
    "$UV" run python -m harness.eval_cer --config "$CONFIG"; r=$?; [[ $r -ne 0 ]] && ec=$r
    "$UV" run python -m harness.eval_forgery --config "$CONFIG"; r=$?; [[ $r -ne 0 ]] && ec=$r
    "$UV" run python -m harness.eval_face --config "$CONFIG"; r=$?; [[ $r -ne 0 ]] && ec=$r
    # TC6 still goes through jmeter path (preflight already done)
    mkdir -p results
    rm -f results/tc6.jtl results/jmeter.log results/tc6_identity.json
    IDENTITY_URL="$("$UV" run python -c "import yaml; e=yaml.safe_load(open('$CONFIG',encoding='utf-8'))['endpoints']['identity']; print(e.rstrip('/'))")"
    IDENTITY_HOST="${IDENTITY_HOST:-$("$UV" run python -c "from urllib.parse import urlparse; u=urlparse('$IDENTITY_URL'); print(u.hostname or '127.0.0.1')")}"
    IDENTITY_PORT="${IDENTITY_PORT:-$("$UV" run python -c "from urllib.parse import urlparse; u=urlparse('$IDENTITY_URL'); print(u.port or 80)")}"
    java -jar "$JMETER_HOME/bin/ApacheJMeter.jar" -n \
      -t harness/tc6.jmx \
      -l results/tc6.jtl \
      -j results/jmeter.log \
      -JRESUME_CSV=data/4_resume/jmeter_resumes.csv \
      -JIDENTITY_HOST="$IDENTITY_HOST" \
      -JIDENTITY_PORT="$IDENTITY_PORT"
    r=$?; [[ $r -ne 0 ]] && ec=$r
    if [[ -f results/tc6.jtl ]]; then
      "$UV" run python -m harness.tc6_parse results/tc6.jtl -o results/tc6_identity.json --config "$CONFIG"; r=$?; [[ $r -ne 0 ]] && ec=$r
    else
      ec=1
    fi
    "$0" report; r=$?; [[ $r -ne 0 ]] && ec=$r
    set -e
    exit "$ec"
    ;;
  ""|-h|--help|help)
    cat <<'EOF'
redrob-verify runner

  ./run.sh bootstrap-gpu   # GPU host one-shot: sync → models → MIDV → up → smoke-ocr
  ./run.sh setup           # uv sync
  ./run.sh up              # docker compose up (checks host ports; does not kill)
  ./run.sh down            # docker compose down
  ./run.sh logs            # docker compose logs -f
  ./run.sh ps              # docker compose ps
  ./run.sh fetch-models    # OpenCV Zoo YuNet + SFace ONNX
  ./run.sh fetch-midv      # MIDV-2020 FTP → results/midv_archives/
  ./run.sh ingest-midv     # archives → data/{1_ocr,2_forgery,3_face}
  ./run.sh preflight       # GET /v1/meta on all endpoints (exit 1 on mismatch)
  ./run.sh smoke-ocr       # one-image OCR probe (fail on empty/hang)
  ./run.sh gen-data        # fixtures + indian docs (+ resumes unless skipped)
  ./run.sh gen-fixtures    # plumbing → results/dev_fixtures
  ./run.sh gen-indian-docs # → results/indian_docs
  ./run.sh gen-resumes     # → data/4_resume
  ./run.sh build-midv-face-pairs
  ./run.sh build-ds2-splits
  ./run.sh build-face-pairs -- --input <identities_dir> [--origin field_collected]
  ./run.sh gen-forgery
  ./run.sh mock            # mock OCR/forgery/identity on :8000
  ./run.sh ocr|face|forgery|identity
  ./run.sh eval-all        # preflight + TC1–TC6 + report.html
  ./run.sh eval-cer | eval-forgery | eval-face | eval-tc6 | report | freeze
  ./run.sh test            # pytest (includes preflight honesty tests)

Environment:
  UV, CONFIG, JMETER_HOME, IDENTITY_HOST, IDENTITY_PORT, IDENTITY_SOURCE_DELAY_SEC
  SKIP_MIDV=1 SKIP_UP=1 SKIP_SMOKE=1   # bootstrap-gpu only
EOF
    ;;
  *)
    die "unknown command: $cmd (try: ./run.sh help)"
    ;;
esac
