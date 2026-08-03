#!/usr/bin/env bash
# GPU host entrypoint — thin wrapper around ./run.sh bootstrap-gpu
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/run.sh" bootstrap-gpu "$@"
