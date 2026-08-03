#!/usr/bin/env bash
# From your laptop: push current branch to origin, then SSH to the GPU host and pull.
#
# Setup (local only, never commit):
#   cp .gpu.env.example .gpu.env   # then fill GPU_SSH_HOST / GPU_SSH_KEY
#   # or export GPU_SSH_HOST=... GPU_SSH_KEY=... in your shell
#
# Usage:
#   ./run.sh sync-gpu
#   SYNC_REBUILD=1 ./run.sh sync-gpu
#   SYNC_REBUILD=1 SYNC_SERVICES=all ./run.sh sync-gpu
#
# Required: GPU_SSH_HOST, GPU_SSH_KEY
# Optional:
#   GPU_SSH_USER   ubuntu
#   GPU_REPO_DIR   redrob-verify   (relative to remote $HOME, or absolute /path)
#   SYNC_PUSH=1    push local branch before remote pull (set 0 to pull-only)
#   SYNC_REBUILD=0 after pull: docker compose build + up for SYNC_SERVICES
#   SYNC_SERVICES  ocr | all
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Local machine config — gitignored. Do not put host/key defaults in this script.
if [[ -f "$ROOT/.gpu.env" ]]; then
  # shellcheck disable=SC1091
  set -a
  # shellcheck source=/dev/null
  source "$ROOT/.gpu.env"
  set +a
fi

GPU_SSH_USER="${GPU_SSH_USER:-ubuntu}"
GPU_REPO_DIR="${GPU_REPO_DIR:-redrob-verify}"
SYNC_PUSH="${SYNC_PUSH:-1}"
SYNC_REBUILD="${SYNC_REBUILD:-0}"
SYNC_SERVICES="${SYNC_SERVICES:-ocr}"

die() { echo "ERROR: $*" >&2; exit 1; }

[[ -n "${GPU_SSH_HOST:-}" ]] || die "GPU_SSH_HOST unset — copy .gpu.env.example → .gpu.env (gitignored)"
[[ -n "${GPU_SSH_KEY:-}" ]] || die "GPU_SSH_KEY unset — copy .gpu.env.example → .gpu.env (gitignored)"

# Expand ~ in key path (Windows Git Bash / bash)
case "$GPU_SSH_KEY" in
  "~"/*) GPU_SSH_KEY="${HOME}/${GPU_SSH_KEY#~/}" ;;
esac

[[ -f "$GPU_SSH_KEY" ]] || die "SSH key not found: $GPU_SSH_KEY (set GPU_SSH_KEY)"

branch="$(git rev-parse --abbrev-ref HEAD)"
[[ "$branch" != "HEAD" ]] || die "detached HEAD — checkout a branch first"
head_sha="$(git rev-parse --short HEAD)"

if [[ -n "$(git status --porcelain)" ]]; then
  echo "WARNING: local working tree has uncommitted changes; only committed commits will sync."
  git status -sb
fi

if [[ "$SYNC_PUSH" == "1" ]]; then
  echo "==> local: git push origin $branch ($head_sha)"
  git push -u origin "HEAD:refs/heads/$branch"
else
  echo "==> skip local push (SYNC_PUSH=0)"
fi

remote_target="${GPU_SSH_USER}@${GPU_SSH_HOST}"
ssh_opts=(-i "$GPU_SSH_KEY" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new)

echo "==> remote (${GPU_SSH_USER}@***): git pull → ${GPU_REPO_DIR}"

# Pass args positionally so Git Bash printf %q cannot mangle paths.
# shellcheck disable=SC2029
ssh "${ssh_opts[@]}" "$remote_target" \
  bash -s -- "$GPU_REPO_DIR" "$branch" "$SYNC_REBUILD" "$SYNC_SERVICES" <<'REMOTE'
set -euo pipefail
GPU_REPO_DIR="$1"
BRANCH="$2"
SYNC_REBUILD="$3"
SYNC_SERVICES="$4"

# Strip accidental leading ~/ from older configs
case "$GPU_REPO_DIR" in
  "~/"*) GPU_REPO_DIR="${GPU_REPO_DIR#~/}" ;;
  "~") GPU_REPO_DIR="" ;;
esac
if [[ "$GPU_REPO_DIR" == /* ]]; then
  cd "$GPU_REPO_DIR"
else
  cd "$HOME/${GPU_REPO_DIR}"
fi
echo "cwd=$(pwd) before=$(git rev-parse --short HEAD) branch=$(git rev-parse --abbrev-ref HEAD)"

# chmod / line-ending dirt on scripts must not block pull
if [[ -n "$(git status --porcelain)" ]]; then
  echo "stashing local dirt on GPU before pull"
  git stash push -u -m "sync_gpu auto-stash $(date -u +%Y%m%dT%H%M%SZ)" || true
fi

git fetch origin
git checkout "$BRANCH"
git pull --ff-only origin "$BRANCH"
echo "after=$(git rev-parse --short HEAD)"

if [[ "${SYNC_REBUILD}" == "1" ]]; then
  if [[ "${SYNC_SERVICES}" == "all" ]]; then
    echo "==> rebuild all services"
    docker compose up -d --build
  else
    echo "==> rebuild service: ${SYNC_SERVICES}"
    # shellcheck disable=SC2086
    docker compose build ${SYNC_SERVICES}
    # shellcheck disable=SC2086
    docker compose up -d ${SYNC_SERVICES}
  fi
  echo "==> /v1/meta"
  for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
    meta="$(curl -sS --max-time 3 http://127.0.0.1:8001/v1/meta 2>/dev/null || true)"
    if echo "$meta" | grep -q '"service"'; then
      echo "$meta"
      break
    fi
    echo "wait_$i"
    sleep 5
  done
fi
REMOTE

echo "==> sync-gpu done ($branch)"
