#!/usr/bin/env bash
# Convenience launcher without compose.  Reads DATA_DIR / WEIGHTS_DIR from
# docker/.env (or the environment).  Usage:
#   docker/run.sh [-- command ...]
# Example:
#   docker/run.sh -- python tools/verify_install.py --variant small
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$ROOT/docker/.env" ] && set -a && . "$ROOT/docker/.env" && set +a
DATA_DIR="${DATA_DIR:-$ROOT/data}"
WEIGHTS_DIR="${WEIGHTS_DIR:-$ROOT/Pretrained_Weights}"
[ "${1:-}" = "--" ] && shift
mkdir -p "$ROOT/work_dirs"
exec docker run --rm -it --gpus all --ipc=host \
  -e HF_HOME=/workspace/hf_cache -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$ROOT":/workspace/U-MV \
  -v "$DATA_DIR":/data \
  -v "$WEIGHTS_DIR":/weights:ro \
  -v umv_hf_cache:/workspace/hf_cache \
  -v "$ROOT/work_dirs":/workspace/U-MV/work_dirs \
  -w /workspace/U-MV umv:latest "${@:-/bin/bash}"
