#!/usr/bin/env bash
# Convenience launcher without compose.  Usage:
#   docker/run.sh [DATA_DIR] [-- command ...]
# Example:
#   docker/run.sh /mnt/h/UAV_Data -- python tools/verify_install.py --variant small
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${1:-$ROOT/data}"; shift || true
[ "${1:-}" = "--" ] && shift
mkdir -p "$ROOT/work_dirs"
exec docker run --rm -it --gpus all --ipc=host \
  -e HF_HOME=/workspace/hf_cache -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-0}" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -v "$ROOT":/workspace/U-MV \
  -v "$DATA_DIR":/data \
  -v umv_hf_cache:/workspace/hf_cache \
  -v "$ROOT/work_dirs":/workspace/U-MV/work_dirs \
  -w /workspace/U-MV umv:latest "${@:-/bin/bash}"
