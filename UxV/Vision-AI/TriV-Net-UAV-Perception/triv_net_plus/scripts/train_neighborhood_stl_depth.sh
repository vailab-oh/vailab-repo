#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

exec "$PYTHON_BIN" train.py \
  --config configs/neighborhood/stl_depth.yaml \
  --run-name neighborhood_stl_depth
