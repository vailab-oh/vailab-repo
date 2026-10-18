#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PACKAGE_ROOT"
CONFIG_PATH="$1"
shift
export CUDA_VISIBLE_DEVICES="${GPU_ID:-0}"
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'CUDA_VISIBLE_DEVICES=%s %s train.py --config %s\n' "$CUDA_VISIBLE_DEVICES" "${PYTHON_BIN:-python}" "$CONFIG_PATH"
  exit 0
fi
exec "${PYTHON_BIN:-python}" -u train.py --config "$CONFIG_PATH" "$@"
