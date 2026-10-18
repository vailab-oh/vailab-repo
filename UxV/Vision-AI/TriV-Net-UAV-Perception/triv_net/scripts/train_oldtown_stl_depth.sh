#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/_train.sh" configs/original_trivnet/oldtown_original_trivnet_depth.yaml "$@"
