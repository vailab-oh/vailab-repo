#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PACKAGE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$PACKAGE_ROOT"
for task in stl_seg stl_depth stl_od mtl; do
  echo "[run] OldTown original TriV-Net ${task}"
  bash "$SCRIPT_DIR/train_oldtown_${task}.sh" "$@"
done
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo '[dry-run] No training started.'
else
  echo '[done] All four original TriV-Net runs completed.'
fi
