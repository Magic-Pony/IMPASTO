#!/usr/bin/env bash
set -euo pipefail
IMPASTO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONNOUSERSITE=1
unset PYTHONPATH
export PYTHONPATH="$IMPASTO_ROOT/src"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export MPLCONFIGDIR="$IMPASTO_ROOT/.cache/matplotlib"
export XDG_CACHE_HOME="$IMPASTO_ROOT/.cache"
cd "$IMPASTO_ROOT"
exec "$IMPASTO_ROOT/.conda/bin/python" "$@"
