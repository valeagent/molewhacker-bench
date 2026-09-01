#!/usr/bin/env bash
# =============================================================================
# run_all.sh — POSIX shell wrapper around 04_run_all.jl
# =============================================================================
# Usage (from the MoleWhacker repository root):
#
#     bash experiments/scripts/run_all.sh                 # full headline run
#     bash experiments/scripts/run_all.sh --scaling       # + scaling subset
#     bash experiments/scripts/run_all.sh --dryrun        # plan only
#
# Env overrides:
#     JULIA_BIN   override the julia executable (default: julia)
#     JULIA_THREADS  override -t (default: auto)
#     JULIA_PROJECT  override --project (default: . — i.e. parent project)
# =============================================================================
set -euo pipefail

JULIA_BIN="${JULIA_BIN:-julia}"
JULIA_THREADS="${JULIA_THREADS:-auto}"
JULIA_PROJECT_LOCAL="${JULIA_PROJECT:-.}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT="$ROOT_DIR/experiments/scripts/04_run_all.jl"

cd "$ROOT_DIR"

echo "[run_all.sh] julia=$JULIA_BIN  threads=$JULIA_THREADS  project=$JULIA_PROJECT_LOCAL"
echo "[run_all.sh] script=$SCRIPT"
echo "[run_all.sh] args: $*"

exec "$JULIA_BIN" --project="$JULIA_PROJECT_LOCAL" -t "$JULIA_THREADS" "$SCRIPT" "$@"
