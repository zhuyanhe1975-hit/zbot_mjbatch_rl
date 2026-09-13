#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CHECKPOINT="${CHECKPOINT:-runs/zbot_walk.pt}"
if [[ $# -gt 0 && "$1" != -* ]]; then
  CHECKPOINT="$1"
  shift
fi

if [[ ! -f "$CHECKPOINT" ]]; then
  echo "Checkpoint not found: $CHECKPOINT" >&2
  exit 1
fi

exec uv run python scripts/play.py "$CHECKPOINT" "$@"
