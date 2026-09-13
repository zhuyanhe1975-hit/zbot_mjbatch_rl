#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

CHECKPOINT="${CHECKPOINT:-runs/zbot_walk.pt}"
POLL_SECONDS="${POLL_SECONDS:-2}"
PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
  echo "Virtual environment not found; run 'uv sync --dev' first." >&2
  exit 1
fi

echo "Watching $CHECKPOINT (poll every ${POLL_SECONDS}s). Press Ctrl-C to stop."

while [[ ! -f "$CHECKPOINT" ]]; do
  sleep "$POLL_SECONDS"
done

exec "$PYTHON" scripts/play.py "$CHECKPOINT" --watch --poll-seconds "$POLL_SECONDS"
