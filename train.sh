#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

NUM_ENVS="${NUM_ENVS:-512}"
ITERATIONS="${ITERATIONS:-1000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"
CHECKPOINT="${CHECKPOINT:-runs/zbot_walk.pt}"
WATCH=false
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --watch)
      WATCH=true
      ;;
    --output)
      if [[ $# -lt 2 ]]; then
        echo "--output requires a checkpoint path" >&2
        exit 2
      fi
      CHECKPOINT="$2"
      shift
      ;;
    --output=*)
      CHECKPOINT="${1#*=}"
      ;;
    *)
      TRAIN_ARGS+=("$1")
      ;;
  esac
  shift
done

WATCH_PID=""
cleanup() {
  if [[ -n "$WATCH_PID" ]] && kill -0 "$WATCH_PID" 2>/dev/null; then
    kill "$WATCH_PID" 2>/dev/null || true
    wait "$WATCH_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "$WATCH" == true ]]; then
  echo "Starting checkpoint viewer for $CHECKPOINT"
  CHECKPOINT="$CHECKPOINT" "$PROJECT_DIR/watch.sh" &
  WATCH_PID=$!
fi

uv run python scripts/train.py \
  --num-envs "$NUM_ENVS" \
  --iterations "$ITERATIONS" \
  --save-interval "$SAVE_INTERVAL" \
  --output "$CHECKPOINT" \
  ${TRAIN_ARGS[@]+"${TRAIN_ARGS[@]}"}
