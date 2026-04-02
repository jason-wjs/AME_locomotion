#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

MODE="${1:-play}"
VARIANT="${VARIANT:-g1}"
PHASE="${PHASE:-finetune}"
GLOBAL_CONTEXT="${GLOBAL_CONTEXT:-off}"
NUM_ENVS="${NUM_ENVS:-1}"
NUM_STEPS="${NUM_STEPS:-400}"
NUM_EPISODES="${NUM_EPISODES:-64}"
DEVICE="${DEVICE:-}"
CHECKPOINT_FILE="${CHECKPOINT_FILE:-}"
VIEWER_BACKEND="${VIEWER_BACKEND:-auto}"
DUMP_FILE="${DUMP_FILE:-artifacts/attention_dump.pt}"
DUMP_EVERY="${DUMP_EVERY:-1}"

find_latest_checkpoint() {
  find "$ROOT_DIR/logs/rsl_rl/ame_${VARIANT}_v3_${PHASE}" -type f -name 'model_*.pt' 2>/dev/null \
    | sort \
    | tail -n 1
}

resolve_checkpoint() {
  local checkpoint="$CHECKPOINT_FILE"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find_latest_checkpoint)"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "No checkpoint found. Set CHECKPOINT_FILE=/abs/path/to/model.pt first." >&2
    exit 1
  fi
  if [[ ! -f "$checkpoint" ]]; then
    echo "Checkpoint not found: $checkpoint" >&2
    exit 1
  fi
  printf '%s\n' "$checkpoint"
}

build_device_args() {
  if [[ -n "$DEVICE" ]]; then
    printf '%s\n' --device "$DEVICE"
  fi
}

run_play() {
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  mapfile -t DEVICE_ARGS < <(build_device_args)
  uv run ame-play \
    --variant "$VARIANT" \
    --phase "$PHASE" \
    --global-context "$GLOBAL_CONTEXT" \
    --checkpoint-file "$checkpoint" \
    --num-envs "$NUM_ENVS" \
    --num-steps "$NUM_STEPS" \
    --viewer \
    --viewer-backend "$VIEWER_BACKEND" \
    "${DEVICE_ARGS[@]}"
}

run_headless_play() {
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  mapfile -t DEVICE_ARGS < <(build_device_args)
  uv run ame-play \
    --variant "$VARIANT" \
    --phase "$PHASE" \
    --global-context "$GLOBAL_CONTEXT" \
    --checkpoint-file "$checkpoint" \
    --num-envs "$NUM_ENVS" \
    --num-steps "$NUM_STEPS" \
    "${DEVICE_ARGS[@]}"
}

run_eval() {
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  mapfile -t DEVICE_ARGS < <(build_device_args)
  uv run ame-eval \
    --variant "$VARIANT" \
    --phase "$PHASE" \
    --global-context "$GLOBAL_CONTEXT" \
    --checkpoint-file "$checkpoint" \
    --num-envs "$NUM_ENVS" \
    --num-episodes "$NUM_EPISODES" \
    "${DEVICE_ARGS[@]}"
}

run_dump() {
  local checkpoint
  checkpoint="$(resolve_checkpoint)"
  mapfile -t DEVICE_ARGS < <(build_device_args)
  uv run ame-play \
    --variant "$VARIANT" \
    --phase "$PHASE" \
    --global-context "$GLOBAL_CONTEXT" \
    --checkpoint-file "$checkpoint" \
    --num-envs "$NUM_ENVS" \
    --num-steps "$NUM_STEPS" \
    --dump-file "$DUMP_FILE" \
    --dump-every "$DUMP_EVERY" \
    "${DEVICE_ARGS[@]}"
  uv run ame-plot-attention \
    --dump-file "$DUMP_FILE" \
    --output-dir artifacts/attention
}

case "$MODE" in
  play)
    run_play
    ;;
  headless)
    run_headless_play
    ;;
  eval)
    run_eval
    ;;
  dump)
    run_dump
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  scripts/run_play.sh play
  scripts/run_play.sh headless
  scripts/run_play.sh eval
  scripts/run_play.sh dump

Optional environment variables:
  VARIANT=g1
  VARIANT=paper
  PHASE=base
  PHASE=finetune
  GLOBAL_CONTEXT=off
  GLOBAL_CONTEXT=on
  CHECKPOINT_FILE=/abs/path/to/model.pt
  NUM_ENVS=1
  NUM_STEPS=400
  NUM_EPISODES=64
  DEVICE=cpu
  DEVICE=cuda:0
  VIEWER_BACKEND=auto
  VIEWER_BACKEND=native
  VIEWER_BACKEND=viser
  DUMP_FILE=artifacts/attention_dump.pt
  DUMP_EVERY=1
EOF
    exit 2
    ;;
esac
