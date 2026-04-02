#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

MODE="${1:-base}"
VARIANT="${VARIANT:-g1}"
GLOBAL_CONTEXT="${GLOBAL_CONTEXT:-off}"

to_tyro_bool() {
  local value="${1,,}"
  case "$value" in
    true) printf 'True\n' ;;
    false) printf 'False\n' ;;
    *)
      echo "Expected boolean true/false, got: $1" >&2
      exit 2
      ;;
  esac
}

normalize_gpu_args() {
  local raw="${GPU_ARGS:-"--gpu-ids [0]"}"
  local -a parts
  read -r -a parts <<<"$raw"

  if [[ "${parts[0]:-}" != "--gpu-ids" ]]; then
    echo "GPU_ARGS must start with '--gpu-ids', got: $raw" >&2
    exit 2
  fi

  if [[ "${#parts[@]}" -eq 2 ]]; then
    case "${parts[1]}" in
      None|all|\[*\])
        printf '%s\n' "${parts[@]}"
        return
        ;;
      *)
        if [[ "${parts[1]}" =~ ^[0-9]+$ ]]; then
          printf '%s\n' "--gpu-ids" "[${parts[1]}]"
          return
        fi
        ;;
    esac
  fi

  if [[ "${#parts[@]}" -gt 2 ]]; then
    local idx
    local joined=""
    for idx in "${parts[@]:1}"; do
      if [[ ! "$idx" =~ ^[0-9]+$ ]]; then
        echo "Unsupported GPU_ARGS value: $raw" >&2
        exit 2
      fi
      if [[ -n "$joined" ]]; then
        joined+=","
      fi
      joined+="$idx"
    done
    printf '%s\n' "--gpu-ids" "[${joined}]"
    return
  fi

  echo "Unsupported GPU_ARGS value: $raw" >&2
  exit 2
}

mapfile -t GPU_ARGS < <(normalize_gpu_args)

BASE_ENVS="${BASE_ENVS:-4096}"
FINETUNE_ENVS="${FINETUNE_ENVS:-4096}"
BASE_ITERS="${BASE_ITERS:-15000}"
FINETUNE_ITERS="${FINETUNE_ITERS:-3200}"
BASE_RUN_NAME="${BASE_RUN_NAME:-${VARIANT}_base}"
FINETUNE_RUN_NAME="${FINETUNE_RUN_NAME:-${VARIANT}_finetune}"
LOGGER="${LOGGER:-tensorboard}"
WANDB_PROJECT="${WANDB_PROJECT:-ame_g1}"
WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_USERNAME:-}}"
UPLOAD_MODEL="$(to_tyro_bool "${UPLOAD_MODEL:-true}")"
VIDEO="$(to_tyro_bool "${VIDEO:-false}")"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
NAN_GUARD="$(to_tyro_bool "${NAN_GUARD:-false}")"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-}"

if [[ "$LOGGER" == "wandb" ]]; then
  export WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"
  export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$ROOT_DIR/.cache/wandb}"
  export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-$ROOT_DIR/.config/wandb}"
  export WANDB_DATA_DIR="${WANDB_DATA_DIR:-$ROOT_DIR/.local/share/wandb}"
  mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$WANDB_CONFIG_DIR" "$WANDB_DATA_DIR"

  if [[ -n "$WANDB_ENTITY" ]]; then
    export WANDB_USERNAME="$WANDB_ENTITY"
  fi
fi

find_latest_base_checkpoint() {
  find "$ROOT_DIR/logs/rsl_rl/ame_${VARIANT}_v3_base" -type f -name 'model_*.pt' 2>/dev/null \
    | sort \
    | tail -n 1
}

build_common_args() {
  local run_name="$1"
  local num_envs="$2"
  local max_iterations="$3"
  local -a args=(
    --env.scene.num-envs "$num_envs"
    --agent.run-name "$run_name"
    --agent.max-iterations "$max_iterations"
    --agent.logger "$LOGGER"
    --agent.wandb-project "$WANDB_PROJECT"
    --agent.upload-model "$UPLOAD_MODEL"
    --video "$VIDEO"
    --video-length "$VIDEO_LENGTH"
    --video-interval "$VIDEO_INTERVAL"
    --enable-nan-guard "$NAN_GUARD"
  )
  printf '%s\n' "${args[@]}"
}

run_base() {
  mapfile -t COMMON_ARGS < <(build_common_args "$BASE_RUN_NAME" "$BASE_ENVS" "$BASE_ITERS")
  uv run ame-train \
    --variant "$VARIANT" \
    --phase base \
    --global-context "$GLOBAL_CONTEXT" \
    "${GPU_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
}

run_finetune() {
  local checkpoint="$BASE_CHECKPOINT"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find_latest_base_checkpoint)"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "No base checkpoint found. Set BASE_CHECKPOINT=/abs/path/to/model.pt first." >&2
    exit 1
  fi
  mapfile -t COMMON_ARGS < <(build_common_args "$FINETUNE_RUN_NAME" "$FINETUNE_ENVS" "$FINETUNE_ITERS")
  uv run ame-train \
    --variant "$VARIANT" \
    --phase finetune \
    --global-context "$GLOBAL_CONTEXT" \
    --checkpoint-file "$checkpoint" \
    "${GPU_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
}

run_smoke() {
  uv run ame-train \
    --variant "$VARIANT" \
    --phase base \
    --global-context "$GLOBAL_CONTEXT" \
    --gpu-ids None \
    --env.scene.num-envs 64 \
    --agent.run-name smoke_cpu \
    --agent.max-iterations 3 \
    --agent.logger "$LOGGER" \
    --agent.wandb-project "$WANDB_PROJECT" \
    --agent.upload-model "$UPLOAD_MODEL" \
    --video False \
    --enable-nan-guard True
}

case "$MODE" in
  smoke)
    run_smoke
    ;;
  base)
    run_base
    ;;
  finetune)
    run_finetune
    ;;
  full)
    run_base
    run_finetune
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  scripts/run_train.sh smoke
  scripts/run_train.sh base
  scripts/run_train.sh finetune
  scripts/run_train.sh full

Optional environment variables:
  VARIANT=g1
  VARIANT=paper
  GLOBAL_CONTEXT=off
  GLOBAL_CONTEXT=on
  GPU_ARGS="--gpu-ids [0]"
  GPU_ARGS="--gpu-ids [0,1]"
  BASE_ENVS=4096
  FINETUNE_ENVS=4096
  BASE_ITERS=15000
  FINETUNE_ITERS=3200
  BASE_RUN_NAME=my_base
  FINETUNE_RUN_NAME=my_finetune
  BASE_CHECKPOINT=/abs/path/to/model.pt
  LOGGER=tensorboard
  LOGGER=wandb
  WANDB_PROJECT=ame_g1
  WANDB_USERNAME=my_team_or_user
  WANDB_ENTITY=my_team_or_user
  WANDB_MODE=offline
  WANDB_TAGS=ame,g1,base
  UPLOAD_MODEL=false
  VIDEO=true
  VIDEO_LENGTH=200
  VIDEO_INTERVAL=2000
  NAN_GUARD=true
EOF
    exit 2
    ;;
esac
