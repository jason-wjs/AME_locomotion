#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib}"

MODE="${1:-stage1}"
VARIANT="${VARIANT:-g1}"

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

# GPU selection:
#   single GPU: GPU_ARGS="--gpu-ids [0]"
#   multi GPU : GPU_ARGS="--gpu-ids [0,1]"
#   CPU only  : GPU_ARGS="--gpu-ids None"
#
# Backward compatibility:
#   GPU_ARGS="--gpu-ids 0 1" is normalized to "--gpu-ids [0,1]".
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

# Shared overrides. Edit these rather than touching the command body below.
STAGE1_ENVS="${STAGE1_ENVS:-4096}"
STAGE2_ENVS="${STAGE2_ENVS:-4096}"
STAGE1_ITERS="${STAGE1_ITERS:-15000}"
STAGE2_ITERS="${STAGE2_ITERS:-3200}"
STAGE1_RUN_NAME="${STAGE1_RUN_NAME:-${VARIANT}_stage1}"
STAGE2_RUN_NAME="${STAGE2_RUN_NAME:-${VARIANT}_stage2}"
LOGGER="${LOGGER:-tensorboard}"
WANDB_PROJECT="${WANDB_PROJECT:-ame_g1}"
WANDB_ENTITY="${WANDB_ENTITY:-${WANDB_USERNAME:-}}"
UPLOAD_MODEL="$(to_tyro_bool "${UPLOAD_MODEL:-true}")"
VIDEO="$(to_tyro_bool "${VIDEO:-false}")"
VIDEO_LENGTH="${VIDEO_LENGTH:-200}"
VIDEO_INTERVAL="${VIDEO_INTERVAL:-2000}"
NAN_GUARD="$(to_tyro_bool "${NAN_GUARD:-false}")"

# Optional manual override for stage2.
STAGE1_CHECKPOINT="${STAGE1_CHECKPOINT:-}"

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

find_latest_stage1_checkpoint() {
  find "$ROOT_DIR/logs/rsl_rl/ame_${VARIANT}_v2_stage1" -type f -name 'model_*.pt' 2>/dev/null \
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

run_stage1() {
  mapfile -t COMMON_ARGS < <(build_common_args "$STAGE1_RUN_NAME" "$STAGE1_ENVS" "$STAGE1_ITERS")
  uv run ame-train \
    --variant "$VARIANT" \
    --stage stage1 \
    "${GPU_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
}

run_stage2() {
  local checkpoint="${STAGE1_CHECKPOINT}"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find_latest_stage1_checkpoint)"
  fi
  if [[ -z "$checkpoint" ]]; then
    echo "No Stage 1 checkpoint found. Set STAGE1_CHECKPOINT=/abs/path/to/model.pt first." >&2
    exit 1
  fi
  mapfile -t COMMON_ARGS < <(build_common_args "$STAGE2_RUN_NAME" "$STAGE2_ENVS" "$STAGE2_ITERS")
  uv run ame-train \
    --variant "$VARIANT" \
    --stage stage2 \
    --checkpoint-file "$checkpoint" \
    "${GPU_ARGS[@]}" \
    "${COMMON_ARGS[@]}"
}

run_smoke() {
  uv run ame-train \
    --variant "$VARIANT" \
    --stage stage1 \
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
  stage1)
    run_stage1
    ;;
  stage2)
    run_stage2
    ;;
  full)
    run_stage1
    run_stage2
    ;;
  *)
    cat >&2 <<'EOF'
Usage:
  scripts/train_ame1_g1.sh smoke
  scripts/train_ame1_g1.sh stage1
  scripts/train_ame1_g1.sh stage2
  scripts/train_ame1_g1.sh full

Optional environment variables:
  VARIANT=g1
  VARIANT=paper
  GPU_ARGS="--gpu-ids [0]"
  GPU_ARGS="--gpu-ids [0,1]"
  STAGE1_ENVS=4096
  STAGE2_ENVS=4096
  STAGE1_ITERS=15000
  STAGE2_ITERS=3200
  STAGE1_RUN_NAME=my_stage1
  STAGE2_RUN_NAME=my_stage2
  STAGE1_CHECKPOINT=/abs/path/to/model.pt
  LOGGER=tensorboard
  LOGGER=wandb
  WANDB_PROJECT=ame_g1
  WANDB_USERNAME=my_team_or_user
  WANDB_ENTITY=my_team_or_user
  WANDB_MODE=offline
  WANDB_TAGS=ame,g1,stage1
  UPLOAD_MODEL=false
  VIDEO=true
  VIDEO_LENGTH=200
  VIDEO_INTERVAL=2000
  NAN_GUARD=true
EOF
    exit 2
    ;;
esac
