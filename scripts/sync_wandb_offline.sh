#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

WANDB_DIR="${WANDB_DIR:-$ROOT_DIR/wandb}"
PATTERN="${1:-offline-run-*}"

if [[ ! -d "$WANDB_DIR" ]]; then
  echo "wandb directory not found: $WANDB_DIR" >&2
  exit 1
fi

shopt -s nullglob
runs=("$WANDB_DIR"/$PATTERN)
shopt -u nullglob

if [[ "${#runs[@]}" -eq 0 ]]; then
  echo "No offline wandb runs found under $WANDB_DIR matching $PATTERN" >&2
  exit 1
fi

.venv/bin/wandb sync "${runs[@]}"
