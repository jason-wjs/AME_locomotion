"""AME training entrypoint."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import tyro

import mjlab
from ame_g1 import assert_installed_mjlab
from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import DEFAULT_VARIANT, resolve_task_spec
from ame_g1.learning.runner import launch_ame_training
from mjlab.scripts.train import TrainConfig

@dataclass(frozen=True)
class AmeTrainArgs:
  checkpoint_file: str | None = None


@dataclass(frozen=True)
class TaskArgs:
  variant: Literal["paper", "g1"] = DEFAULT_VARIANT
  stage: Literal["stage1", "stage2"] = "stage1"


def main() -> None:
  assert_installed_mjlab(mjlab)
  register_ame_tasks()
  task_args, remaining_args = tyro.cli(
    TaskArgs,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  task_spec = resolve_task_spec(task_args.variant, task_args.stage)
  ame_args, remaining_args = tyro.cli(
    AmeTrainArgs,
    args=remaining_args,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(task_spec.task_id),
    prog=f"{Path(sys.argv[0]).name} --variant {task_args.variant} --stage {task_args.stage}",
    config=mjlab.TYRO_FLAGS,
  )
  checkpoint_path: Path | None = None
  if ame_args.checkpoint_file is not None:
    checkpoint_path = Path(ame_args.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
  launch_ame_training(task_spec=task_spec, args=args, checkpoint_file=checkpoint_path)


if __name__ == "__main__":
  main()
