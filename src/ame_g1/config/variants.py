"""Variant and phase registry for AME tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AmePhase = Literal["base", "finetune"]
AmeVariant = Literal["paper", "g1", "ame2", "ame2-gaze"]

DEFAULT_VARIANT: AmeVariant = "g1"
TRAIN_NUM_ENVS = 4096
PLAY_NUM_ENVS = 1


@dataclass(frozen=True)
class AmeTaskSpec:
  """Static task identifiers for a specific AME variant/phase pair."""

  variant: AmeVariant
  phase: AmePhase
  task_id: str
  experiment_name: str
  train_num_envs: int = TRAIN_NUM_ENVS
  play_num_envs: int = PLAY_NUM_ENVS


_TASK_SPECS: dict[tuple[AmeVariant, AmePhase], AmeTaskSpec] = {
  ("ame2-gaze", "base"): AmeTaskSpec(
    variant="ame2-gaze", phase="base", task_id="AME2-G1-Gaze-Teacher",
    experiment_name="ame2_g1_gaze_teacher",
  ),
  ("ame2", "base"): AmeTaskSpec(
    variant="ame2", phase="base", task_id="AME2-G1-Teacher",
    experiment_name="ame2_g1_teacher",
  ),
  ("paper", "base"): AmeTaskSpec(
    variant="paper",
    phase="base",
    task_id="AME-Paper-Base-Unitree-G1",
    experiment_name="ame_paper_v3_base",
  ),
  ("paper", "finetune"): AmeTaskSpec(
    variant="paper",
    phase="finetune",
    task_id="AME-Paper-Finetune-Unitree-G1",
    experiment_name="ame_paper_v3_finetune",
  ),
  ("g1", "base"): AmeTaskSpec(
    variant="g1",
    phase="base",
    task_id="AME-G1-Base-Unitree-G1",
    experiment_name="ame_g1_v3_base",
    train_num_envs=1792,
  ),
  ("g1", "finetune"): AmeTaskSpec(
    variant="g1",
    phase="finetune",
    task_id="AME-G1-Finetune-Unitree-G1",
    experiment_name="ame_g1_v3_finetune",
  ),
}


def iter_task_specs() -> tuple[AmeTaskSpec, ...]:
  """Return all registered task specs in a stable order."""

  return tuple(_TASK_SPECS.values())


def resolve_task_spec(
  variant: AmeVariant = DEFAULT_VARIANT,
  phase: AmePhase = "base",
) -> AmeTaskSpec:
  """Return the task spec for the requested variant/phase pair."""

  try:
    return _TASK_SPECS[(variant, phase)]
  except KeyError as error:
    raise ValueError(f"Unsupported AME task pair: variant={variant!r}, phase={phase!r}") from error
