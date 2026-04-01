"""Variant and stage registry for AME tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AmeStage = Literal["stage1", "stage2"]
AmeVariant = Literal["paper", "g1"]

DEFAULT_VARIANT: AmeVariant = "g1"
TRAIN_NUM_ENVS = 4096
PLAY_NUM_ENVS = 1


@dataclass(frozen=True)
class AmeTaskSpec:
  """Static task identifiers for a specific AME variant/stage pair."""

  variant: AmeVariant
  stage: AmeStage
  task_id: str
  experiment_name: str
  train_num_envs: int = TRAIN_NUM_ENVS
  play_num_envs: int = PLAY_NUM_ENVS


_TASK_SPECS: dict[tuple[AmeVariant, AmeStage], AmeTaskSpec] = {
  ("paper", "stage1"): AmeTaskSpec(
    variant="paper",
    stage="stage1",
    task_id="AME-Paper-Stage1-Unitree-G1",
    experiment_name="ame_paper_v2_stage1",
  ),
  ("paper", "stage2"): AmeTaskSpec(
    variant="paper",
    stage="stage2",
    task_id="AME-Paper-Stage2-Unitree-G1",
    experiment_name="ame_paper_v2_stage2",
  ),
  ("g1", "stage1"): AmeTaskSpec(
    variant="g1",
    stage="stage1",
    task_id="AME-G1-Stage1-Unitree-G1",
    experiment_name="ame_g1_v2_stage1",
  ),
  ("g1", "stage2"): AmeTaskSpec(
    variant="g1",
    stage="stage2",
    task_id="AME-G1-Stage2-Unitree-G1",
    experiment_name="ame_g1_v2_stage2",
  ),
}


def iter_task_specs() -> tuple[AmeTaskSpec, ...]:
  """Return all registered task specs in a stable order."""

  return tuple(
    _TASK_SPECS[(variant, stage)]
    for variant in ("paper", "g1")
    for stage in ("stage1", "stage2")
  )


def resolve_task_spec(
  variant: AmeVariant = DEFAULT_VARIANT,
  stage: AmeStage = "stage1",
) -> AmeTaskSpec:
  """Return the task spec for the requested variant/stage pair."""

  try:
    return _TASK_SPECS[(variant, stage)]
  except KeyError as error:
    raise ValueError(f"Unsupported AME task pair: variant={variant!r}, stage={stage!r}") from error

