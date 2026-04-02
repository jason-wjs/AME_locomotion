"""Variant and phase registry for AME tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

AmePhase = Literal["base", "finetune"]
AmeVariant = Literal["paper", "g1"]

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

  return tuple(
    _TASK_SPECS[(variant, phase)]
    for variant in ("paper", "g1")
    for phase in ("base", "finetune")
  )


def resolve_task_spec(
  variant: AmeVariant = DEFAULT_VARIANT,
  phase: AmePhase = "base",
) -> AmeTaskSpec:
  """Return the task spec for the requested variant/phase pair."""

  try:
    return _TASK_SPECS[(variant, phase)]
  except KeyError as error:
    raise ValueError(f"Unsupported AME task pair: variant={variant!r}, phase={phase!r}") from error
