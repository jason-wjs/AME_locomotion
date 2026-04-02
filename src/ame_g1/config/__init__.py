"""Configuration package for AME."""

from ame_g1.config.g1 import register_tasks as register
from ame_g1.config.variants import AmePhase as AmePhase
from ame_g1.config.variants import AmeTaskSpec as AmeTaskSpec
from ame_g1.config.variants import AmeVariant as AmeVariant
from ame_g1.config.variants import DEFAULT_VARIANT as DEFAULT_VARIANT
from ame_g1.config.variants import iter_task_specs as iter_task_specs
from ame_g1.config.variants import resolve_task_spec as resolve_task_spec

__all__ = [
  "AmePhase",
  "AmeTaskSpec",
  "AmeVariant",
  "DEFAULT_VARIANT",
  "iter_task_specs",
  "register",
  "resolve_task_spec",
]
