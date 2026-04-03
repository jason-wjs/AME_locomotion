"""Terrain generator configuration for AME."""

from __future__ import annotations

from mjlab.terrains import (
  BoxNarrowBeamsTerrainCfg,
  BoxOpenStairsTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainGeneratorCfg,
)

from ame_g1.config.variants import AmePhase
from ame_g1.terrains.families import (
  GapTerrainCfg,
  PalletTerrainCfg,
  StairRunTerrainCfg,
  StonePathTerrainCfg,
)

_TERRAIN_SIZE = (8.0, 4.0)
_PLAY_TERRAIN_ROWS = 5
_TRAIN_TERRAIN_TILE_BUDGET = 40


def _paper_base_terrains():
  return {
    "pallets": PalletTerrainCfg(proportion=1.0),
    "gaps": GapTerrainCfg(proportion=1.0),
    "grid_stones": StonePathTerrainCfg(proportion=1.0, num_columns=3),
    "beams": BoxNarrowBeamsTerrainCfg(
      proportion=1.0,
      num_beams=14,
      beam_width_range=(0.18, 0.12),
      beam_height=0.18,
      spacing=0.7,
      platform_width=1.0,
      border_width=0.35,
      floor_depth=1.8,
    ),
  }


def _guide_extra_base_terrains():
  return {
    "stairs": StairRunTerrainCfg(proportion=1.0),
    "pits": BoxOpenStairsTerrainCfg(
      proportion=1.0,
      inverted=True,
      step_height_range=(0.08, 0.18),
      step_width_range=(0.45, 0.28),
      platform_width=1.0,
      border_width=0.35,
    ),
    "rough": HfRandomUniformTerrainCfg(
      proportion=1.0,
      noise_range=(-0.08, 0.08),
      noise_step=0.01,
      horizontal_scale=0.1,
      vertical_scale=0.005,
      border_width=0.5,
    ),
  }


def _paper_finetune_only_terrains():
  return {
    "pentagon_stones": StonePathTerrainCfg(
      proportion=1.0,
      num_columns=2,
      pentagon_top=True,
      stone_length_range=(0.28, 0.18),
      stone_width_range=(0.28, 0.18),
      stone_gap_range=(0.16, 0.42),
      stone_height_range=(0.04, 0.18),
    ),
    "single_column_stones": StonePathTerrainCfg(
      proportion=1.0,
      num_columns=1,
      stone_length_range=(0.26, 0.18),
      stone_width_range=(0.26, 0.18),
      stone_gap_range=(0.16, 0.42),
      stone_height_range=(0.02, 0.2),
    ),
    "narrow_pallets": PalletTerrainCfg(
      proportion=1.0,
      pallet_width_range=(0.45, 0.18),
      pallet_length_range=(0.38, 0.24),
      gap_range=(0.18, 0.45),
      height_range=(0.0, 0.2),
    ),
    "consecutive_gaps": GapTerrainCfg(
      proportion=1.0,
      support_width_range=(0.48, 0.26),
      support_length_range=(0.42, 0.24),
      gap_range=(0.22, 0.55),
      num_gaps_range=(5, 8),
    ),
  }


def _guide_extra_finetune_terrains():
  return {
    "narrow_stairs": StairRunTerrainCfg(
      proportion=1.0,
      step_height_range=(0.06, 0.12),
      step_depth_range=(0.28, 0.18),
      stair_width_range=(0.55, 0.28),
    ),
  }


def make_phase_terrain_generator(phase: AmePhase, play: bool = False) -> TerrainGeneratorCfg:
  base_terrains = _paper_base_terrains()
  if phase == "finetune":
    sub_terrains = {**base_terrains, **_paper_finetune_only_terrains()}
  else:
    sub_terrains = base_terrains
  if play:
    num_rows = _PLAY_TERRAIN_ROWS
  else:
    # Keep the training terrain tile budget stable across phases so finetune can
    # add terrain families without doubling the imported terrain world size.
    num_rows = max(1, _TRAIN_TERRAIN_TILE_BUDGET // len(sub_terrains))

  return TerrainGeneratorCfg(
    seed=42,
    curriculum=not play,
    size=_TERRAIN_SIZE,
    border_width=1.0,
    border_height=1.2,
    num_rows=num_rows,
    num_cols=len(sub_terrains),
    difficulty_range=(0.0, 1.0),
    color_scheme="height",
    sub_terrains=sub_terrains,
  )
