# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from leggedrobotics/ame2_minimal 8beb9a6, ame2/tasks/terrains.py.
"""The twelve positive-weight AME-2 G1 terrain families.

mjlab curriculum uses one column per family: ten rows produce 120 tiles, versus
600 in the official configuration. Proportions remain robot spawning weights.
"""
from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg, TerrainGeneratorCfg

from .ame2_geometry import (
    RoughTerrainCfg,
    PyramidStairsCfg,
    PyramidStairsInvCfg,
    BoxesTerrainCfg,
    ObstaclesTerrainCfg,
    ClimbUpCfg,
    ClimbDownCfg,
    ClimbingConsecutiveCfg,
    GapTerrainCfg,
    PalletZcCfg,
    BalanceBeamCfg,
    StonesCfg,
)


def make_ame2_terrain_generator(num_rows: int = 10, seed: int = 42) -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=seed, curriculum=True, size=(12.0, 12.0), border_width=10.0,
        num_rows=num_rows, num_cols=12, sub_terrains={
            'rough': RoughTerrainCfg(
                proportion=0.05,
                max_slope_deg=20.0,
                max_roughness_m=0.16,
                undulation_res=3.0,
                border_width=0.25,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(0.0, 5.5),
                    y_range=(-5.5, 5.5),
                )},
            ),
            'stair_down': PyramidStairsCfg(
                proportion=0.05,
                max_step_height=0.27,
                border_size=2.0,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(1.0, 6.0),
                    y_range=(-0.2, 0.2),
                )},
            ),
            'stair_up': PyramidStairsInvCfg(
                proportion=0.05,
                max_step_height=0.27,
                border_size=2.0,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(1.0, 6.0),
                    y_range=(-0.2, 0.2),
                )},
            ),
            'boxes': BoxesTerrainCfg(
                proportion=0.05,
                num_boxes=20,
                box_size_range=(0.8, 2.0),
                min_max_height=0.05,
                max_max_height=0.3,
                platform_size=2.0,
                dilation_size=0.1,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(0.0, 5.5),
                    y_range=(-5.5, 5.5),
                )},
            ),
            'obstacles': ObstaclesTerrainCfg(
                proportion=0.05,
                max_slope_deg=20.0,
                undulation_res=3.0,
                max_obstacle_density=0.5,
                obstacle_height_range=(0.2, 2.0),
                obstacle_size_range=(0.3, 0.8),
                platform_size=2.0,
                dilation_size=0.1,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.2,
                    max_height_diff=0.35,
                    x_range=(2.0, 5.5),
                    y_range=(-5.5, 5.5),
                )},
            ),
            'climb_up': ClimbUpCfg(
                proportion=0.2,
                min_height=0.1,
                max_height=0.65,
                rough=True,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(2.6, 5.5),
                    y_range=(-3.0, 3.0),
                )},
                wall_x=2.5,
            ),
            'climb_down': ClimbDownCfg(
                proportion=0.05,
                min_height=0.2,
                max_height=0.88,
                rough=True,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(2.6, 5.5),
                    y_range=(-3.0, 3.0),
                )},
                wall_x=1.0,
            ),
            'consecutive': ClimbingConsecutiveCfg(
                proportion=0.05,
                min_height=0.05,
                max_height_lower=0.35,
                max_height_upper=0.35,
                rough=False,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.15,
                    x_range=(4.0, 5.0),
                    y_range=(-4.0, 4.0),
                )},
                wall_offsets=(0.8, 1.7, 2.2, 3.1),
            ),
            'gap': GapTerrainCfg(
                proportion=0.05,
                min_gap=0.18,
                max_gap=1.2,
                platform_height=2.0,
                gap_x=1.0,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.1,
                    x_range=(1.8, 5.0),
                    y_range=(-3.0, 3.0),
                )},
            ),
            'pallets': PalletZcCfg(
                proportion=0.05,
                platform_size=2.0,
                border_size=2.0,
                step_height=(0.0, 0.2),
                step_width=(0.4, 0.16),
                gap_size=(0.16, 0.24),
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(3.5, 5.8),
                    y_range=(-5.0, 5.0),
                )},
            ),
            'beam': BalanceBeamCfg(
                proportion=0.05,
                gap_x=1.0,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(4.0, 5.8),
                    y_range=(-0.5, 0.5),
                )},
            ),
            'stones': StonesCfg(
                proportion=0.3,
                boundary_size=2.0,
                min_box_size=0.41,
                max_box_size=0.71,
                higher_flat=True,
                max_box_height=0.06,
                flat_rise=(-0.05, 0.15),
                min_gap_size=0.08,
                max_gap_size=0.12,
                flat_patch_sampling={"target": FlatPatchSamplingCfg(
                    num_patches=30,
                    patch_radius=0.3,
                    max_height_diff=0.3,
                    x_range=(1.0, 5.8),
                    y_range=(-5.0, 5.0),
                )},
            ),
        },
    )
