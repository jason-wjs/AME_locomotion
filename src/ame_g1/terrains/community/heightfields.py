# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Community base sparse height arrays, before collision representation.

Stepping stones follows IsaacLab v2.3.0 hf_terrains.py; concentric gaps follows
the fetched AME_Locomotion loco_hf_terrains.py. A supplied NumPy generator owns
all draws. MuJoCo collision geometry is deliberately not constructed here.
"""
from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class SparseHeightfield:
    size: tuple[float, float] = (8.0, 8.0)
    horizontal_scale: float = 0.05
    vertical_scale: float = 0.005
    border_width: float = 0.25
    platform_width: float = 2.0


def stepping_stones(difficulty, cfg, rng):
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    stone_width = int((0.5 - difficulty * 0.25) / cfg.horizontal_scale)
    distance = int((0.05 + difficulty * 0.2) / cfg.horizontal_scale)
    height_max = int(0.05 / cfg.vertical_scale)
    heights = np.arange(-height_max - 1, height_max)
    field = np.full((width, length), int(-2.0 / cfg.vertical_scale))
    # The actual source base tiles are square, so the source column-wise branch
    # is the complete required case. No unused rectangular-terrain mode is added.
    assert width == length
    start_y = 0
    while start_y < length:
        stop_y = min(length, start_y + stone_width)
        start_x = int(rng.integers(0, stone_width))
        stop_x = max(0, start_x - distance)
        field[:stop_x, start_y:stop_y] = rng.choice(heights)
        while start_x < width:
            stop_x = min(width, start_x + stone_width)
            field[start_x:stop_x, start_y:stop_y] = rng.choice(heights)
            start_x += stone_width + distance
        start_y += stone_width + distance
    platform = int(cfg.platform_width / cfg.horizontal_scale)
    low, high = (width - platform) // 2, (width + platform) // 2
    field[low:high, low:high] = 0
    return np.rint(field).astype(np.int16)


def concentric_gaps(difficulty, cfg, rng):
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    gap = int((0.1 + difficulty * 0.4) / cfg.horizontal_scale)
    ground = int(0.5 / cfg.horizontal_scale)
    height_max = int(0.025 / cfg.vertical_scale)
    platform = int(cfg.platform_width / cfg.horizontal_scale)
    field = np.zeros((width, length))
    x, y, end_x, end_y = 0, 0, width, length
    is_gap = True
    while end_x - x > platform and end_y - y > platform:
        if is_gap:
            field[x:end_x, y:end_y] = -int(2.0 / cfg.vertical_scale)
            step = gap
        else:
            field[x:end_x, y:end_y] = rng.integers(-height_max, height_max + 1)
            step = ground
        x, y, end_x, end_y = x + step, y + step, end_x - step, end_y - step
        is_gap = not is_gap
    field[(width-platform)//2:(width+platform)//2,
          (length-platform)//2:(length+platform)//2] = 0
    return np.rint(field).astype(np.int16)


def bordered_heightfield(generator, difficulty, cfg, rng):
    """Preserve source +1 grid, padded border and central maximum origin."""
    width = int(cfg.size[0] / cfg.horizontal_scale) + 1
    length = int(cfg.size[1] / cfg.horizontal_scale) + 1
    border = int(cfg.border_width / cfg.horizontal_scale) + 1
    inner = replace(cfg, size=((width - 2*border) * cfg.horizontal_scale,
                              (length - 2*border) * cfg.horizontal_scale))
    field = np.zeros((width, length), dtype=np.int16)
    field[border:-border, border:-border] = generator(difficulty, inner, rng)
    x1, x2 = (int((cfg.size[0]/2 + offset) / cfg.horizontal_scale) for offset in (-1, 1))
    y1, y2 = (int((cfg.size[1]/2 + offset) / cfg.horizontal_scale) for offset in (-1, 1))
    origin = np.array([cfg.size[0]/2, cfg.size[1]/2,
                       field[x1:x2, y1:y2].max() * cfg.vertical_scale])
    return field, origin
