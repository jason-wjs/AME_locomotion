# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Fixed community base continuous terrain geometry, before collision emission.

Adapted from IsaacLab v2.3.0 (3c6e67bb5c7ada942a6d1884ab69338f57596f77):
terrains/height_field/hf_terrains.py and terrains/trimesh/mesh_terrains.py.
Community recipe: 8 m square; grid .45 m, platform 2 m; rough .02..10 m
in .02 m steps, .1 m coarse grid; signed pyramid slope 0..0.4.
Height functions use heightfields.SparseHeightfield/bordered_heightfield.
Draws belong to the supplied NumPy generator, not historical Torch/NumPy RNGs.
"""
import numpy as np


def rough(difficulty, cfg, rng):
    """Source roughness ignores difficulty; return unpadded integer heights."""
    from scipy.interpolate import RectBivariateSpline

    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    coarse_width = int(cfg.size[0] / 0.1)
    coarse_length = int(cfg.size[1] / 0.1)
    low, high = int(0.02 / cfg.vertical_scale), int(0.10 / cfg.vertical_scale)
    step = int(0.02 / cfg.vertical_scale)
    samples = rng.choice(np.arange(low, high + step, step),
                         size=(coarse_width, coarse_length))
    # Keep the source coordinates, including their common scale factor.
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, coarse_width)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, coarse_length)
    spline = RectBivariateSpline(x, y, samples)
    fine_x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width)
    fine_y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length)
    return np.rint(spline(fine_x, fine_y)).astype(np.int16)


def pyramid_slope(difficulty, cfg, rng, *, inverted=False):
    """Source signed pyramid and platform clipping; no random draws."""
    slope = -difficulty * 0.4 if inverted else difficulty * 0.4
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    height = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    center_x, center_y = int(width / 2), int(length / 2)
    x, y = np.arange(width), np.arange(length)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = ((center_x - np.abs(center_x - xx)) / center_x).reshape(width, 1)
    yy = ((center_y - np.abs(center_y - yy)) / center_y).reshape(1, length)
    field = height * xx * yy
    half_platform = int(cfg.platform_width / cfg.horizontal_scale / 2)
    platform_z = field[width // 2 - half_platform, length // 2 - half_platform]
    return np.rint(np.clip(field, min(0, platform_z), max(0, platform_z))).astype(np.int16)


def grid_boxes(difficulty, rng):
    """Return source-equivalent box min/max bounds (metres) and center origin.

    All cells are kept, including cells partially under the platform. Removing
    cells by center would leave unsupported strips around its edges.
    """
    size, cell, platform = 8.0, 0.45, 2.0
    height = 0.05 + difficulty * (0.2 - 0.05)
    count = int(size / cell)
    residual = size - count * cell
    edge = residual / 2
    boxes = [
        [[0, 0, -1], [size, edge, 0]],
        [[0, size - edge, -1], [size, size, 0]],
        [[0, edge, -1], [edge, size - edge, 0]],
        [[size - edge, edge, -1], [size, size - edge, 0]],
    ]
    # Source Torch draws one scalar for every cell, then offsets only top Z.
    tops = np.asarray(rng.uniform(-height, height, (count, count)), dtype=np.float32)
    # Source Torch builds the offsets/noise in float32, then adds them to
    # float64 template vertices. Retain that quantization for bound parity.
    offsets = np.arange(count, dtype=np.float32) * cell + np.float32(residual / 2)
    for i in range(count):
        for j in range(count):
            x, y = float(offsets[i]), float(offsets[j])
            boxes.append([[x, y, -1], [x + cell, y + cell, tops[i, j]]])
    low, high = (size - platform) / 2, (size + platform) / 2
    boxes.append([[low, low, -1], [high, high, height]])
    return np.asarray(boxes, dtype=float), np.array([size / 2, size / 2, height])
