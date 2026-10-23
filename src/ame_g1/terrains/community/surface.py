# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Unmodified triangle conversion from declared IsaacLab v2.3.0 baseline.

Source commit: 3c6e67bb5c7ada942a6d1884ab69338f57596f77.
Source formatting and imports are adapted for mjlab.
"""
from __future__ import annotations
import numpy as np

def convert_height_field_to_mesh(height_field: np.ndarray, horizontal_scale: float, vertical_scale: float, slope_threshold: float | None=None) -> tuple[np.ndarray, np.ndarray]:
    """Convert a height-field array to a triangle mesh represented by vertices and triangles.

    This function converts a height-field array to a triangle mesh represented by vertices and triangles.
    The height-field array is assumed to be a 2D array of floats, where each element represents the height
    of the terrain at that location. The height-field array is assumed to be in the form of a matrix, where
    the first dimension represents the x-axis and the second dimension represents the y-axis.

    The function can also correct vertical surfaces above the provide slope threshold. This is helpful to
    avoid having long vertical surfaces in the mesh. The correction is done by moving the vertices of the
    vertical surfaces to minimum of the two neighboring vertices.

    The correction is done in the following way:
    If :math:`\\frac{y_2 - y_1}{x_2 - x_1} > threshold`, then move A to A' (i.e., set :math:`x_1' = x_2`).
    This is repeated along all directions.

    .. code-block:: none

                B(x_2,y_2)
                    /|
                   / |
                  /  |
        (x_1,y_1)A---A'(x_1',y_1)

    Args:
        height_field: The input height-field array.
        horizontal_scale: The discretization of the terrain along the x and y axis.
        vertical_scale: The discretization of the terrain along the z axis.
        slope_threshold: The slope threshold above which surfaces are made vertical.
            Defaults to None, in which case no correction is applied.

    Returns:
        The vertices and triangles of the mesh:
        - **vertices** (np.ndarray(float)): Array of shape (num_vertices, 3).
          Each row represents the location of each vertex (in m).
        - **triangles** (np.ndarray(int)): Array of shape (num_triangles, 3).
          Each row represents the indices of the 3 vertices connected by this triangle.
    """
    (num_rows, num_cols) = height_field.shape
    y = np.linspace(0, (num_cols - 1) * horizontal_scale, num_cols)
    x = np.linspace(0, (num_rows - 1) * horizontal_scale, num_rows)
    (yy, xx) = np.meshgrid(y, x)
    hf = height_field.copy()
    if slope_threshold is not None:
        slope_threshold *= horizontal_scale / vertical_scale
        move_x = np.zeros((num_rows, num_cols))
        move_y = np.zeros((num_rows, num_cols))
        move_corners = np.zeros((num_rows, num_cols))
        move_x[:num_rows - 1, :] += hf[1:num_rows, :] - hf[:num_rows - 1, :] > slope_threshold
        move_x[1:num_rows, :] -= hf[:num_rows - 1, :] - hf[1:num_rows, :] > slope_threshold
        move_y[:, :num_cols - 1] += hf[:, 1:num_cols] - hf[:, :num_cols - 1] > slope_threshold
        move_y[:, 1:num_cols] -= hf[:, :num_cols - 1] - hf[:, 1:num_cols] > slope_threshold
        move_corners[:num_rows - 1, :num_cols - 1] += hf[1:num_rows, 1:num_cols] - hf[:num_rows - 1, :num_cols - 1] > slope_threshold
        move_corners[1:num_rows, 1:num_cols] -= hf[:num_rows - 1, :num_cols - 1] - hf[1:num_rows, 1:num_cols] > slope_threshold
        xx += (move_x + move_corners * (move_x == 0)) * horizontal_scale
        yy += (move_y + move_corners * (move_y == 0)) * horizontal_scale
    vertices = np.zeros((num_rows * num_cols, 3), dtype=np.float32)
    vertices[:, 0] = xx.flatten()
    vertices[:, 1] = yy.flatten()
    vertices[:, 2] = hf.flatten() * vertical_scale
    triangles = -np.ones((2 * (num_rows - 1) * (num_cols - 1), 3), dtype=np.uint32)
    for i in range(num_rows - 1):
        ind0 = np.arange(0, num_cols - 1) + i * num_cols
        ind1 = ind0 + 1
        ind2 = ind0 + num_cols
        ind3 = ind2 + 1
        start = 2 * i * (num_cols - 1)
        stop = start + 2 * (num_cols - 1)
        triangles[start:stop:2, 0] = ind0
        triangles[start:stop:2, 1] = ind3
        triangles[start:stop:2, 2] = ind1
        triangles[start + 1:stop:2, 0] = ind0
        triangles[start + 1:stop:2, 1] = ind2
        triangles[start + 1:stop:2, 2] = ind3
    return (vertices, triangles)
