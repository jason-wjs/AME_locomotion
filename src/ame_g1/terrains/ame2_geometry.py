# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from leggedrobotics/ame2_minimal 8beb9a6, ame2/tasks/custom_terrains.py.
"""AME-2 terrain geometry in native MuJoCo collision and ray geometry.

Meshes remain separate connected components. Roughened components are convexified
individually, approximating their local surface but never bridging separate stones.
Heightfields retain quantized elevations; Isaac's slope-threshold mesh vertex
shifting is not applied to native MuJoCo heightfields.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import mujoco
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage as ndimage
import trimesh

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput


def make_border(outer_size, inner_size, height, position):
    """Four independent boxes surrounding the inner rectangle."""
    ox, oy = outer_size
    ix, iy = inner_size
    cx, cy, cz = position
    boxes = []
    for size, center in (
        ((ox, (oy - iy) / 2, height), (cx, cy + (oy + iy) / 4, cz)),
        ((ox, (oy - iy) / 2, height), (cx, cy - (oy + iy) / 4, cz)),
        (((ox - ix) / 2, iy, height), (cx + (ox + ix) / 4, cy, cz)),
        (((ox - ix) / 2, iy, height), (cx - (ox + ix) / 4, cy, cz)),
    ):
        pose = np.eye(4)
        pose[:3, 3] = center
        boxes.append(trimesh.creation.box(size, transform=pose))
    return boxes


def _clip_to_tile(mesh, size):
    """Intersect a convex component with tile boundaries, retaining closed solids.

    Upstream rotated wall boxes can protrude into adjacent families. Clipping
    prevents neighboring tile surfaces from invalidating sampled target heights.
    """
    for axis, bound, sign in ((0, 0., -1), (0, size[0], 1), (1, 0., -1), (1, size[1], 1)):
        vertices = mesh.vertices
        signed = sign * (vertices[:, axis] - bound)
        inside = signed <= 1e-9
        if inside.all():
            continue
        if not inside.any():
            return None
        edges = mesh.edges_unique
        edges = edges[inside[edges[:, 0]] != inside[edges[:, 1]]]
        a, b = edges[:, 0], edges[:, 1]
        ratio = signed[a] / (signed[a] - signed[b])
        intersections = vertices[a] + ratio[:, None] * (vertices[b] - vertices[a])
        points = np.concatenate([vertices[inside], intersections])
        if np.any(np.ptp(points, axis=0) < 1e-9):
            return None
        mesh = trimesh.convex.convex_hull(points)
    return mesh


def _add_meshes(spec, meshes):
    geometries = []
    body = spec.body("terrain")
    for mesh in meshes:
        name = f"ame2_mesh_{len(spec.meshes)}"
        asset = spec.add_mesh(name=name, uservert=mesh.vertices.astype(np.float32).ravel().tolist(),
                              userface=mesh.faces.astype(np.int32).ravel().tolist())
        geom = body.add_geom(type=mujoco.mjtGeom.mjGEOM_MESH, meshname=asset.name, mass=0)
        geometries.append(TerrainGeometry(geom=geom))
    return geometries


def _add_heightfield(spec, physical_heights, size):
    # Source arrays index x,y; MuJoCo heightfield rows/columns index y,x.
    heights = physical_heights.T
    low = float(heights.min())
    span = max(float(heights.max()) - low, 0.001)
    field = spec.add_hfield(
        name=f"ame2_hfield_{len(spec.hfields)}", nrow=heights.shape[0], ncol=heights.shape[1],
        size=[size[0] / 2, size[1] / 2, span, 0.1],
        userdata=((heights - low) / span).astype(np.float32).ravel().tolist(),
    )
    geom = spec.body("terrain").add_geom(
        type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=field.name,
        pos=[size[0] / 2, size[1] / 2, low], mass=0,
    )
    return [TerrainGeometry(geom=geom, hfield=field)]


def _sample_targets(tile_spec, cfg, origin, rng):
    if not cfg.flat_patch_sampling:
        return None
    model = tile_spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    direction = np.array([0., 0., -1.])
    geom_id = np.zeros(1, dtype=np.int32)
    result = {}
    for name, patch in cfg.flat_patch_sampling.items():
        radius = patch.patch_radius
        spacing = patch.grid_resolution or 0.05
        axis = np.linspace(-radius, radius, max(3, int(np.ceil(2 * radius / spacing)) + 1))
        xx, yy = np.meshgrid(axis, axis)
        disk = np.stack([xx.ravel(), yy.ravel()], axis=-1)
        disk = disk[np.linalg.norm(disk, axis=1) <= radius + 1e-8]
        angles = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        offsets = np.concatenate([np.zeros((1, 2)), disk,
                                  radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)])
        lower = np.maximum(np.array([patch.x_range[0], patch.y_range[0]]) + origin[:2], radius)
        upper = np.minimum(np.array([patch.x_range[1], patch.y_range[1]]) + origin[:2],
                           np.array(cfg.size) - radius)
        if np.any(lower > upper):
            raise ValueError(f"{type(cfg).__name__}: target window {name!r} is outside the tile")
        accepted = []
        for _ in range(10000):
            center = rng.uniform(lower, upper)
            heights = []
            for point in center + offsets:
                start = np.array([point[0], point[1], 100.])
                distance = mujoco.mj_ray(model, data, start, direction, None, True, -1, geom_id)
                if distance < 0:
                    break
                heights.append(100. - distance)
            if len(heights) != len(offsets):
                continue
            if (max(heights) - min(heights) <= patch.max_height_diff
                    and patch.z_range[0] <= heights[0] - origin[2] <= patch.z_range[1]):
                accepted.append([*center, heights[0]])
                if len(accepted) == patch.num_patches:
                    break
        if len(accepted) != patch.num_patches:
            raise RuntimeError(f"{type(cfg).__name__}: found {len(accepted)}/{patch.num_patches} "
                               f"native-surface {name!r} targets in 10000 candidates")
        result[name] = np.asarray(accepted)
    return result


@dataclass(kw_only=True)
class _MeshTerrainCfg(SubTerrainCfg):
    def function(self, difficulty, spec, rng):
        raw_meshes, origin = self._generate(difficulty, self, rng)
        # A single source mesh may contain hundreds of disconnected stones.
        # MuJoCo computes one convex collision hull per asset, so split first.
        meshes = []
        for mesh in raw_meshes:
            for component in mesh.split(only_watertight=False):
                solid = _clip_to_tile(component.convex_hull, self.size)
                if solid is not None:
                    meshes.append(solid)
        tile = mujoco.MjSpec()
        tile.worldbody.add_body(name="terrain")
        _add_meshes(tile, meshes)
        patches = _sample_targets(tile, self, origin, rng)
        return TerrainOutput(origin=origin, geometries=_add_meshes(spec, meshes), flat_patches=patches)


@dataclass(kw_only=True)
class _HeightfieldTerrainCfg(SubTerrainCfg):
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_width: float = 0.0

    def function(self, difficulty, spec, rng):
        # Match IsaacLab v2.3.2 height_field_to_mesh's endpoint and border grid.
        border = int(self.border_width / self.horizontal_scale) + 1
        shape = tuple(int(s / self.horizontal_scale) + 1 for s in self.size)
        inner = replace(self, size=tuple((n - 2 * border) * self.horizontal_scale for n in shape))
        quantized = np.zeros(shape, dtype=np.int16)
        quantized[border:-border, border:-border] = self._generate(difficulty, inner, rng)
        heights = quantized.astype(np.float64) * self.vertical_scale
        lo = [int((s / 2 - 1) / self.horizontal_scale) for s in self.size]
        hi = [int((s / 2 + 1) / self.horizontal_scale) for s in self.size]
        spawn_z = heights[lo[0]:hi[0], lo[1]:hi[1]].max()
        origin = np.array([self.size[0] / 2, self.size[1] / 2, spawn_z])
        tile = mujoco.MjSpec()
        tile.worldbody.add_body(name="terrain")
        _add_heightfield(tile, heights, self.size)
        patches = _sample_targets(tile, self, origin, rng)
        return TerrainOutput(origin=origin, geometries=_add_heightfield(spec, heights, self.size), flat_patches=patches)


def climb_up(difficulty: float, cfg: ClimbUpCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Climb-up terrain: lower ground, a row of 4-5 rotated boxes at ``wall_x``, then a raised plateau.

    The robot starts at the tile center on the lower ground. Box yaw (up to +-45 deg) and
    roll/pitch (~+-5 deg) scale with difficulty. Boxes stay inside the tile along y.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    wall_x = cfg.wall_x
    nominal_height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    max_noise = cfg.rough_rise * difficulty
    meshes = []
    sc = np.array([tw / 2, tl / 2])
    ground_height = 0.1
    lg_w = sc[0] + wall_x
    lg_center_x = lg_w / 2
    lg_pose = np.eye(4)
    lg_pose[:3, 3] = [lg_center_x, sc[1], -ground_height / 2]
    lg_box = trimesh.creation.box([lg_w, tl, ground_height], transform=lg_pose)
    meshes.append(lg_box)
    p_w = tw - lg_w
    p_center_x = lg_w + p_w / 2
    p_pose = np.eye(4)
    p_pose[:3, 3] = [p_center_x, sc[1], nominal_height / 2]
    plateau = trimesh.creation.box([p_w, tl, nominal_height], transform=p_pose)
    meshes.append(plateau)
    num_boxes = rng.integers(4, 6)
    margin = 1.2
    y_range = tl - 2 * margin
    y_step = y_range / max(1, num_boxes - 1)
    for i in range(num_boxes):
        bh = nominal_height + rng.uniform(-0.05, 0.05)
        bw = rng.uniform(0.8, 1.6)
        bl = y_step * 1.2
        cy = margin + i * y_step + rng.uniform(-0.2, 0.2)
        cx = lg_w
        cz = bh / 2
        yaw = rng.uniform(-np.pi / 4, np.pi / 4) * difficulty
        roll = rng.uniform(-0.087, 0.087) * difficulty
        pitch = rng.uniform(-0.087, 0.087) * difficulty * 1.5
        pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
        pose[:3, 3] = [cx, cy, cz]
        box = trimesh.creation.box((bw, bl, bh), transform=pose)
        if cfg.rough:
            box = box.subdivide().subdivide().subdivide()
            top_threshold = bh * 0.9
            top_indices = np.where(box.vertices[:, 2] > top_threshold)[0]
            if len(top_indices) > 0:
                box.vertices[top_indices, 2] += rng.uniform(-max_noise, max_noise, size=len(top_indices))
        meshes.append(box)
    origin = np.array([sc[0], sc[1], 0.0])
    return (meshes, origin)

def rough_terrain(difficulty: float, cfg: RoughTerrainCfg, rng: np.random.Generator) -> np.ndarray:
    """Custom rough terrain with slopes and noise conditioned on difficulty."""
    target_slope_deg = difficulty * cfg.max_slope_deg
    target_roughness = difficulty * cfg.max_roughness_m
    slope_tan = np.tan(np.deg2rad(target_slope_deg))
    undulation_height_range = slope_tan * cfg.undulation_res
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    if target_slope_deg > 0.001:
        res_rows = int(cfg.size[0] / cfg.undulation_res) + 1
        res_cols = int(cfg.size[1] / cfg.undulation_res) + 1
        low_res_hf = rng.uniform(-undulation_height_range / 2, undulation_height_range / 2, (res_rows, res_cols))
        x = np.linspace(0, cfg.size[0], res_rows)
        y = np.linspace(0, cfg.size[1], res_cols)
        func = interpolate.RectBivariateSpline(x, y, low_res_hf)
        x_new = np.linspace(0, cfg.size[0], num_rows)
        y_new = np.linspace(0, cfg.size[1], num_cols)
        hf_undulations = func(x_new, y_new)
    else:
        hf_undulations = np.zeros((num_rows, num_cols))
    if target_roughness > 0.0001:
        hf_noise = rng.uniform(-target_roughness / 2, target_roughness / 2, (num_rows, num_cols))
    else:
        hf_noise = np.zeros((num_rows, num_cols))
    hf_total = hf_undulations + hf_noise
    return np.rint(hf_total / cfg.vertical_scale).astype(np.int16)

def pyramid_stairs(difficulty: float, cfg: PyramidStairsCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate a pyramid stair mesh terrain (ported)."""
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    step_width = rng.uniform(cfg.step_width[0], cfg.step_width[1])
    (tl, tw) = cfg.size
    num_steps = min(tl - 2 * cfg.border_size - cfg.platform_size, tw - 2 * cfg.border_size - cfg.platform_size) // (2 * step_width) + 1
    pos = [0.5 * tl, 0.5 * tw, 0.0]
    meshes = []
    border_meshes = make_border((tl, tw), (tl - 2 * cfg.border_size, tw - 2 * cfg.border_size), step_height, [pos[0], pos[1], -step_height / 2])
    meshes += border_meshes
    ground_h = rng.uniform(cfg.ground_height[0], cfg.ground_height[1])
    dim = [tl, tw, 0.1]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1], -0.05 + ground_h]
    env_length = tl - 2 * cfg.border_size
    env_width = tw - 2 * cfg.border_size
    for k in range(int(num_steps)):
        if cfg.holes:
            box_length = rng.uniform(cfg.passage_width[0], cfg.passage_width[1])
            box_width = rng.uniform(cfg.passage_width[0], cfg.passage_width[1])
        else:
            box_length = env_length - k * 2 * step_width
            box_width = env_width - k * 2 * step_width
        dims_ns = [box_length, step_width + rng.uniform(-0.02, 0.02), 0.02]
        pose_n = np.eye(4)
        pose_n[:3, -1] = [pos[0], pos[1] + env_width / 2 - (k + 0.5) * step_width, pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ns, pose_n))
        pose_s = np.eye(4)
        pose_s[:3, -1] = [pos[0], pos[1] - env_width / 2 + (k + 0.5) * step_width, pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ns, pose_s))
        dims_ew = [step_width + rng.uniform(-0.02, 0.02), box_width, 0.02]
        pose_e = np.eye(4)
        pose_e[:3, -1] = [pos[0] - env_length / 2 + (k + 0.5) * step_width, pos[1], pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ew, pose_e))
        pose_w = np.eye(4)
        pose_w[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1], pos[2] + (k + 1) * step_height - 0.01]
        meshes.append(trimesh.creation.box(dims_ew, pose_w))
        if cfg.rand_walls:
            for side in [1, -1]:
                if rng.uniform(0.0, 1.0) < cfg.wall_prob:
                    wh = rng.uniform(cfg.wall_h[0], cfg.wall_h[1])
                    wt = rng.uniform(cfg.wall_thick[0], cfg.wall_thick[1])
                    dims_wall = [step_width, wt, wh]
                    pose_wall = np.eye(4)
                    pose_wall[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1] + side * (box_width / 2 + wt / 2), pos[2] + (k + 1) * step_height + wh / 2]
                    meshes.append(trimesh.creation.box(dims_wall, pose_wall))
    curr_k = int(num_steps) - 1
    dims_mid = [env_length - 2 * (curr_k + 1) * step_width, env_width - 2 * (curr_k + 1) * step_width, (curr_k + 2) * step_height]
    pose_mid = np.eye(4)
    pose_mid[:3, -1] = [pos[0], pos[1], pos[2] + curr_k * step_height / 2]
    meshes.append(trimesh.creation.box(dims_mid, pose_mid))
    origin = np.array([pos[0], pos[1], (num_steps + 1) * step_height])
    return (meshes, origin)

def pyramid_stairs_inv(difficulty: float, cfg: PyramidStairsInvCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Generate an inverted pyramid stair mesh terrain (ported)."""
    step_height = cfg.min_step_height + (cfg.max_step_height - cfg.min_step_height) * difficulty
    step_width = rng.uniform(cfg.step_width[0], cfg.step_width[1])
    (tl, tw) = cfg.size
    num_steps = min(tl - 2 * cfg.border_size - cfg.platform_size, tw - 2 * cfg.border_size - cfg.platform_size) // (2 * step_width) + 2
    pos = [0.5 * tl, 0.5 * tw, 0.0]
    tot_height = num_steps * step_height
    meshes = []
    border_meshes = make_border((tl, tw), (tl - 2 * cfg.border_size, tw - 2 * cfg.border_size), tot_height, [pos[0], pos[1], -tot_height / 2])
    meshes += border_meshes
    ground_h = rng.uniform(cfg.ground_height[0], cfg.ground_height[1])
    dim = [tl, tw, 0.1]
    pose = np.eye(4)
    pose[:3, -1] = [pos[0], pos[1], -0.05 + ground_h - tot_height]
    env_length = tl - 2 * cfg.border_size
    env_width = tw - 2 * cfg.border_size
    for k in range(int(num_steps) - 1):
        if cfg.holes:
            box_length = rng.uniform(cfg.passage_width[0], cfg.passage_width[1])
            box_width = rng.uniform(cfg.passage_width[0], cfg.passage_width[1])
        else:
            box_length = env_length - k * 2 * step_width
            box_width = env_width - k * 2 * step_width
        thickness = rng.uniform(0.02, step_height)
        target_top = pos[2] - (k + 1) * step_height
        dims_ns = [box_length, step_width + rng.uniform(-0.02, 0.02), thickness]
        pose_n = np.eye(4)
        pose_n[:3, -1] = [pos[0], pos[1] + env_width / 2 - (k + 0.5) * step_width, target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ns, pose_n))
        pose_s = np.eye(4)
        pose_s[:3, -1] = [pos[0], pos[1] - env_width / 2 + (k + 0.5) * step_width, target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ns, pose_s))
        dims_ew = [step_width + rng.uniform(-0.02, 0.02), box_width, thickness]
        pose_e = np.eye(4)
        pose_e[:3, -1] = [pos[0] - env_length / 2 + (k + 0.5) * step_width, pos[1], target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ew, pose_e))
        pose_w = np.eye(4)
        pose_w[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1], target_top - thickness / 2]
        meshes.append(trimesh.creation.box(dims_ew, pose_w))
        if cfg.rand_walls:
            for side in [1, -1]:
                if rng.uniform(0.0, 1.0) < cfg.wall_prob:
                    wh = rng.uniform(cfg.wall_h[0], cfg.wall_h[1])
                    wt = rng.uniform(cfg.wall_thick[0], cfg.wall_thick[1])
                    dims_wall = [step_width, wt, wh]
                    pose_wall = np.eye(4)
                    pose_wall[:3, -1] = [pos[0] + env_length / 2 - (k + 0.5) * step_width, pos[1] + side * (box_width / 2 + wt / 2), pos[2] - (k + 1) * step_height + wh / 2]
                    meshes.append(trimesh.creation.box(dims_wall, pose_wall))
    curr_k = int(num_steps) - 1
    # The source bottom platform is a zero-thickness mesh; give it a 2cm
    # solid underside for MuJoCo while retaining its original top surface.
    dims_mid = [cfg.platform_size, cfg.platform_size, 0.02]
    pose_mid = np.eye(4)
    pose_mid[:3, -1] = [pos[0], pos[1], pos[2] - tot_height / 2 - (curr_k + 1) * step_height / 2 - 0.01]
    meshes.append(trimesh.creation.box(dims_mid, pose_mid))
    origin = np.array([pos[0], pos[1], -(num_steps - 1) * step_height])
    return (meshes, origin)

def boxes_terrain(difficulty: float, cfg: BoxesTerrainCfg, rng: np.random.Generator) -> np.ndarray:
    """Generate a custom boxes height field terrain."""
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    hf = np.zeros((num_rows, num_cols))
    max_h = cfg.min_max_height + (cfg.max_max_height - cfg.min_max_height) * difficulty
    for _ in range(cfg.num_boxes):
        bw_m = rng.uniform(cfg.box_size_range[0], cfg.box_size_range[1])
        bl_m = rng.uniform(cfg.box_size_range[0], cfg.box_size_range[1])
        bw = max(1, int(bw_m / cfg.horizontal_scale))
        bl = max(1, int(bl_m / cfg.horizontal_scale))
        if num_rows - bw <= 0 or num_cols - bl <= 0:
            continue
        r = rng.integers(0, num_rows - bw)
        c = rng.integers(0, num_cols - bl)
        h = rng.uniform(-max_h, max_h)
        hf[r:r + bw, c:c + bl] += h
    if cfg.dilation_size > 0:
        d_size = max(1, int(cfg.dilation_size / cfg.horizontal_scale))
        hf_pos = np.maximum(hf, 0)
        hf_neg = np.minimum(hf, 0)
        hf_pos = ndimage.grey_dilation(hf_pos, size=(d_size, d_size))
        hf_neg = ndimage.grey_erosion(hf_neg, size=(d_size, d_size))
        hf = hf_pos + hf_neg
    p_rows = int(cfg.platform_size / cfg.horizontal_scale)
    p_cols = int(cfg.platform_size / cfg.horizontal_scale)
    r_start = (num_rows - p_rows) // 2
    c_start = (num_cols - p_cols) // 2
    hf[r_start:r_start + p_rows, c_start:c_start + p_cols] = 0.0
    return np.rint(hf / cfg.vertical_scale).astype(np.int16)

def obstacles_terrain(difficulty: float, cfg: ObstaclesTerrainCfg, rng: np.random.Generator) -> np.ndarray:
    """Generate a custom obstacles height field terrain."""
    num_rows = int(cfg.size[0] / cfg.horizontal_scale)
    num_cols = int(cfg.size[1] / cfg.horizontal_scale)
    target_slope_deg = difficulty * cfg.max_slope_deg
    slope_tan = np.tan(np.deg2rad(target_slope_deg))
    undulation_height_range = slope_tan * cfg.undulation_res
    if target_slope_deg > 0.001:
        res_rows = int(cfg.size[0] / cfg.undulation_res) + 1
        res_cols = int(cfg.size[1] / cfg.undulation_res) + 1
        low_res_hf = rng.uniform(-undulation_height_range / 2, undulation_height_range / 2, (res_rows, res_cols))
        x = np.linspace(0, cfg.size[0], res_rows)
        y = np.linspace(0, cfg.size[1], res_cols)
        func = interpolate.RectBivariateSpline(x, y, low_res_hf)
        x_new = np.linspace(0, cfg.size[0], num_rows)
        y_new = np.linspace(0, cfg.size[1], num_cols)
        hf = func(x_new, y_new)
    else:
        hf = np.zeros((num_rows, num_cols))
    target_density = difficulty * cfg.max_obstacle_density
    area = cfg.size[0] * cfg.size[1]
    num_obstacles = int(target_density * area)
    hf_obs = np.zeros_like(hf)
    p_rows = int(cfg.platform_size / cfg.horizontal_scale)
    p_cols = int(cfg.platform_size / cfg.horizontal_scale)
    r_start = (num_rows - p_rows) // 2
    r_end = r_start + p_rows
    c_start = (num_cols - p_cols) // 2
    c_end = c_start + p_cols
    for _ in range(num_obstacles):
        bw_m = rng.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])
        bl_m = rng.uniform(cfg.obstacle_size_range[0], cfg.obstacle_size_range[1])
        bw = max(1, int(bw_m / cfg.horizontal_scale))
        bl = max(1, int(bl_m / cfg.horizontal_scale))
        if num_rows - bw <= 0 or num_cols - bl <= 0:
            continue
        r = rng.integers(0, num_rows - bw)
        c = rng.integers(0, num_cols - bl)
        if r < r_end and r + bw > r_start and (c < c_end) and (c + bl > c_start):
            continue
        h = rng.uniform(cfg.obstacle_height_range[0], cfg.obstacle_height_range[1])
        if rng.random() > 0.5:
            h *= -1
        hf_obs[r:r + bw, c:c + bl] += h
    if cfg.dilation_size > 0:
        d_size = max(1, int(cfg.dilation_size / cfg.horizontal_scale))
        hf_obs_pos = np.maximum(hf_obs, 0)
        hf_obs_neg = np.minimum(hf_obs, 0)
        hf_obs_pos = ndimage.grey_dilation(hf_obs_pos, size=(d_size, d_size))
        hf_obs_neg = ndimage.grey_erosion(hf_obs_neg, size=(d_size, d_size))
        hf_obs = hf_obs_pos + hf_obs_neg
    hf += hf_obs
    return np.rint(hf / cfg.vertical_scale).astype(np.int16)

def climb_down(difficulty: float, cfg: ClimbDownCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Climb-down terrain: raised plateau, a row of 4-5 rotated boxes at ``wall_x``, then lower ground.

    The robot starts at the tile center on the raised plateau.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    wall_x = cfg.wall_x
    nominal_height = cfg.min_height + (cfg.max_height - cfg.min_height) * difficulty
    max_noise = cfg.rough_rise * difficulty
    meshes = []
    sc = np.array([tw / 2, tl / 2])
    hp_w = sc[0] + wall_x
    hp_center_x = hp_w / 2
    hp_pose = np.eye(4)
    hp_pose[:3, 3] = [hp_center_x, sc[1], nominal_height / 2]
    plateau = trimesh.creation.box([hp_w, tl, nominal_height], transform=hp_pose)
    meshes.append(plateau)
    lg_height = 0.1
    lg_w = tw - hp_w
    lg_center_x = hp_w + lg_w / 2
    lg_pose = np.eye(4)
    lg_pose[:3, 3] = [lg_center_x, sc[1], -lg_height / 2]
    lg_box = trimesh.creation.box([lg_w, tl, lg_height], transform=lg_pose)
    meshes.append(lg_box)
    num_boxes = rng.integers(4, 6)
    margin = 1.2
    y_range = tl - 2 * margin
    y_step = y_range / max(1, num_boxes - 1)
    for i in range(num_boxes):
        bh = nominal_height + rng.uniform(-0.05, 0.05)
        bw = rng.uniform(0.8, 1.6)
        bl = y_step * 1.2
        cy = margin + i * y_step + rng.uniform(-0.2, 0.2)
        cx = hp_w
        cz = bh / 2
        yaw = rng.uniform(-np.pi / 4, np.pi / 4) * difficulty
        roll = rng.uniform(-0.087, 0.087) * difficulty
        pitch = rng.uniform(-0.087, 0.087) * difficulty * 1.5
        pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
        pose[:3, 3] = [cx, cy, cz]
        box = trimesh.creation.box((bw, bl, bh), transform=pose)
        if cfg.rough:
            box = box.subdivide().subdivide().subdivide()
            top_threshold = bh * 0.9
            top_indices = np.where(box.vertices[:, 2] > top_threshold)[0]
            if len(top_indices) > 0:
                box.vertices[top_indices, 2] += rng.uniform(-max_noise, max_noise, size=len(top_indices))
        meshes.append(box)
    origin = np.array([sc[0], sc[1], nominal_height])
    return (meshes, origin)

def climbing_consecutive(difficulty: float, cfg: ClimbingConsecutiveCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom climbing_consecutive terrain function.
    
    Creates four sequential transitions (Up-Up-Down-Down) relative to the center.
    Wall 1: Climb UP to h1 (lower layer max)
    Wall 2: Climb UP to h1+h2 (upper layer max)
    Wall 3: Climb DOWN to h1 (upper layer max)
    Wall 4: Climb DOWN to Ground (lower layer max)
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2
    meshes = []
    h_min = cfg.min_height
    h_lower = h_min + (cfg.max_height_lower - h_min) * difficulty
    h_upper = h_min + (cfg.max_height_upper - h_min) * difficulty
    (x1, x2, x3, x4) = [sc[0] + dx for dx in cfg.wall_offsets]
    h1 = h_lower
    h_total = h_upper + h_lower
    p1_w = x2 - x1
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [x1 + p1_w / 2, sc[1], h1 / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, h1], transform=p1_pose))
    peak_w = x3 - x2
    peak_pose = np.eye(4)
    peak_pose[:3, 3] = [x2 + peak_w / 2, sc[1], h_total / 2]
    meshes.append(trimesh.creation.box([peak_w, tl, h_total], transform=peak_pose))
    p2_w = x4 - x3
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [x3 + p2_w / 2, sc[1], h1 / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, h1], transform=p2_pose))
    base_pose = np.eye(4)
    base_pose[:3, 3] = [tw / 2, tl / 2, -0.05]
    meshes.append(trimesh.creation.box([tw, tl, 0.1], transform=base_pose))
    walls = [(x1, h1), (x2, h_total), (x3, h_total), (x4, h1)]
    margin = 1.2
    y_range = tl - 2 * margin
    for (wall_idx, (wall_x, wall_h)) in enumerate(walls):
        num_boxes = rng.integers(8, 12)
        y_step = y_range / max(1, num_boxes - 1)
        for i in range(num_boxes):
            if rng.random() < 0.3:
                continue
            bh = wall_h + rng.uniform(-0.05, 0.05)
            bw = 0.5
            bl = 0.5
            cy = margin + i * y_step + rng.uniform(-0.35, 0.35)
            cz = bh / 2
            cx = wall_x + rng.uniform(-0.1, 0.1)
            yaw = np.pi / 4
            roll = 0.0
            pitch = 0.0
            pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
            pose[:3, 3] = [cx, cy, cz]
            box = trimesh.creation.box((bw, bl, bh), transform=pose)
            meshes.append(box)
    origin = np.array([sc[0], sc[1], 0.0])
    return (meshes, origin)

def gap_terrain(difficulty: float, cfg: GapTerrainCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom gap terrain function.
    
    Creates two platforms separated by a gap.
    Gap width scales from min_gap to max_gap with difficulty.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2
    meshes = []
    gap_width = cfg.min_gap + (cfg.max_gap - cfg.min_gap) * difficulty
    ph = cfg.platform_height
    p1_w = sc[0] + cfg.gap_x
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [p1_w / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, ph], transform=p1_pose))
    delta_h = rng.uniform(-0.25 * gap_width, 0.25 * gap_width)
    p2_start_x = p1_w + gap_width
    p2_w = tw - p2_start_x
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [p2_start_x + p2_w / 2, sc[1], delta_h - ph / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, ph], transform=p2_pose))
    if rng.uniform(0, 1) < cfg.floor_ratio:
        z_floor = rng.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))
    origin = np.array([sc[0], sc[1], 0])
    return (meshes, origin)

def balance_beam_terrain(difficulty: float, cfg: BalanceBeamCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom balance beam terrain function.
    
    Creates two platforms separated by a gap, bridged by a beam.
    Beam width scales with difficulty.
    """
    tw = cfg.size[0]
    tl = cfg.size[1]
    sc = np.array(cfg.size) / 2
    meshes = []
    gap_width = cfg.gap_size
    bw = cfg.beam_width[0] + (cfg.beam_width[1] - cfg.beam_width[0]) * difficulty
    ph = cfg.platform_height
    p1_w = sc[0] + cfg.gap_x
    p1_pose = np.eye(4)
    p1_pose[:3, 3] = [p1_w / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([p1_w, tl, ph], transform=p1_pose))
    delta_h = rng.uniform(-0.05 * difficulty, 0.05 * difficulty)
    p2_start_x = p1_w + gap_width
    p2_w = tw - p2_start_x
    p2_pose = np.eye(4)
    p2_pose[:3, 3] = [p2_start_x + p2_w / 2, sc[1], delta_h - ph / 2]
    meshes.append(trimesh.creation.box([p2_w, tl, ph], transform=p2_pose))
    max_rpy = cfg.max_rpy * difficulty
    roll = rng.uniform(-max_rpy, max_rpy)
    pitch = rng.uniform(-max_rpy, max_rpy)
    yaw = rng.uniform(-max_rpy, max_rpy)
    beam_pose = trimesh.transformations.euler_matrix(roll, pitch, yaw)
    beam_pose[:3, 3] = [p1_w + gap_width / 2, sc[1], -ph / 2]
    meshes.append(trimesh.creation.box([gap_width, bw, ph], transform=beam_pose))
    if rng.uniform(0, 1) < cfg.floor_ratio:
        z_floor = rng.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))
    origin = np.array([sc[0], sc[1], 0])
    return (meshes, origin)

def pallets_up_down_zc(difficulty: float, cfg: PalletZcCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Pallet terrain: rotated parallel slats with random height offsets and gaps.

    Slats are clipped to the tile bounds.
    """
    (tw, tl) = (cfg.size[0], cfg.size[1])
    sc = np.array(cfg.size) / 2
    meshes = []
    platform_pose = np.eye(4)
    platform_pose[:3, 3] = [sc[0], sc[1], -0.1]
    platform = trimesh.creation.box([cfg.platform_size, cfg.platform_size, 0.2], transform=platform_pose)
    meshes.append(platform)
    border_meshes = make_border((tw, tl), (tw - 2 * cfg.border_size, tl - 2 * cfg.border_size), 1.0, [sc[0], sc[1], -0.5])
    meshes += border_meshes
    terrain_ori = rng.uniform(-np.pi / 2, np.pi / 2)
    step_height = cfg.step_height[0] + (cfg.step_height[1] - cfg.step_height[0]) * difficulty
    step_width = cfg.step_width[0] + (cfg.step_width[1] - cfg.step_width[0]) * difficulty
    gap_size = cfg.gap_size[0] + (cfg.gap_size[1] - cfg.gap_size[0]) * difficulty
    R_full = trimesh.transformations.rotation_matrix(terrain_ori, [0, 0, 1])
    R = R_full[:3, :3]
    period = step_width + gap_size
    scale_coef = 1.0 / max(np.abs(R[0, 0]), np.abs(R[1, 0]), 1e-06)
    num_steps = int(max(tw, tl) * scale_coef / period)
    for i in range(-num_steps, num_steps + 1):
        x_off = i * period
        cx = sc[0] + x_off * R[0, 0]
        cy = sc[1] + x_off * R[1, 0]
        (ux, uy) = (R[0, 1], R[1, 1])
        (s_min, s_max) = (-max(tw, tl), max(tw, tl))
        valid = True
        for (b_val, dir_val, limit) in [(cx, ux, tw), (cy, uy, tl)]:
            if np.abs(dir_val) > 1e-06:
                (l1, l2) = (-b_val / dir_val, (limit - b_val) / dir_val)
                s_min = max(s_min, min(l1, l2))
                s_max = min(s_max, max(l1, l2))
            elif b_val < -0.001 or b_val > limit + 0.001:
                valid = False
                break
        if valid and s_max > s_min + 0.05:
            l_slat = s_max - s_min
            h_rand = step_height * rng.uniform(-0.5, 0.5) if np.abs(i) > 0 else 0.0
            s_pose = np.eye(4)
            s_pose[:3, 3] = [cx + (s_max + s_min) / 2 * ux, cy + (s_max + s_min) / 2 * uy, -0.1 + h_rand]
            s_pose[:3, :3] = R
            slat = trimesh.creation.box([step_width, l_slat, 0.2])
            slat.apply_transform(s_pose)
            meshes.append(slat)
    if rng.uniform(0, 1) < cfg.floor_ratio:
        z_floor = rng.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))
    origin = np.array([sc[0], sc[1], step_height])
    return (meshes, origin)

def stones_terrain(difficulty: float, cfg: StonesCfg, rng: np.random.Generator) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    """Custom stones terrain function.
    
    Creates a grid of stones with randomized shifts, gaps, and heights.
    """
    (tw, tl) = (cfg.size[0], cfg.size[1])
    sc = np.array(cfg.size) / 2
    box_height = cfg.min_box_height + (cfg.max_box_height - cfg.min_box_height) * difficulty
    gap_size = cfg.min_gap_size + (cfg.max_gap_size - cfg.min_gap_size) * difficulty
    box_size = cfg.max_box_size - (cfg.max_box_size - cfg.min_box_size) * difficulty
    ph_rise = rng.uniform(cfg.flat_rise[0], cfg.flat_rise[1]) if cfg.higher_flat else 0.0
    flat_rise = [ph_rise, ph_rise]
    meshes = []
    num_boxes_x = int((tw - cfg.boundary_size) / box_size)
    num_boxes_y = int((tl - cfg.boundary_size) / box_size)
    border_x = tw - num_boxes_x * box_size
    border_y = tl - num_boxes_y * box_size
    border_meshes = make_border((tw, tl), (tw - border_x, tl - border_y), 1.0, [sc[0], sc[1], -0.5])
    meshes += border_meshes
    box_dim = [box_size, box_size, 1.0]
    base_box = trimesh.creation.box(box_dim)
    base_box.apply_translation([box_size * 0.5, box_size * 0.5, -0.5])
    vertices = base_box.vertices
    faces = base_box.faces
    num_total_boxes = num_boxes_x * num_boxes_y
    all_indices = np.arange(num_total_boxes)
    sink_indices = rng.choice(all_indices, int(num_total_boxes * cfg.sink_ratio), replace=False)
    remaining_indices = np.setdiff1d(all_indices, sink_indices)
    rise_indices = rng.choice(remaining_indices, int(num_total_boxes * cfg.rise_ratio), replace=False)
    all_vertices = []
    all_faces = []
    x_indices = np.arange(num_boxes_x)
    y_indices = np.arange(num_boxes_y)
    (xx, yy) = np.meshgrid(x_indices, y_indices, indexing='ij')
    xx = xx.flatten()
    yy = yy.flatten().astype(float)
    row_shifts = rng.uniform(-0.5, 0.5, num_boxes_x)
    for i in range(num_boxes_x):
        yy[xx == i] += row_shifts[i]
    offsets = box_size * np.stack([xx, yy], axis=1)
    offsets[:, 0] += border_x / 2
    offsets[:, 1] += border_y / 2
    for i in range(num_total_boxes):
        v = vertices.copy()
        v[:, :2] += offsets[i]
        local_center = box_size / 2
        v[v[:, 0] > offsets[i, 0] + local_center, 0] -= gap_size
        v[v[:, 0] < offsets[i, 0] + local_center, 0] += gap_size
        v[v[:, 1] > offsets[i, 1] + local_center, 1] -= gap_size
        v[v[:, 1] < offsets[i, 1] + local_center, 1] += gap_size
        xy_noise = rng.uniform(-gap_size / 1.1, gap_size / 1.1, 2)
        z_noise = rng.uniform(-box_height, box_height)
        top_verts = v[:, 2] > -0.1
        v[top_verts, :2] += xy_noise
        v[top_verts, 2] += z_noise
        if i in sink_indices:
            v[:, 2] -= 1.5
        if i in rise_indices:
            dist_x = np.abs(offsets[i][0] + box_size / 2 - sc[0])
            dist_y = np.abs(offsets[i][1] + box_size / 2 - sc[1])
            exclusion_radius = cfg.platform_size / 2 + box_size / 2
            if dist_x >= exclusion_radius or dist_y >= exclusion_radius:
                v[top_verts, 2] += 1.5
        all_vertices.append(v)
        all_faces.append(faces + i * len(vertices))
    if num_total_boxes > 0:
        stones_mesh = trimesh.Trimesh(vertices=np.vstack(all_vertices), faces=np.vstack(all_faces))
        meshes.append(stones_mesh)
    platform_dim = [cfg.platform_size, cfg.platform_size, 1.0 + box_height]
    platform_pose = np.eye(4)
    platform_pose[:3, 3] = [sc[0], sc[1], -0.5 + box_height / 2 + flat_rise[0]]
    platform = trimesh.creation.box(platform_dim, transform=platform_pose)
    platform = platform.subdivide().subdivide()
    top_mask = platform.vertices[:, 2] > platform_pose[2, 3] + box_height / 2 - 0.01
    platform.vertices[top_mask] += rng.uniform(-0.02, 0.02, size=(np.sum(top_mask), 3))
    meshes.append(platform)
    if rng.uniform(0, 1) < cfg.floor_ratio:
        z_floor = rng.uniform(cfg.floor_height[0], cfg.floor_height[1])
        floor_thick = 0.1
        floor_pose = np.eye(4)
        floor_pose[:3, 3] = [tw / 2, tl / 2, z_floor - floor_thick / 2]
        meshes.append(trimesh.creation.box([tw, tl, floor_thick], transform=floor_pose))
    outer_border_meshes = make_border((tw, tl), (tw - 2 * cfg.boundary_size, tl - 2 * cfg.boundary_size), 0.2, [sc[0], sc[1], -0.1 + flat_rise[1]])
    meshes += outer_border_meshes
    origin = np.array([sc[0], sc[1], box_height + 0.05 + flat_rise[0]])
    return (meshes, origin)

@dataclass(kw_only=True)
class ClimbUpCfg(_MeshTerrainCfg):
    """Configuration for a custom climb_up terrain."""
    min_height: float = 0.5
    max_height: float = 1.0
    rough: bool = True
    rough_rise: float = 0.05
    box_height_diff_max: float = 0.05
    wall_x: float = 1.0
    _generate = staticmethod(climb_up)

@dataclass(kw_only=True)
class RoughTerrainCfg(_HeightfieldTerrainCfg):
    """Configuration for a custom rough terrain."""
    max_slope_deg: float = 20.0
    max_roughness_m: float = 0.16
    undulation_res: float = 2.0
    _generate = staticmethod(rough_terrain)

@dataclass(kw_only=True)
class PyramidStairsCfg(_MeshTerrainCfg):
    """Configuration for a pyramid stair mesh terrain ported from another simulator."""
    min_step_height: float = 0.05
    max_step_height: float = 0.23
    step_width: tuple[float, float] = (0.27, 0.35)
    border_size: float = 1.4
    platform_size: float = 2.0
    passage_width: tuple[float, float] = (0.8, 1.8)
    ground_height: tuple[float, float] = (-1.5, -0.0)
    holes: bool = True
    rand_walls: bool = True
    wall_h: tuple[float, float] = (0.4, 1.8)
    wall_prob: float = 0.35
    wall_thick: tuple[float, float] = (0.05, 0.2)
    _generate = staticmethod(pyramid_stairs)

@dataclass(kw_only=True)
class PyramidStairsInvCfg(PyramidStairsCfg):
    """Configuration for an inverted pyramid stair mesh terrain ported from another simulator."""
    _generate = staticmethod(pyramid_stairs_inv)

@dataclass(kw_only=True)
class BoxesTerrainCfg(_HeightfieldTerrainCfg):
    """Configuration for a custom boxes height field terrain."""
    num_boxes: int = 20
    box_size_range: tuple[float, float] = (0.4, 3.0)
    min_max_height: float = 0.05
    max_max_height: float = 0.3
    platform_size: float = 2.0
    dilation_size: float = 0.1
    _generate = staticmethod(boxes_terrain)

@dataclass(kw_only=True)
class ObstaclesTerrainCfg(_HeightfieldTerrainCfg):
    """Configuration for a custom obstacles height field terrain."""
    max_slope_deg: float = 20.0
    undulation_res: float = 3.0
    max_obstacle_density: float = 0.5
    obstacle_height_range: tuple[float, float] = (0.1, 0.5)
    obstacle_size_range: tuple[float, float] = (0.3, 0.8)
    platform_size: float = 2.0
    dilation_size: float = 0.1
    _generate = staticmethod(obstacles_terrain)

@dataclass(kw_only=True)
class ClimbDownCfg(ClimbUpCfg):
    """Configuration for a custom climb_down terrain."""
    _generate = staticmethod(climb_down)

@dataclass(kw_only=True)
class ClimbingConsecutiveCfg(_MeshTerrainCfg):
    """Configuration for a climbing consecutive hill."""
    min_height: float = 0.05
    max_height_lower: float = 0.4
    max_height_upper: float = 0.5
    rough: bool = False
    wall_offsets: tuple[float, float, float, float] = (0.8, 1.7, 2.2, 3.1)
    _generate = staticmethod(climbing_consecutive)

@dataclass(kw_only=True)
class GapTerrainCfg(_MeshTerrainCfg):
    """Configuration for a custom gap terrain."""
    min_gap: float = 0.1
    max_gap: float = 0.5
    platform_height: float = 0.5
    gap_x: float = 0.0
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    _generate = staticmethod(gap_terrain)

@dataclass(kw_only=True)
class BalanceBeamCfg(_MeshTerrainCfg):
    """Configuration for a custom balance beam terrain."""
    gap_size: float = 2.0
    gap_x: float = 0.0
    beam_width: tuple[float, float] = (0.5, 0.17)
    platform_height: float = 2.5
    max_rpy: float = 0.087
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    _generate = staticmethod(balance_beam_terrain)

@dataclass(kw_only=True)
class StonesCfg(_MeshTerrainCfg):
    """Configuration for a custom stones terrain."""
    min_box_size: float = 0.35
    max_box_size: float = 0.85
    min_box_height: float = 0.01
    max_box_height: float = 0.06
    min_gap_size: float = 0.04
    max_gap_size: float = 0.12
    platform_size: float = 1.4
    boundary_size: float = 2.0
    higher_flat: bool = True
    flat_rise: tuple[float, float] = (-0.05, 0.15)
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    sink_ratio: float = 0.06
    rise_ratio: float = 0.04
    _generate = staticmethod(stones_terrain)

@dataclass(kw_only=True)
class PalletZcCfg(_MeshTerrainCfg):
    """Configuration for a custom pallets terrain."""
    platform_size: float = 2.0
    border_size: float = 1.5
    step_height: tuple[float, float] = (0.0, 0.3)
    step_width: tuple[float, float] = (0.4, 0.16)
    gap_size: tuple[float, float] = (0.08, 0.35)
    floor_ratio: float = 0.5
    floor_height: tuple[float, float] = (-1.5, -0.65)
    target_range_x: tuple[float, float] = (4.0, 4.01)
    target_range_y: tuple[float, float] = (0.0, 0.01)
    _generate = staticmethod(pallets_up_down_zc)
