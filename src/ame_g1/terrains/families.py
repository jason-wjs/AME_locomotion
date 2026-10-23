"""Terrain families used by AME for Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainGeometry, TerrainOutput
from mjlab.terrains.utils import make_border
from mjlab.utils.color import RGBA

_FLOOR_COLOR: RGBA = (0.18, 0.18, 0.18, 1.0)
_SUPPORT_COLOR: RGBA = (0.48, 0.58, 0.64, 1.0)
_ALT_SUPPORT_COLOR: RGBA = (0.67, 0.72, 0.76, 1.0)


def _add_box(
  body,
  center: tuple[float, float, float],
  size: tuple[float, float, float],
  color: RGBA,
) -> TerrainGeometry:
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_BOX,
    pos=center,
    size=(
      max(size[0] * 0.5, 1e-4),
      max(size[1] * 0.5, 1e-4),
      max(size[2] * 0.5, 1e-4),
    ),
  )
  return TerrainGeometry(geom=geom, color=color)


def _support_geometry(
  body,
  center_xy: tuple[float, float],
  size_xy: tuple[float, float],
  top_height: float,
  floor_depth: float,
  color: RGBA = _SUPPORT_COLOR,
) -> TerrainGeometry:
  full_height = floor_depth + top_height
  center_z = top_height - full_height * 0.5
  return _add_box(
    body,
    (center_xy[0], center_xy[1], center_z),
    (size_xy[0], size_xy[1], full_height),
    color,
  )


def _add_floor(body, size: tuple[float, float], floor_depth: float) -> TerrainGeometry:
  return _add_box(
    body,
    (size[0] * 0.5, size[1] * 0.5, -floor_depth - 0.05),
    (size[0], size[1], 0.1),
    _FLOOR_COLOR,
  )


def _create_pentagon_mesh(
  spec: mujoco.MjSpec,
  name: str,
  radius: float,
  height: float,
  yaw: float,
):
  bottom_z = -height * 0.5
  top_z = height * 0.5
  vertices: list[list[float]] = []
  for z in (bottom_z, top_z):
    for idx in range(5):
      theta = yaw + idx * (2.0 * np.pi / 5.0)
      vertices.append([radius * np.cos(theta), radius * np.sin(theta), z])

  bottom = [0, 1, 2, 3, 4]
  top = [5, 6, 7, 8, 9]
  faces: list[int] = []
  for idx in range(1, 4):
    faces.extend([bottom[0], bottom[idx + 1], bottom[idx]])
    faces.extend([top[0], top[idx], top[idx + 1]])
  for idx in range(5):
    next_idx = (idx + 1) % 5
    faces.extend([bottom[idx], bottom[next_idx], top[next_idx]])
    faces.extend([bottom[idx], top[next_idx], top[idx]])

  return spec.add_mesh(
    name=name,
    uservert=np.asarray(vertices, dtype=np.float64).reshape(-1).tolist(),
    userface=faces,
  )


def _add_pentagon_support(
  body,
  spec: mujoco.MjSpec,
  center_xy: tuple[float, float],
  radius: float,
  top_height: float,
  floor_depth: float,
  name: str,
  yaw: float,
  color: RGBA = _ALT_SUPPORT_COLOR,
) -> TerrainGeometry:
  height = floor_depth + top_height
  mesh = _create_pentagon_mesh(spec, name=name, radius=radius, height=height, yaw=yaw)
  geom = body.add_geom(
    type=mujoco.mjtGeom.mjGEOM_MESH,
    meshname=mesh.name,
    pos=(center_xy[0], center_xy[1], top_height - height * 0.5),
  )
  return TerrainGeometry(geom=geom, color=color)


@dataclass(kw_only=True)
class RadialBeamsTerrainCfg(SubTerrainCfg):
  """Radial supports joining the central platform to a rectangular tile rim."""

  num_beams: int = 14
  beam_width_range: tuple[float, float] = (0.18, 0.12)
  """Beam widths at difficulty zero and one, respectively."""
  beam_height: float = 0.18
  platform_width: float = 1.0
  border_width: float = 0.35
  floor_depth: float = 1.8

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    center = np.asarray(self.size) / 2
    inner = center - self.border_width
    width = float(np.interp(difficulty, (0., 1.), self.beam_width_range))
    radius = min(
      self.platform_width / 2,
      width / (2 * np.sqrt(2) * np.sin(np.pi / self.num_beams)),
    )
    geometries = [_add_floor(body, self.size, self.floor_depth)]
    for geom in make_border(
      body, self.size, tuple(2 * inner), self.beam_height + self.floor_depth,
      (*center, (self.beam_height - self.floor_depth) / 2),
    ):
      geometries.append(TerrainGeometry(geom=geom, color=_SUPPORT_COLOR))
    geometries.append(_support_geometry(
      body, tuple(center), (2 * radius, 2 * radius),
      self.beam_height, self.floor_depth,
    ))

    for angle in np.linspace(0, 2 * np.pi, self.num_beams, endpoint=False):
      direction = np.array([np.cos(angle), np.sin(angle)])
      projection = np.abs(direction)
      lateral = width / 2 * projection[::-1]
      distances = np.divide(
        inner, projection, out=np.full(2, np.inf), where=projection > 0,
      )
      side = int(np.argmin(distances))
      # The entire beam tip reaches the rim; all corners stay inside the tile.
      end = (inner[side] + lateral[side]) / projection[side]
      outer_limit = np.divide(
        center - lateral, projection,
        out=np.full(2, np.inf), where=projection > 0,
      ).min()
      end = min(end, outer_limit)
      geometry = _support_geometry(
        body, tuple(center + (radius + end) / 2 * direction),
        (end - radius, width), self.beam_height, self.floor_depth,
      )
      geometry.geom.quat = (np.cos(angle / 2), 0., 0., np.sin(angle / 2))
      geometries.append(geometry)
    return TerrainOutput(
      origin=np.array([*center, self.beam_height]), geometries=geometries,
    )


@dataclass(kw_only=True)
class StairRunTerrainCfg(SubTerrainCfg):
  step_height_range: tuple[float, float] = (0.04, 0.11)
  step_depth_range: tuple[float, float] = (0.32, 0.24)
  stair_width_range: tuple[float, float] = (0.9, 0.45)
  start_platform_length: float = 1.0
  landing_length: float = 1.0
  floor_depth: float = 1.8

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    del rng
    body = spec.body("terrain")
    geometries = [_add_floor(body, self.size, self.floor_depth)]

    step_height = self.step_height_range[0] + difficulty * (
      self.step_height_range[1] - self.step_height_range[0]
    )
    step_depth = self.step_depth_range[0] - difficulty * (
      self.step_depth_range[0] - self.step_depth_range[1]
    )
    stair_width = self.stair_width_range[0] - difficulty * (
      self.stair_width_range[0] - self.stair_width_range[1]
    )

    center_y = self.size[1] * 0.5
    geometries.append(
      _support_geometry(
        body,
        (self.start_platform_length * 0.5, center_y),
        (self.start_platform_length, stair_width + 0.25),
        0.0,
        self.floor_depth,
      )
    )

    x_cursor = self.start_platform_length
    num_steps = max(int((self.size[0] - self.start_platform_length - self.landing_length) / step_depth), 3)
    for step_idx in range(num_steps):
      top_height = (step_idx + 1) * step_height
      center_x = x_cursor + step_depth * 0.5
      geometries.append(
        _support_geometry(
          body,
          (center_x, center_y),
          (step_depth, stair_width),
          top_height,
          self.floor_depth,
          _SUPPORT_COLOR if step_idx % 2 == 0 else _ALT_SUPPORT_COLOR,
        )
      )
      x_cursor += step_depth

    landing_height = num_steps * step_height
    landing_length = max(self.size[0] - x_cursor, self.landing_length)
    geometries.append(
      _support_geometry(
        body,
        (x_cursor + landing_length * 0.5, center_y),
        (landing_length, stair_width),
        landing_height,
        self.floor_depth,
      )
    )
    origin = np.array([0.6, center_y, 0.0], dtype=np.float64)
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class PalletTerrainCfg(SubTerrainCfg):
  pallet_width_range: tuple[float, float] = (0.7, 0.28)
  pallet_length_range: tuple[float, float] = (0.45, 0.28)
  gap_range: tuple[float, float] = (0.12, 0.42)
  height_range: tuple[float, float] = (0.0, 0.18)
  start_platform_length: float = 1.0
  landing_length: float = 1.0
  floor_depth: float = 1.8

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = [_add_floor(body, self.size, self.floor_depth)]
    center_y = self.size[1] * 0.5
    path_width = self.pallet_width_range[0] - difficulty * (
      self.pallet_width_range[0] - self.pallet_width_range[1]
    )
    geometries.append(
      _support_geometry(
        body,
        (self.start_platform_length * 0.5, center_y),
        (self.start_platform_length, path_width + 0.25),
        0.0,
        self.floor_depth,
      )
    )

    x_cursor = self.start_platform_length
    max_x = self.size[0] - self.landing_length
    idx = 0
    while x_cursor < max_x:
      pallet_length = self.pallet_length_range[0] - difficulty * (
        self.pallet_length_range[0] - self.pallet_length_range[1]
      )
      gap = self.gap_range[0] + difficulty * (self.gap_range[1] - self.gap_range[0])
      top_height = rng.uniform(self.height_range[0], self.height_range[1]) * (0.5 + 0.5 * difficulty)
      center_x = x_cursor + pallet_length * 0.5
      lateral_offset = rng.uniform(-0.08, 0.08) * difficulty
      geometries.append(
        _support_geometry(
          body,
          (center_x, center_y + lateral_offset),
          (pallet_length, path_width),
          top_height,
          self.floor_depth,
          _SUPPORT_COLOR if idx % 2 == 0 else _ALT_SUPPORT_COLOR,
        )
      )
      x_cursor += pallet_length + gap
      idx += 1

    landing_length = self.size[0] - x_cursor
    geometries.append(
      _support_geometry(
        body,
        (x_cursor + landing_length * 0.5, center_y),
        (landing_length, path_width),
        0.0,
        self.floor_depth,
      )
    )
    origin = np.array([0.6, center_y, 0.0], dtype=np.float64)
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class GapTerrainCfg(SubTerrainCfg):
  support_width_range: tuple[float, float] = (0.75, 0.4)
  support_length_range: tuple[float, float] = (0.7, 0.4)
  gap_range: tuple[float, float] = (0.15, 0.48)
  num_gaps_range: tuple[int, int] = (3, 6)
  start_platform_length: float = 1.0
  landing_length: float = 1.0
  floor_depth: float = 1.8

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = [_add_floor(body, self.size, self.floor_depth)]
    center_y = self.size[1] * 0.5
    support_width = self.support_width_range[0] - difficulty * (
      self.support_width_range[0] - self.support_width_range[1]
    )
    support_length = self.support_length_range[0] - difficulty * (
      self.support_length_range[0] - self.support_length_range[1]
    )
    geometries.append(
      _support_geometry(
        body,
        (self.start_platform_length * 0.5, center_y),
        (self.start_platform_length, support_width + 0.25),
        0.0,
        self.floor_depth,
      )
    )

    num_gaps = int(round(self.num_gaps_range[0] + difficulty * (self.num_gaps_range[1] - self.num_gaps_range[0])))
    x_cursor = self.start_platform_length
    for gap_idx in range(num_gaps + 1):
      center_x = x_cursor + support_length * 0.5
      geometries.append(
        _support_geometry(
          body,
          (center_x, center_y),
          (support_length, support_width),
          0.0,
          self.floor_depth,
          _SUPPORT_COLOR if gap_idx % 2 == 0 else _ALT_SUPPORT_COLOR,
        )
      )
      x_cursor += support_length
      if gap_idx < num_gaps:
        x_cursor += self.gap_range[0] + difficulty * (self.gap_range[1] - self.gap_range[0])

    landing_length = max(self.size[0] - x_cursor, self.landing_length)
    geometries.append(
      _support_geometry(
        body,
        (x_cursor + landing_length * 0.5, center_y),
        (landing_length, support_width),
        0.0,
        self.floor_depth,
      )
    )
    origin = np.array([0.6, center_y, 0.0], dtype=np.float64)
    return TerrainOutput(origin=origin, geometries=geometries)


@dataclass(kw_only=True)
class StonePathTerrainCfg(SubTerrainCfg):
  num_columns: int = 3
  stone_length_range: tuple[float, float] = (0.32, 0.18)
  stone_width_range: tuple[float, float] = (0.32, 0.18)
  stone_gap_range: tuple[float, float] = (0.12, 0.4)
  stone_height_range: tuple[float, float] = (0.0, 0.16)
  start_platform_length: float = 1.0
  landing_length: float = 1.0
  floor_depth: float = 1.8
  pentagon_top: bool = False

  def function(
    self,
    difficulty: float,
    spec: mujoco.MjSpec,
    rng: np.random.Generator,
  ) -> TerrainOutput:
    body = spec.body("terrain")
    geometries = [_add_floor(body, self.size, self.floor_depth)]
    center_y = self.size[1] * 0.5
    geometries.append(
      _support_geometry(
        body,
        (self.start_platform_length * 0.5, center_y),
        (self.start_platform_length, 0.9),
        0.0,
        self.floor_depth,
      )
    )

    stone_length = self.stone_length_range[0] - difficulty * (
      self.stone_length_range[0] - self.stone_length_range[1]
    )
    stone_width = self.stone_width_range[0] - difficulty * (
      self.stone_width_range[0] - self.stone_width_range[1]
    )
    x_cursor = self.start_platform_length
    column_span = max(0.0, self.size[1] - 0.8)
    column_positions = np.linspace(
      center_y - column_span * 0.5,
      center_y + column_span * 0.5,
      self.num_columns,
    )
    row_idx = 0
    while x_cursor < self.size[0] - self.landing_length - stone_length:
      for col_idx, col_y in enumerate(column_positions):
        top_height = rng.uniform(*self.stone_height_range) * (0.5 + 0.5 * difficulty)
        center_x = x_cursor + stone_length * 0.5 + rng.uniform(-0.03, 0.03) * difficulty
        center_xy = (center_x, float(col_y + rng.uniform(-0.05, 0.05) * difficulty))
        if self.pentagon_top:
          geometries.append(
            _add_pentagon_support(
              body,
              spec,
              center_xy,
              radius=min(stone_length, stone_width) * 0.5,
              top_height=top_height,
              floor_depth=self.floor_depth,
              name=f"ame_pentagon_{row_idx}_{col_idx}_{rng.integers(int(1e9))}",
              yaw=float(rng.uniform(0.0, np.pi)),
            )
          )
        else:
          geometries.append(
            _support_geometry(
              body,
              center_xy,
              (
                stone_length * (0.9 + 0.2 * rng.uniform()),
                stone_width * (0.9 + 0.2 * rng.uniform()),
              ),
              top_height,
              self.floor_depth,
              _SUPPORT_COLOR if (row_idx + col_idx) % 2 == 0 else _ALT_SUPPORT_COLOR,
            )
          )
      x_cursor += stone_length + self.stone_gap_range[0] + difficulty * (
        self.stone_gap_range[1] - self.stone_gap_range[0]
      )
      row_idx += 1

    landing_length = self.size[0] - x_cursor
    geometries.append(
      _support_geometry(
        body,
        (x_cursor + landing_length * 0.5, center_y),
        (landing_length, 0.9),
        0.0,
        self.floor_depth,
      )
    )
    origin = np.array([0.6, center_y, 0.0], dtype=np.float64)
    return TerrainOutput(origin=origin, geometries=geometries)
