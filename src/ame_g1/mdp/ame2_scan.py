# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""AME-2 scan geometry and per-episode pose drift for mjlab's raycast sensor.

The reference Isaac Lab RayCaster adds drift to the tracked world pose before
ray queries. This adapter shifts mjlab's prepared ray buffer and its matching
postprocessing origin; it never changes the robot pose or copies the ray kernel.
"""

from dataclasses import dataclass, field

import torch
import warp as wp
from mjlab.sensor import GridPatternCfg, RayCastSensor, RayCastSensorCfg


@dataclass
class AME2GridPatternCfg(GridPatternCfg):
  size: tuple[float, float] = (2.0, 1.12)
  resolution: float = 0.08

  def generate_rays(self, mj_model, device):
    offsets, directions = super().generate_rays(mj_model, device)
    offsets = offsets + offsets.new_tensor((0.5, 0.0, 20.0))
    return offsets, directions


@dataclass
class AME2RayCastSensorCfg(RayCastSensorCfg):
  pattern: AME2GridPatternCfg = field(default_factory=AME2GridPatternCfg)
  ray_alignment: str = "yaw"
  max_distance: float = 100.0
  drift_range: tuple[float, float] = (-0.04, 0.04)

  def build(self):
    return AME2RayCastSensor(self)


class AME2RayCastSensor(RayCastSensor):
  cfg: AME2RayCastSensorCfg

  def initialize(self, mj_model, model, data, device):
    super().initialize(mj_model, model, data, device)
    self.drift = torch.zeros(data.nworld, 3, device=device)

  def reset(self, env_ids=None):
    super().reset(env_ids)
    ids = slice(None) if env_ids is None else env_ids
    self.drift[ids] = torch.empty_like(self.drift[ids]).uniform_(*self.cfg.drift_range)

  def prepare_rays(self):
    super().prepare_rays()
    self._cached_world_origins += self.drift[:, None, :]
    self._cached_frame_pos += self.drift[:, None, :]
    wp.to_torch(self._ray_pnt).view_as(self._cached_world_origins).copy_(self._cached_world_origins)
