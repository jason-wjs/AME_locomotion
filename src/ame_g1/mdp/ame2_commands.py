# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""AME-2 goal commands, adapted from leggedrobotics/ame2_minimal for mjlab."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat


@dataclass(kw_only=True)
class SafeTerrainBasedPose2dCommandCfg(CommandTermCfg):
  entity_name: str = "robot"
  patch_name: str = "target"
  heading_range: tuple[float, float] = (-math.pi / 3, math.pi / 3)
  simple_heading: bool = False
  cone_angle_deg: float = 45.0

  def build(self, env):
    return SafeTerrainBasedPose2dCommand(self, env)


class SafeTerrainBasedPose2dCommand(CommandTerm):
  """Sample terrain goals within the official forward cone around each tile origin."""

  cfg: SafeTerrainBasedPose2dCommandCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.robot = env.scene[cfg.entity_name]
    self.terrain = env.scene.terrain
    self.valid_targets = self.terrain.flat_patches[cfg.patch_name]
    self.pos_command_w = torch.zeros(self.num_envs, 3, device=self.device)
    self.heading_command_w = torch.zeros(self.num_envs, device=self.device)
    self._command = torch.zeros(self.num_envs, 4, device=self.device)
    self.metrics["error_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

  @property
  def command(self):
    return self._command

  def _resample_command(self, env_ids):
    patches = self.valid_targets[
      self.terrain.terrain_levels[env_ids], self.terrain.terrain_types[env_ids]
    ]
    relative = patches[..., :2] - self.terrain.env_origins[env_ids, None, :2]
    in_cone = relative[..., 1].abs() <= (
      relative[..., 0] * math.tan(math.radians(self.cfg.cone_angle_deg)) + 0.1
    )
    # Uniformly sample any patch only when none lies within the cone, as upstream.
    weights = torch.where(in_cone.any(dim=1, keepdim=True), in_cone, True).float()
    indices = torch.multinomial(weights, 1).squeeze(1)
    self.pos_command_w[env_ids] = patches[torch.arange(len(env_ids), device=self.device), indices]
    self.pos_command_w[env_ids, 2] += self.robot.data.default_root_state[env_ids, 2]
    if self.cfg.simple_heading:
      direction = self.pos_command_w[env_ids] - self.robot.data.root_link_pos_w[env_ids]
      heading = torch.atan2(direction[:, 1], direction[:, 0])
      flipped = wrap_to_pi(heading + math.pi)
      current = self.robot.data.heading_w[env_ids]
      self.heading_command_w[env_ids] = torch.where(
        wrap_to_pi(heading - current).abs() < wrap_to_pi(flipped - current).abs(),
        heading, flipped,
      )
    else:
      self.heading_command_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(
        *self.cfg.heading_range
      )

  def _update_command(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self._command[ids, :3] = quat_apply_inverse(
      yaw_quat(self.robot.data.root_link_quat_w[ids]),
      self.pos_command_w[ids] - self.robot.data.root_link_pos_w[ids],
    )
    self._command[ids, 3] = wrap_to_pi(
      self.heading_command_w[ids] - self.robot.data.heading_w[ids]
    )

  def _update_metrics(self):
    self.metrics["error_pos"][:] = torch.norm(
      self.pos_command_w[:, :2] - self.robot.data.root_link_pos_w[:, :2], dim=1
    )
    self.metrics["error_heading"][:] = wrap_to_pi(
      self.heading_command_w - self.robot.data.heading_w
    ).abs()
