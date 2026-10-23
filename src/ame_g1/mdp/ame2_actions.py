# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AME-2 action delay, measured in physics substeps rather than policy steps.

Adapted from leggedrobotics/ame2_minimal, commit 8beb9a6, tasks/mdp/actions.py.
Reset histories use the neutral position target, avoiding stale targets from the
previous episode (the upstream implementation retains processed actions on reset).
"""

from dataclasses import dataclass, field

import torch
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg


@dataclass(kw_only=True)
class Ame2DelayedJointPositionActionCfg(JointPositionActionCfg):
  scale: float = 0.25
  use_default_offset: bool = True
  clip: dict[str, tuple] | None = field(default_factory=lambda: {".*": (-100.0, 100.0)})
  delay_range: tuple[int, int] = (0, 1)

  def __post_init__(self):
    super().__post_init__()
    if not 0 <= self.delay_range[0] <= self.delay_range[1]:
      raise ValueError("delay_range must contain ordered, nonnegative physics-step delays.")

  def build(self, env):
    return Ame2DelayedJointPositionAction(self, env)


class Ame2DelayedJointPositionAction(JointPositionAction):
  """Sample one delay per environment on reset and apply it at each physics step."""

  cfg: Ame2DelayedJointPositionActionCfg

  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.max_delay = cfg.delay_range[1]
    self.action_history = torch.zeros(
      self.num_envs, self.max_delay + 1, self.action_dim, device=self.device,
    )
    self.delay_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self._env_indices = torch.arange(self.num_envs, device=self.device)
    self.reset()

  def reset(self, env_ids=None):
    indices = self._env_indices if env_ids is None else self._env_indices[env_ids]
    super().reset(indices)
    self.delay_steps[indices] = torch.randint(
      self.cfg.delay_range[0], self.cfg.delay_range[1] + 1,
      (len(indices),), device=self.device,
    )
    offset = self._offset[indices] if isinstance(self._offset, torch.Tensor) else self._offset
    self._processed_actions[indices] = offset
    self.action_history[indices] = self._processed_actions[indices].unsqueeze(1)

  def apply_actions(self):
    # mjlab calls this once per simulation substep. Advancing in process_actions
    # would multiply the intended latency by the control decimation.
    if self.max_delay:
      self.action_history[:, 1:].copy_(self.action_history[:, :-1].clone())
    self.action_history[:, 0].copy_(self._processed_actions)
    delayed = self.action_history[self._env_indices, self.delay_steps]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    self._entity.set_joint_position_target(delayed - encoder_bias, joint_ids=self._target_ids)
