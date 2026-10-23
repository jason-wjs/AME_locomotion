# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Reward-component storage adapted from the pinned AME-2 RSL-RL fork."""

import torch
from rsl_rl.storage import RolloutStorage


class RewardComponentStorage(RolloutStorage):
  """Keep component returns and one summed policy advantage per sample."""

  def __init__(self, num_envs, num_steps, obs, actions_shape, num_components, device="cpu"):
    super().__init__("rl", num_envs, num_steps, obs, actions_shape, device)
    if num_components < 1:
      raise ValueError("The teacher requires at least one reward component.")
    self.num_reward_components = num_components
    shape = (num_steps, num_envs, num_components)
    self.rewards = torch.zeros(shape, device=device)
    self.values = torch.zeros(shape, device=device)
    self.returns = torch.zeros(shape, device=device)

  def add_transition(self, transition):
    # Upstream reshapes rewards to (N, 1); every other buffer operation is unchanged.
    if self.step >= self.num_transitions_per_env:
      raise OverflowError("Rollout buffer overflow; clear it before the next rollout.")
    self.observations[self.step].copy_(transition.observations)
    self.actions[self.step].copy_(transition.actions)
    self.rewards[self.step].copy_(transition.rewards)
    self.dones[self.step].copy_(transition.dones.view(-1, 1))
    self.values[self.step].copy_(transition.values)
    self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
    if self.distribution_params is None:
      self.distribution_params = tuple(
        torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
        for p in transition.distribution_params
      )
    for destination, source in zip(self.distribution_params, transition.distribution_params):
      destination[self.step].copy_(source)
    self.step += 1
