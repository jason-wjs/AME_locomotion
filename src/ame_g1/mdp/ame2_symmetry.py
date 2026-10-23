# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Critic-only G1 reflection, adapted from leggedrobotics/ame2_minimal.

Resolve left/right pairs from names instead of assuming Isaac Lab's joint order.
The actor observations are intentionally not augmented by this training feature.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from .ame2_observations import contact_state_indices


def _other_side(name):
  if "left_" in name:
    return name.replace("left_", "right_", 1)
  if "right_" in name:
    return name.replace("right_", "left_", 1)
  return name


def _reflection_spec(env):
  if not hasattr(env, "_ame2_reflection_spec"):
    joint_names = list(env.action_manager.get_term("joint_pos").target_names)
    joint_indices = [joint_names.index(_other_side(name)) for name in joint_names]
    signs = [-1. if "_roll_" in name or "_yaw_" in name else 1. for name in joint_names]
    cfg = env.observation_manager.get_term_cfg("critic", "contact_states")
    names = env.scene[cfg.params["sensor_name"]].primary_names
    selection = contact_state_indices(
      env, cfg.params["sensor_name"], cfg.params.get("sensor_indices"),
      cfg.params.get("mass_threshold", .1), cfg.params.get("ignore_bodies", ()),
    )
    names = [names[i] for i in selection]
    contact_indices = [names.index(_other_side(name)) for name in names]
    height_cfg = env.observation_manager.get_term_cfg("critic", "height_scan")
    pattern = env.scene[height_cfg.params["sensor_name"]].cfg.pattern
    nx = int(round(pattern.size[0] / pattern.resolution)) + 1
    ny = int(round(pattern.size[1] / pattern.resolution)) + 1
    env._ame2_reflection_spec = (joint_indices, signs, contact_indices, nx, ny)
  return env._ame2_reflection_spec


def mirror_critic(obs, env):
  joint_indices, signs, contacts, nx, ny = _reflection_spec(env)
  joints = len(joint_indices)
  expected = 9 + 3 * joints + 4 + 1 + len(contacts) + nx * ny
  if obs.shape[-1] != expected:
    raise ValueError(f"AME-2 critic layout expects {expected} channels, got {obs.shape[-1]}.")
  reflected = obs.clone()
  reflected[:, :9] *= obs.new_tensor([1, -1, 1, -1, 1, -1, 1, -1, 1])
  index = 9
  joint_signs = obs.new_tensor(signs)
  for _ in range(3):
    reflected[:, index:index+joints] = obs[:, index:index+joints][:, joint_indices] * joint_signs
    index += joints
  reflected[:, index:index+4] *= obs.new_tensor([1, -1, -1, 1])
  index += 5  # Pose command and unchanged time remaining.
  reflected[:, index:index+len(contacts)] = obs[:, index:index+len(contacts)][:, contacts]
  index += len(contacts)
  reflected[:, index:] = obs[:, index:].reshape(-1, nx, ny).flip(-1).flatten(1)
  return reflected


def ame2_g1_symmetry_augmentation(obs, actions=None, env=None, for_critic=True):
  if not for_critic:
    raise ValueError("AME-2 reflection is enabled only for the critic loss.")
  env = getattr(env, "unwrapped", env)
  if isinstance(obs, TensorDict):
    values = {
      key: torch.cat((value, mirror_critic(value, env) if key == "critic" else value), dim=0)
      for key, value in obs.items()
    }
    augmented = TensorDict(values, batch_size=[2 * obs.batch_size[0]], device=obs.device)
  else:
    augmented = torch.cat((obs, mirror_critic(obs, env)), dim=0)
  augmented_actions = None
  if actions is not None:
    joints, signs, *_ = _reflection_spec(env)
    augmented_actions = torch.cat((actions, actions[:, joints] * actions.new_tensor(signs)), dim=0)
  return augmented, augmented_actions
