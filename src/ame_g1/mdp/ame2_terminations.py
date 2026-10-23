# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""Teacher termination rules adapted from leggedrobotics/ame2_minimal."""

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse
from .ame2_rewards import _forces, body_acceleration

_ROBOT = SceneEntityCfg("robot")


def ame2_bad_orientation(env, asset_cfg=_ROBOT, start_time=0., limit_gx=.985, limit_gy=.7, limit_gz=0.):
  data = env.scene[asset_cfg.name].data
  if asset_cfg.body_names is not None:
    gravity = quat_apply_inverse(data.body_link_quat_w[:, asset_cfg.body_ids[0]], data.gravity_vec_w)
  else:
    gravity = data.projected_gravity_b
  bad = (gravity[:, 0].abs() > limit_gx) | (gravity[:, 1].abs() > limit_gy) | (gravity[:, 2] > limit_gz)
  return bad & (env.episode_length_buf * env.step_dt >= start_time)


def ame2_illegal_contact(env, threshold, sensor_name, start_time=0., sensor_indices=None):
  forces = _forces(env, sensor_name, sensor_indices).norm(dim=-1).mean(dim=-1)
  return (forces > threshold).any(dim=1) & (env.episode_length_buf * env.step_dt >= start_time)


def ame2_body_lin_acc_out_of_limit(env, threshold, asset_cfg, foot_sensor_name, sensor_threshold=1., start_time=0., sensor_indices=None):
  acceleration = body_acceleration(env, asset_cfg.name)[:, asset_cfg.body_ids, 2].abs()
  contact = _forces(env, foot_sensor_name, sensor_indices).norm(dim=-1).amax(dim=-1) > sensor_threshold
  return ((acceleration > threshold) & contact).any(dim=1) & (env.episode_length_buf * env.step_dt >= start_time)


class ame2_stagnation:
  """Compare root position at fixed windows; reset history only for resetting envs."""

  def __init__(self, cfg, env):
    self.asset_cfg = cfg.params.get("asset_cfg", _ROBOT)
    self.last_pos = env.scene[self.asset_cfg.name].data.root_link_pos_w.clone()
    self.last_time = env.episode_length_buf.float() * env.step_dt

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.last_time[ids] = 0.

  def __call__(self, env, distance_threshold, goal_distance_threshold, time_window, command_name, asset_cfg=_ROBOT):
    position = env.scene[asset_cfg.name].data.root_link_pos_w
    time = env.episode_length_buf * env.step_dt
    fresh = env.episode_length_buf <= 1
    self.last_pos[fresh] = position[fresh]
    self.last_time[fresh] = time[fresh]
    check = time - self.last_time >= time_window
    distance = (position - self.last_pos).norm(dim=1)
    goal_distance = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
    stagnant = check & (distance < distance_threshold) & (goal_distance > goal_distance_threshold)
    self.last_pos[check] = position[check]
    self.last_time[check] = time[check]
    return stagnant
