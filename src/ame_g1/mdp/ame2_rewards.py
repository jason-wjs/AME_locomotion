# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""AME-2 teacher rewards adapted to mjlab state and contact history conventions."""

from __future__ import annotations

import re

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

_ROBOT = SceneEntityCfg("robot")


def _forces(env, sensor_name, sensor_indices=None):
  ids = slice(None) if sensor_indices is None else sensor_indices
  # mjlab: [environment, body, history, xyz]. Sensors must use netforce reduction.
  return env.scene[sensor_name].data.force_history[:, ids]


def _scan_range(env, sensor_name):
  sensor = env.scene[sensor_name]
  heights = sensor.data.hit_pos_w[..., 2]
  valid = (sensor.data.distances >= 0) & torch.isfinite(heights)
  lower = torch.where(valid, heights, float("inf")).amin(dim=1)
  upper = torch.where(valid, heights, -float("inf")).amax(dim=1)
  # Missing geometry is not flat terrain: disable flat-terrain shaping there.
  return torch.where(valid.all(dim=1), upper - lower, float("inf"))


def body_acceleration(env, entity_name="robot"):
  """Position finite differences at the control rate, shared by rewards/termination.

  The first two steps after reset are blanked as in the official sensor. A cache
  keyed by the control step prevents two consumers advancing history twice.
  """
  key = f"_ame2_body_acceleration_{entity_name}"
  position = env.scene[entity_name].data.body_link_pos_w
  if not hasattr(env, key):
    setattr(env, key, {
      "step": env.common_step_counter, "pos": position.clone(),
      "vel": torch.zeros_like(position), "acc": torch.zeros_like(position),
    })
  state = getattr(env, key)
  if state["step"] != env.common_step_counter:
    velocity = (position - state["pos"]) / env.step_dt
    acceleration = (velocity - state["vel"]) / env.step_dt
    state["acc"] = torch.where(
      (env.episode_length_buf <= 2)[:, None, None], 0.0, acceleration
    )
    state["pos"].copy_(position)
    state["vel"].copy_(velocity)
    state["step"] = env.common_step_counter
  return state["acc"]


def ame2_is_terminated_term(env, term_keys="early_.*"):
  value = torch.zeros(env.num_envs, device=env.device)
  for name in env.termination_manager.active_terms:
    if re.fullmatch(term_keys, name):
      value += env.termination_manager.get_term(name)
  return value * (~env.termination_manager.time_outs)


def ame2_move2goal(env, command_name, v_min=0.3, v_max=2.0, cos_thr=0.5, dist_thr=0.5, asset_cfg=_ROBOT):
  goal = env.command_manager.get_command(command_name)[:, :2]
  asset = env.scene[asset_cfg.name]
  if asset_cfg.body_names is not None:
    velocity = quat_apply_inverse(
      asset.data.root_link_quat_w,
      asset.data.body_com_lin_vel_w[:, asset_cfg.body_ids[0]],
    )[:, :2]
  else:
    velocity = asset.data.root_com_lin_vel_b[:, :2]
  speed, distance = velocity.norm(dim=1), goal.norm(dim=1)
  cosine = (velocity * goal).sum(dim=1) / (speed * distance + 1e-6)
  return ((distance < dist_thr) | ((cosine > cos_thr) & (speed >= v_min) & (speed <= v_max))).float()


def ame2_position_tracking(env, T, command_name, std=2.0):
  distance = env.command_manager.get_command(command_name)[:, :2].norm(dim=1)
  remaining = (env.max_episode_length - env.episode_length_buf) * env.step_dt
  return (1.0 / (1.0 + (distance / std).square())) * (remaining < T).float() / T


def ame2_heading_tracking(env, T, command_name, dist_thr=0.5):
  command = env.command_manager.get_command(command_name)
  remaining = (env.max_episode_length - env.episode_length_buf) * env.step_dt
  return (1.0 / (1.0 + command[:, -1].square())) * (remaining < T).float() / T * (command[:, :2].norm(dim=1) < dist_thr)


def ame2_standatgoal(env, command_name, sensor_name, asset_cfg=_ROBOT, dist_thr=0.5, yaw_thr=0.5, sensor_threshold=1.0, sensor_indices=None):
  command = env.command_manager.get_command(command_name)
  distance = command[:, :2].norm(dim=1)
  at_goal = (distance < dist_thr) & (command[:, -1].abs() < yaw_thr)
  forces = _forces(env, sensor_name, sensor_indices).norm(dim=-1).amax(dim=-1)
  missing_feet = (forces <= sensor_threshold).float().mean(dim=1)
  asset = env.scene[asset_cfg.name]
  tilt = 1 - asset.data.projected_gravity_b[:, 2].square()
  posture = (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]).abs().mean(dim=1)
  return at_goal.float() * torch.exp(-(missing_feet + tilt + posture + distance) / 4)


def ame2_base_roll_rate_l2(env, asset_cfg=_ROBOT):
  data = env.scene[asset_cfg.name].data
  if asset_cfg.body_names is not None:
    body = asset_cfg.body_ids[0]
    velocity = quat_apply_inverse(data.body_link_quat_w[:, body], data.body_link_ang_vel_w[:, body])
  else:
    velocity = data.root_link_ang_vel_b
  return velocity[:, 0].square()


def ame2_joint_regularization_l2(env, asset_cfg=_ROBOT):
  data = env.scene[asset_cfg.name].data
  ids = asset_cfg.joint_ids
  return (data.joint_vel[:, ids].square() + .01 * data.qfrc_actuator[:, ids].square() + .001 * data.joint_acc[:, ids].square()).sum(dim=1)


def ame2_action_rate_l2(env):
  return (env.action_manager.action - env.action_manager.prev_action).square().sum(dim=1)


def ame2_link_contact_forces_l2(env, sensor_name, robot_weight, sensor_indices=None):
  forces = _forces(env, sensor_name, sensor_indices).norm(dim=-1).mean(dim=-1).clamp(max=2 * robot_weight)
  return (forces - robot_weight).clamp(min=0).square().sum(dim=1)


def ame2_body_lin_acc_l1(env, asset_cfg=_ROBOT, mass_threshold=0.1):
  asset = env.scene[asset_cfg.name]
  masses = env.sim.get_default_field("body_mass")[asset.data.indexing.body_ids][asset_cfg.body_ids]
  accelerations = body_acceleration(env, asset_cfg.name)[:, asset_cfg.body_ids]
  return (accelerations.norm(dim=-1) * (masses >= mass_threshold)).sum(dim=1)


def ame2_undesired_spinning(env, asset_cfg=_ROBOT, yaw_rate_thr=2.0):
  data = env.scene[asset_cfg.name].data
  yaw = data.body_link_ang_vel_w[:, asset_cfg.body_ids[0], 2] if asset_cfg.body_names is not None else data.root_link_ang_vel_b[:, 2]
  return (yaw.abs() > yaw_rate_thr).float()


def ame2_undesired_leaping(env, sensor_name, height_scanner_name, elevation_thr=0.3, sensor_threshold=1.0, start_time=0.0, sensor_indices=None):
  contacts = _forces(env, sensor_name, sensor_indices).norm(dim=-1) > sensor_threshold
  leaped = (~contacts.any(dim=1)).any(dim=1)
  return (leaped & (_scan_range(env, height_scanner_name) < elevation_thr) & (env.episode_length_buf * env.step_dt >= start_time)).float()


def ame2_undesired_non_foot_contacts(env, sensor_name, sensor_threshold=1.0, sensor_indices=None):
  forces = _forces(env, sensor_name, sensor_indices).norm(dim=-1).amax(dim=-1)
  count = (forces > sensor_threshold).float().sum(dim=1)
  sensor = env.scene[sensor_name]
  if sensor.cfg.track_air_time:
    ids = slice(None) if sensor_indices is None else sensor_indices
    count += sensor.compute_first_contact(env.step_dt)[:, ids].float().sum(dim=1)
  return count


def ame2_undesired_stumbling(env, sensor_name, sensor_indices=None):
  forces = _forces(env, sensor_name, sensor_indices)
  horizontal = forces[..., :2].norm(dim=-1).amax(dim=-1)
  # MuJoCo reports the force exerted by the primary body; Isaac Lab reports
  # the opposite reaction on that body. Ground support must have positive Z.
  vertical = (-forces[..., 2]).amax(dim=-1)
  return (horizontal > 1.1 * vertical + 1).float().sum(dim=1)


def ame2_undesired_slippage(env, sensor_name, asset_cfg, slip_vel_thr=.25, sensor_threshold=1., contact_time_thr=.025, sensor_indices=None):
  """Body selection must match the sensor's explicitly configured channel order."""
  sensor = env.scene[sensor_name]
  ids = slice(None) if sensor_indices is None else sensor_indices
  del sensor_threshold  # Upstream uses sustained contact time rather than force here.
  speed = env.scene[asset_cfg.name].data.body_com_lin_vel_w[:, asset_cfg.body_ids, :2].norm(dim=-1)
  contact_time = sensor.data.current_contact_time[:, ids]
  return ((contact_time > contact_time_thr) & (speed > slip_vel_thr)).float().sum(dim=1)


def ame2_undesired_self_collision(env, sensor_name, sensor_threshold=1.0):
  """Use a sensor with secondary restricted to the other robot collision bodies."""
  return (_forces(env, sensor_name).norm(dim=-1).amax(dim=-1) > sensor_threshold).float().sum(dim=1)


def ame2_undesired_torso_collision(env, sensor_name, sensor_threshold=1.0):
  """One torso indicator after summing its arm-contact force norms, as upstream.

  Each sensor primary is an arm link, with the torso as the sole secondary.
  This retains separate pair forces instead of cancelling them in a torso net force.
  """
  pair_forces = _forces(env, sensor_name).norm(dim=-1)
  return (pair_forces.sum(dim=1).amax(dim=-1) > sensor_threshold).float()


def ame2_joint_pos_limits(env, asset_cfg=_ROBOT):
  data = env.scene[asset_cfg.name].data
  ids = asset_cfg.joint_ids
  limits = data.joint_pos_limits[..., ids, :]
  q = data.joint_pos[:, ids]
  return ((q - .95 * limits[..., 1]).clamp(min=0) + (.95 * limits[..., 0] - q).clamp(min=0)).sum(dim=1)


def ame2_joint_vel_limits(env, velocity_limits, asset_cfg=_ROBOT):
  """Explicit physical limits in selected joint order; MuJoCo has no velocity-limit field."""
  velocity = env.scene[asset_cfg.name].data.joint_vel[:, asset_cfg.joint_ids]
  limits = torch.as_tensor(velocity_limits, device=velocity.device)
  return (velocity.abs() - .9 * limits).clamp(min=0).sum(dim=1)


def ame2_joint_torque_limits(env, asset_cfg=_ROBOT):
  asset = env.scene[asset_cfg.name]
  # G1 uses one direct drive per actuated joint, with unit gear ratio.
  limits = env.sim.model.actuator_forcerange[..., asset.data.indexing.ctrl_ids, :].abs().amax(dim=-1)
  return (asset.data.actuator_force.abs() - .8 * limits).clamp(min=0).sum(dim=1)


def ame2_optional_uppershaping(env, asset_cfg, height_scanner_name, elevation_thr=.3):
  data = env.scene[asset_cfg.name].data
  difference = data.joint_pos[:, asset_cfg.joint_ids] - data.default_joint_pos[:, asset_cfg.joint_ids]
  scale = torch.where(_scan_range(env, height_scanner_name) < elevation_thr, 1., .1)
  return difference.square().sum(dim=1) * scale


def ame2_optional_foot_yaw(env, asset_cfg, torso_cfg, lin_vel_thr=.3):
  data = env.scene[asset_cfg.name].data
  torso = yaw_quat(data.body_link_quat_w[:, torso_cfg.body_ids[0]])
  feet = yaw_quat(data.body_link_quat_w[:, asset_cfg.body_ids])
  torso_yaw = 2 * torch.atan2(torso[:, 3], torso[:, 0])
  foot_yaw = 2 * torch.atan2(feet[..., 3], feet[..., 0])
  penalty = wrap_to_pi(foot_yaw - torso_yaw[:, None]).square().sum(dim=1)
  return penalty * (.1 + .9 * (data.root_com_lin_vel_b[:, 0].abs() > lin_vel_thr).float())
