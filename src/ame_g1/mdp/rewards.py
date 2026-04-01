"""AME-specific reward helpers."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _body_velocities_b(
  env,
  asset_cfg: SceneEntityCfg,
) -> tuple[torch.Tensor, torch.Tensor]:
  asset = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  lin_vel_w = asset.data.body_link_lin_vel_w[:, asset_cfg.body_ids, :]
  ang_vel_w = asset.data.body_link_ang_vel_w[:, asset_cfg.body_ids, :]
  shape = body_quat_w.shape
  lin_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    lin_vel_w.reshape(-1, 3),
  ).reshape(shape[0], shape[1], 3)
  ang_vel_b = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    ang_vel_w.reshape(-1, 3),
  ).reshape(shape[0], shape[1], 3)
  return lin_vel_b, ang_vel_b


def track_body_linear_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward commanded linear velocity tracking for a specific body in its local frame."""

  command = env.command_manager.get_command(command_name)
  assert command is not None
  lin_vel_b, _ = _body_velocities_b(env, asset_cfg)
  actual = lin_vel_b.squeeze(1)
  xy_error = torch.sum(torch.square(command[:, :2] - actual[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def track_body_angular_velocity(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward commanded yaw-rate tracking for a specific body in its local frame."""

  command = env.command_manager.get_command(command_name)
  assert command is not None
  _, ang_vel_b = _body_velocities_b(env, asset_cfg)
  actual = ang_vel_b.squeeze(1)
  z_error = torch.square(command[:, 2] - actual[:, 2])
  return torch.exp(-z_error / std**2)


def joint_deviation_threshold(
  env,
  threshold: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint deviations from default posture above a squared-norm threshold."""

  asset = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  deviation = torch.sum(torch.square(joint_pos - default_joint_pos), dim=1)
  return torch.clamp(deviation - threshold, min=0.0)


def no_fly_penalty(env, sensor_name: str) -> torch.Tensor:
  """Penalize losing contact with all feet simultaneously."""

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  num_contacts = torch.sum(sensor.data.found > 0, dim=1)
  return (num_contacts == 0).float()


def body_projected_gravity_l2(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize non-upright torso, pelvis, and foot bodies."""

  asset = env.scene[asset_cfg.name]
  body_quat_w = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  batch_size, num_bodies, _ = body_quat_w.shape
  gravity_w = asset.data.gravity_vec_w[:, None, :].expand(batch_size, num_bodies, 3)
  projected_gravity = quat_apply_inverse(
    body_quat_w.reshape(-1, 4),
    gravity_w.reshape(-1, 3),
  )
  projected_gravity = projected_gravity.reshape(batch_size, num_bodies, 3)
  return torch.sum(torch.square(projected_gravity[..., :2]), dim=(1, 2))


def body_linear_velocity_penalty(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize vertical velocity of specific bodies in their local frames."""

  lin_vel_b, _ = _body_velocities_b(env, asset_cfg)
  return torch.sum(torch.square(lin_vel_b[..., 2]), dim=1)


def body_angular_velocity_penalty_b(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch angular velocities in each body's local frame."""

  _, ang_vel_b = _body_velocities_b(env, asset_cfg)
  return torch.sum(torch.square(ang_vel_b[..., :2]), dim=(1, 2))


def foot_contact_force_penalty(
  env,
  sensor_name: str,
  force_threshold: float = 700.0,
) -> torch.Tensor:
  """Penalize impact forces that exceed the configured threshold."""

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_magnitude = torch.norm(sensor.data.force, dim=-1)
  return torch.sum(torch.clamp(force_magnitude - force_threshold, min=0.0), dim=1)


def ankle_orientation_penalty(
  env,
  asset_cfg: SceneEntityCfg,
  sensor_name: str | None = None,
  elevation_threshold: float = 0.18,
  reduced_scale: float = 0.1,
) -> torch.Tensor:
  """Penalize ankle roll-link tilt, with reduced weight on large elevation changes."""

  penalty = body_projected_gravity_l2(env, asset_cfg)
  if sensor_name is None:
    return penalty
  terrain_sensor: RayCastSensor = env.scene[sensor_name]
  heights = terrain_sensor.data.pos_w[:, 2].unsqueeze(1) - terrain_sensor.data.hit_pos_w[..., 2]
  valid_mask = terrain_sensor.data.distances >= 0
  height_max = torch.where(
    valid_mask,
    heights,
    torch.full_like(heights, float("-inf")),
  ).max(dim=1).values
  height_min = torch.where(
    valid_mask,
    heights,
    torch.full_like(heights, float("inf")),
  ).min(dim=1).values
  has_valid = valid_mask.any(dim=1)
  height_span = torch.where(
    has_valid,
    height_max - height_min,
    torch.zeros_like(height_max),
  )
  scale = torch.where(
    height_span > elevation_threshold,
    torch.full_like(height_span, reduced_scale),
    torch.ones_like(height_span),
  )
  return penalty * scale


def standing_joint_position_penalty(
  env,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize deviation from the default standing posture while nearly stationary."""

  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (command_norm < command_threshold).float()
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_pos - default_joint_pos), dim=1) * standing


def standing_joint_velocity_penalty(
  env,
  command_name: str,
  command_threshold: float = 0.1,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize joint motion while commanded motion is near zero."""

  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  command_norm = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  standing = (command_norm < command_threshold).float()
  joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
  return torch.sum(torch.square(joint_vel), dim=1) * standing
