"""AME-specific reward helpers."""

from __future__ import annotations

import torch

from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _command_norm(command: torch.Tensor) -> torch.Tensor:
  return torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])


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


def track_lin_vel_xy_yaw_frame_exp(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward commanded XY velocity tracking in the robot yaw frame."""

  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  actual_xy = quat_apply_inverse(
    yaw_quat(asset.data.root_link_quat_w), asset.data.root_link_lin_vel_w
  )[:, :2]
  xy_error = torch.sum(torch.square(command[:, :2] - actual_xy), dim=1)
  return torch.exp(-xy_error / std**2)


def track_ang_vel_z_world_exp(
  env,
  std: float,
  command_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward commanded yaw-rate tracking in world frame."""

  asset = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  z_error = torch.square(command[:, 2] - asset.data.root_link_ang_vel_w[:, 2])
  return torch.exp(-z_error / std**2)


def ang_vel_xy_l2(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize roll/pitch angular velocity on the root link."""

  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def undesired_contacts(
  env,
  sensor_name: str,
  threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize non-foot contacts whose force exceeds threshold."""

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force_history is not None
  force_norm = torch.norm(sensor.data.force_history, dim=-1).amax(dim=-1)
  return torch.sum((force_norm > threshold).float(), dim=1)


def applied_torque_limits(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize actuator efforts that exceed configured force limits."""

  asset = env.scene[asset_cfg.name]
  applied = torch.abs(asset.data.actuator_force)
  force_range = env.sim.model.actuator_forcerange
  if force_range.ndim == 2:
    max_limit = torch.maximum(torch.abs(force_range[:, 0]), torch.abs(force_range[:, 1]))
    max_limit = max_limit.unsqueeze(0)
  elif force_range.ndim == 3:
    max_limit = torch.maximum(torch.abs(force_range[..., 0]), torch.abs(force_range[..., 1]))
  else:
    raise ValueError(f"Unexpected actuator_forcerange ndim: {force_range.ndim}")

  num_ctrl = min(applied.shape[1], max_limit.shape[1])
  applied = applied[:, :num_ctrl]
  max_limit = max_limit[:, :num_ctrl]
  safe_limit = torch.clamp(max_limit, min=1e-6)
  excess = torch.clamp(applied - safe_limit, min=0.0)
  return torch.sum(excess / safe_limit, dim=1)


def feet_air_time_positive_biped(
  env,
  sensor_name: str,
  command_name: str,
  threshold: float = 0.6,
) -> torch.Tensor:
  """Reward single-foot support during translation, capped by the threshold."""

  sensor: ContactSensor = env.scene[sensor_name]
  air_time = sensor.data.current_air_time
  contact_time = sensor.data.current_contact_time
  assert air_time is not None
  assert contact_time is not None
  in_contact = contact_time > 0.0
  mode_time = torch.where(in_contact, contact_time, air_time)
  single_stance = torch.sum(in_contact, dim=1) == 1
  reward = torch.where(single_stance.unsqueeze(1), mode_time, 0.0).amin(dim=1)
  reward = torch.clamp(reward, max=threshold)
  command = env.command_manager.get_command(command_name)
  assert command is not None
  return reward * (torch.linalg.vector_norm(command[:, :2], dim=1) > 0.1).float()


def air_time_variance_penalty(
  env,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize mismatched left/right swing and stance timing."""

  sensor: ContactSensor = env.scene[sensor_name]
  last_air_time = sensor.data.last_air_time
  last_contact_time = sensor.data.last_contact_time
  assert last_air_time is not None
  assert last_contact_time is not None
  return torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
    torch.clip(last_contact_time, max=0.5),
    dim=1,
  )


def feet_stumble(
  env,
  sensor_name: str,
) -> torch.Tensor:
  """Penalize feet colliding with steep obstacles."""

  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  forces_z = torch.abs(sensor.data.force[:, :, 2])
  forces_xy = torch.linalg.norm(sensor.data.force[:, :, :2], dim=2)
  return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


def feet_too_near(
  env,
  threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize feet distance that is too narrow."""

  asset = env.scene[asset_cfg.name]
  feet_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  distance = torch.norm(feet_pos[:, 0] - feet_pos[:, 1], dim=-1)
  return torch.clamp(threshold - distance, min=0.0)


def joint_coordination_rel(
  env,
  asset_cfg: SceneEntityCfg,
  coord_joints: list[list[str]],
  coord_signs: list[list[float]] | None = None,
) -> torch.Tensor:
  """Penalize mismatch between coordinated cross-body joint pairs."""

  asset = env.scene[asset_cfg.name]
  if not hasattr(env, "_ame_joint_coord_joints_cache") or env._ame_joint_coord_joints_cache is None:
    cache = []
    for joint_pair in coord_joints:
      joint_ids = []
      for joint_name in joint_pair:
        ids, _ = asset.find_joints(joint_name)
        if not ids:
          raise ValueError(f"Joint pattern '{joint_name}' did not match any joints.")
        joint_ids.append(ids[0])
      cache.append((joint_ids[0], joint_ids[1]))
    env._ame_joint_coord_joints_cache = cache

  if coord_signs is None:
    coord_signs = [[1.0, 1.0]] * len(coord_joints)

  default_joint_pos = asset.data.default_joint_pos
  joint_pos = asset.data.joint_pos
  reward = torch.zeros(joint_pos.shape[0], device=joint_pos.device)
  for i, (joint_1_id, joint_2_id) in enumerate(env._ame_joint_coord_joints_cache):
    joint_1_rel = joint_pos[:, joint_1_id] - default_joint_pos[:, joint_1_id]
    joint_2_rel = joint_pos[:, joint_2_id] - default_joint_pos[:, joint_2_id]
    signed_joint_1 = joint_1_rel * coord_signs[i][0]
    signed_joint_2 = joint_2_rel * coord_signs[i][1]
    reward += torch.square(signed_joint_1 - signed_joint_2)
  if env._ame_joint_coord_joints_cache:
    reward /= len(env._ame_joint_coord_joints_cache)
  return reward


def joint_deviation_l1(
  env,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize absolute deviation from default joint pose."""

  asset = env.scene[asset_cfg.name]
  default_joint_pos = asset.data.default_joint_pos[:, asset_cfg.joint_ids]
  joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
  return torch.sum(torch.abs(joint_pos - default_joint_pos), dim=1)


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
