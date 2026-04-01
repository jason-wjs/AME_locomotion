"""AME-specific observation helpers."""

from __future__ import annotations

import torch

from mjlab.sensor import GridPatternCfg, RayCastSensor
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat


def terrain_points(
  env,
  sensor_name: str,
  height_noise_range: tuple[float, float] | None = None,
  apply_drift: bool = False,
  drift_attr: str = "_ame_map_scan_drift_xy",
  clip_height_range: tuple[float, float] | None = None,
) -> torch.Tensor:
  """Return a robot-centric terrain map with shape [B, L, W, 3]."""

  sensor: RayCastSensor = env.scene[sensor_name]
  if not isinstance(sensor.cfg.pattern, GridPatternCfg):
    raise TypeError("AME terrain_points currently supports GridPatternCfg only.")

  shape_attr = f"_ame_{sensor_name}_grid_shape"
  offsets_attr = f"_ame_{sensor_name}_grid_offsets"
  if not hasattr(env, shape_attr) or not hasattr(env, offsets_attr):
    pattern = sensor.cfg.pattern
    num_x = int(round(pattern.size[0] / pattern.resolution)) + 1
    num_y = int(round(pattern.size[1] / pattern.resolution)) + 1
    local_offsets, _ = pattern.generate_rays(None, env.device)
    local_offsets = local_offsets.reshape(num_y, num_x, 3).permute(1, 0, 2).contiguous()
    setattr(env, shape_attr, (num_x, num_y))
    setattr(env, offsets_attr, local_offsets)

  num_x, num_y = getattr(env, shape_attr)
  local_offsets = getattr(env, offsets_attr)

  relative_pos_w = sensor.data.hit_pos_w - sensor.data.pos_w.unsqueeze(1)
  sensor_quat = yaw_quat(sensor.data.quat_w)
  batch_size, num_points, _ = relative_pos_w.shape
  local_points = quat_apply_inverse(
    sensor_quat.unsqueeze(1).expand(batch_size, num_points, 4).reshape(-1, 4),
    relative_pos_w.reshape(-1, 3),
  ).reshape(batch_size, num_points, 3)
  local_points = local_points.reshape(env.num_envs, num_y, num_x, 3).permute(0, 2, 1, 3).contiguous()

  miss_mask = sensor.data.distances < 0
  miss_mask = miss_mask.reshape(env.num_envs, num_y, num_x).permute(0, 2, 1).contiguous()
  points = local_points
  points[..., 0:2] = torch.where(
    miss_mask.unsqueeze(-1),
    local_offsets.unsqueeze(0).expand(env.num_envs, -1, -1, -1)[..., 0:2],
    points[..., 0:2],
  )
  points[..., 2] = torch.where(
    miss_mask,
    torch.full_like(points[..., 2], -sensor.cfg.max_distance),
    points[..., 2],
  )
  if height_noise_range is not None:
    points[..., 2] = points[..., 2] + torch.empty_like(points[..., 2]).uniform_(
      height_noise_range[0],
      height_noise_range[1],
    )
  if apply_drift and hasattr(env, drift_attr):
    drift_xy = getattr(env, drift_attr)
    points[..., 0] += drift_xy[:, None, None, 0]
    points[..., 1] += drift_xy[:, None, None, 1]
  if clip_height_range is not None:
    points[..., 2] = torch.clamp(points[..., 2], min=clip_height_range[0], max=clip_height_range[1])
  return points
