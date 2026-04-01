"""Symmetry augmentation helpers for AME-1 on Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict

PROPRIO_SLICES = {
  "base_lin_vel": slice(0, 3),
  "base_ang_vel": slice(3, 6),
  "projected_gravity": slice(6, 9),
  "joint_pos": slice(9, 38),
  "joint_vel": slice(38, 67),
  "actions": slice(67, 96),
  "command": slice(96, 99),
}

_ORDINARY_VECTOR_SIGNS = torch.tensor([1.0, -1.0, 1.0])
_ANGULAR_VECTOR_SIGNS = torch.tensor([-1.0, 1.0, -1.0])
_COMMAND_SIGNS = torch.tensor([1.0, -1.0, -1.0])


@dataclass(frozen=True)
class G1SymmetrySpec:
  index: torch.Tensor
  signs: torch.Tensor


_G1_MIRROR_NAME = {
  "left_hip_pitch_joint": "right_hip_pitch_joint",
  "left_hip_roll_joint": "right_hip_roll_joint",
  "left_hip_yaw_joint": "right_hip_yaw_joint",
  "left_knee_joint": "right_knee_joint",
  "left_ankle_pitch_joint": "right_ankle_pitch_joint",
  "left_ankle_roll_joint": "right_ankle_roll_joint",
  "right_hip_pitch_joint": "left_hip_pitch_joint",
  "right_hip_roll_joint": "left_hip_roll_joint",
  "right_hip_yaw_joint": "left_hip_yaw_joint",
  "right_knee_joint": "left_knee_joint",
  "right_ankle_pitch_joint": "left_ankle_pitch_joint",
  "right_ankle_roll_joint": "left_ankle_roll_joint",
  "waist_yaw_joint": "waist_yaw_joint",
  "waist_roll_joint": "waist_roll_joint",
  "waist_pitch_joint": "waist_pitch_joint",
  "left_shoulder_pitch_joint": "right_shoulder_pitch_joint",
  "left_shoulder_roll_joint": "right_shoulder_roll_joint",
  "left_shoulder_yaw_joint": "right_shoulder_yaw_joint",
  "left_elbow_joint": "right_elbow_joint",
  "left_wrist_roll_joint": "right_wrist_roll_joint",
  "left_wrist_pitch_joint": "right_wrist_pitch_joint",
  "left_wrist_yaw_joint": "right_wrist_yaw_joint",
  "right_shoulder_pitch_joint": "left_shoulder_pitch_joint",
  "right_shoulder_roll_joint": "left_shoulder_roll_joint",
  "right_shoulder_yaw_joint": "left_shoulder_yaw_joint",
  "right_elbow_joint": "left_elbow_joint",
  "right_wrist_roll_joint": "left_wrist_roll_joint",
  "right_wrist_pitch_joint": "left_wrist_pitch_joint",
  "right_wrist_yaw_joint": "left_wrist_yaw_joint",
}

_G1_MIRROR_SIGN = {
  "left_hip_pitch_joint": 1.0,
  "left_hip_roll_joint": -1.0,
  "left_hip_yaw_joint": -1.0,
  "left_knee_joint": 1.0,
  "left_ankle_pitch_joint": 1.0,
  "left_ankle_roll_joint": -1.0,
  "right_hip_pitch_joint": 1.0,
  "right_hip_roll_joint": -1.0,
  "right_hip_yaw_joint": -1.0,
  "right_knee_joint": 1.0,
  "right_ankle_pitch_joint": 1.0,
  "right_ankle_roll_joint": -1.0,
  "waist_yaw_joint": -1.0,
  "waist_roll_joint": -1.0,
  "waist_pitch_joint": 1.0,
  "left_shoulder_pitch_joint": 1.0,
  "left_shoulder_roll_joint": -1.0,
  "left_shoulder_yaw_joint": -1.0,
  "left_elbow_joint": 1.0,
  "left_wrist_roll_joint": -1.0,
  "left_wrist_pitch_joint": 1.0,
  "left_wrist_yaw_joint": -1.0,
  "right_shoulder_pitch_joint": 1.0,
  "right_shoulder_roll_joint": -1.0,
  "right_shoulder_yaw_joint": -1.0,
  "right_elbow_joint": 1.0,
  "right_wrist_roll_joint": -1.0,
  "right_wrist_pitch_joint": 1.0,
  "right_wrist_yaw_joint": -1.0,
}


def _get_symmetry_spec(env) -> G1SymmetrySpec:
  source_env = getattr(env, "unwrapped", env)
  cached = getattr(source_env, "_ame_g1_symmetry_spec", None)
  if cached is not None:
    return cached

  robot = source_env.scene["robot"]
  joint_names = list(getattr(robot, "joint_names"))
  name_to_index = {name: index for index, name in enumerate(joint_names)}
  index = []
  signs = []
  for name in joint_names:
    if name not in _G1_MIRROR_NAME or name not in _G1_MIRROR_SIGN:
      raise KeyError(f"Missing explicit G1 symmetry mapping for joint '{name}'.")
    mirrored_name = _G1_MIRROR_NAME[name]
    index.append(name_to_index[mirrored_name])
    signs.append(_G1_MIRROR_SIGN[name])
  spec = G1SymmetrySpec(
    index=torch.tensor(index, dtype=torch.long),
    signs=torch.tensor(signs, dtype=torch.float32),
  )
  setattr(source_env, "_ame_g1_symmetry_spec", spec)
  return spec


def _mirror_joint_block(values: torch.Tensor, spec: G1SymmetrySpec) -> torch.Tensor:
  index = spec.index.to(values.device)
  signs = spec.signs.to(values.device)
  return values.index_select(1, index) * signs.unsqueeze(0)


def mirror_g1_proprio(proprio: torch.Tensor, env) -> torch.Tensor:
  """Mirror the flattened AME proprio vector."""

  spec = _get_symmetry_spec(env)
  mirrored = proprio.clone()
  mirrored[:, PROPRIO_SLICES["base_lin_vel"]] *= _ORDINARY_VECTOR_SIGNS.to(proprio.device)
  mirrored[:, PROPRIO_SLICES["base_ang_vel"]] *= _ANGULAR_VECTOR_SIGNS.to(proprio.device)
  mirrored[:, PROPRIO_SLICES["projected_gravity"]] *= _ORDINARY_VECTOR_SIGNS.to(proprio.device)
  for key in ("joint_pos", "joint_vel", "actions"):
    mirrored[:, PROPRIO_SLICES[key]] = _mirror_joint_block(
      proprio[:, PROPRIO_SLICES[key]],
      spec,
    )
  mirrored[:, PROPRIO_SLICES["command"]] *= _COMMAND_SIGNS.to(proprio.device)
  return mirrored


def mirror_g1_actions(actions: torch.Tensor, env) -> torch.Tensor:
  """Mirror joint actions for G1."""

  spec = _get_symmetry_spec(env)
  return _mirror_joint_block(actions, spec)


def mirror_g1_terrain_points(terrain_points: torch.Tensor) -> torch.Tensor:
  """Mirror terrain z-values across the lateral axis while keeping x/y coordinates."""

  mirrored = terrain_points.clone()
  mirrored[..., 2] = terrain_points[..., 2].flip(dims=(2,))
  return mirrored


def _concat_tensors(original: torch.Tensor | None, mirrored: torch.Tensor | None):
  if original is None or mirrored is None:
    return None
  return torch.cat([original, mirrored], dim=0)


def _concat_obs(original: TensorDict | None, mirrored: TensorDict | None):
  if original is None or mirrored is None:
    return None
  tensors = {
    key: torch.cat([original[key], mirrored[key]], dim=0)
    for key in original.keys()
  }
  batch = original.batch_size[0] + mirrored.batch_size[0]
  return TensorDict(tensors, batch_size=[batch], device=original.device)


def augment_g1_symmetry(
  *,
  env,
  obs: TensorDict | None,
  actions: torch.Tensor | None,
):
  """Return the original batch followed by its mirrored counterpart."""

  if obs is None and actions is None:
    raise ValueError("At least one of `obs` or `actions` must be provided.")

  mirrored_obs = None
  if obs is not None:
    mirrored_obs = obs.clone()
    for proprio_key in ("actor_proprio", "critic_proprio"):
      if proprio_key in mirrored_obs.keys():
        mirrored_obs[proprio_key] = mirror_g1_proprio(obs[proprio_key], env)
    for terrain_key in ("actor_terrain", "critic_terrain"):
      if terrain_key in mirrored_obs.keys():
        mirrored_obs[terrain_key] = mirror_g1_terrain_points(obs[terrain_key])

  mirrored_actions = None
  if actions is not None:
    mirrored_actions = mirror_g1_actions(actions, env)

  return _concat_obs(obs, mirrored_obs), _concat_tensors(actions, mirrored_actions)
