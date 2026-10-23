# Copyright (c) 2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from leggedrobotics/ame2_minimal, commit 8beb9a6:
# rsl_rl/rsl_rl/models/AME2_models.py and modules/ame2_modules.py.
# Modifications: RSL-RL 5.4.2 distribution API and deterministic export wrappers.
"""Official AME-2 teacher architecture, adapted to the installed RSL-RL API."""

from __future__ import annotations

import copy

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import unpad_trajectories


class _MLP512(nn.Module):
  def __init__(self, input_dim: int, output_dim: int) -> None:
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(input_dim, 512), nn.ELU(),
      nn.Linear(512, 512), nn.ELU(), nn.Linear(512, output_dim),
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


@torch.jit.unused
def _attention(mha, query, features):
  # Match the pinned reference's attention backend, including small head dimensions.
  with sdpa_kernel(SDPBackend.MATH):
    return mha(query, features, features, need_weights=False)[0]


class _GlobalPoolMLP(nn.Module):
  def __init__(self) -> None:
    super().__init__()
    self.net = nn.Sequential(nn.Linear(96, 64), nn.ELU(), nn.Linear(64, 64))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.net(x)


class _TeacherNetwork(nn.Module):
  def __init__(self, prop_dim: int, map_channels: int, output_dim: int, num_heads: int) -> None:
    super().__init__()
    self.fc_part = nn.Sequential(nn.Linear(map_channels, 16), nn.ELU())
    self.cnn_part = nn.Sequential(
      nn.Conv2d(map_channels - 2, 8, kernel_size=5, padding=2), nn.ELU(),
      nn.Conv2d(8, 48, kernel_size=5, padding=2), nn.ELU(),
    )
    self.local_proj = nn.Sequential(nn.Linear(64, 96), nn.ELU())
    self.prop_encoder = _MLP512(prop_dim, 128)
    # Keep the reference state-dict layout for direct same-weight comparisons.
    self.global_pool_mlp = _GlobalPoolMLP()
    self.query_mlp = _MLP512(192, 96)
    self.mha = nn.MultiheadAttention(96, num_heads=num_heads, batch_first=True)
    self.output_mlp = _MLP512(288, output_dim)

  def get_latent(self, prop: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    fc = self.fc_part(mapping)
    cnn = self.cnn_part(mapping[..., 2:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    local = self.local_proj(torch.cat([fc, cnn], dim=-1)).flatten(1, 2)
    global_feat = self.global_pool_mlp(local).max(dim=1).values
    prop_feat = self.prop_encoder(prop)
    query = self.query_mlp(torch.cat([prop_feat, global_feat], dim=-1)).unsqueeze(1)
    if torch.jit.is_scripting():
      attention, _ = self.mha(query, local, local, need_weights=False)
    else:
      attention = _attention(self.mha, query, local)
    return torch.cat([attention.squeeze(1), prop_feat, global_feat], dim=-1)

  def forward(self, prop: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    return self.output_mlp(self.get_latent(prop, mapping))


class AME2TeacherModel(MLPModel):
  """CNN/FC terrain tokens, global max pooling and dense attention teacher.

  Inputs are ``teacher_prop`` (B, P) and ``teacher_mapping`` (B, L, W, C).
  The first two map channels are x/y coordinates; remaining channels enter the CNN.
  Observation normalization is absent in the pinned teacher implementation.
  """

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    num_heads: int = 32,
    distribution_cfg: dict | None = None,
  ) -> None:
    nn.Module.__init__(self)
    self.obs_groups = obs_groups[obs_set]
    if set(self.obs_groups) != {"teacher_prop", "teacher_mapping"}:
      raise ValueError("AME-2 teacher requires teacher_prop and teacher_mapping observation groups")
    if obs["teacher_prop"].ndim != 2 or obs["teacher_mapping"].ndim != 4:
      raise ValueError("Expected teacher_prop (B,P) and teacher_mapping (B,L,W,C)")
    if obs["teacher_mapping"].shape[-1] < 3:
      raise ValueError("teacher_mapping requires x/y coordinates and at least one terrain channel")
    self.prop_dim = obs["teacher_prop"].shape[-1]
    self.map_shape = tuple(obs["teacher_mapping"].shape[1:])
    self.obs_normalization = False
    self.network = self._make_network(output_dim, num_heads)
    self.distribution = None
    if distribution_cfg is not None:
      cfg = dict(distribution_cfg)
      if cfg.pop("class_name") != "GaussianDistribution":
        raise ValueError("The reference AME-2 teacher uses a state-independent GaussianDistribution")
      self.distribution = GaussianDistribution(output_dim, **cfg)

  def _make_network(self, output_dim: int, num_heads: int) -> nn.Module:
    return _TeacherNetwork(self.prop_dim, self.map_shape[-1], output_dim, num_heads)

  def forward(
    self, obs: TensorDict, masks: torch.Tensor | None = None,
    hidden_state=None, stochastic_output: bool = False,
  ) -> torch.Tensor:
    if masks is not None:
      obs = unpad_trajectories(obs, masks)
    output = self.network.output_mlp(self.get_latent(obs))
    if self.distribution is not None and stochastic_output:
      self.distribution.update(output)
      return self.distribution.sample()
    return output

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    return self.network.get_latent(obs["teacher_prop"].detach(), obs["teacher_mapping"].detach())

  def as_jit(self) -> nn.Module:
    return _TeacherExport(self)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _TeacherOnnx(self, verbose)


@torch.jit.unused
def _bilinear_crop(mapping: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
  # Match the reference: cuDNN's spatial transformer rejects large PPO batches.
  with torch.backends.cudnn.flags(enabled=False):
    return F.grid_sample(mapping, grid, mode="bilinear", align_corners=True)


class _GazeNetwork(nn.Module):
  """Official two-stage map encoder; hard heights with straight-through gaze gradients."""

  def __init__(self, prop_dim, map_channels, output_dim, num_heads, crop_x, crop_y):
    super().__init__()
    self.crop_x, self.crop_y = crop_x, crop_y
    self.fc_part = nn.Sequential(nn.Linear(map_channels, 16), nn.ELU())
    self.cnn_part = nn.Sequential(
      nn.Conv2d(map_channels - 2, 8, 5, padding=2), nn.ELU(),
      nn.Conv2d(8, 48, 5, padding=2), nn.ELU(),
    )
    self.local_proj = nn.Sequential(nn.Linear(64, 96), nn.ELU())
    self.prop_encoder = _MLP512(prop_dim, 128)
    self.global_cnn = nn.Sequential(
      nn.Conv2d(map_channels, 8, 3), nn.ELU(),
      nn.Conv2d(8, 32, 3, dilation=2), nn.ELU(),
    )
    self.global_proj = nn.Sequential(nn.Linear(32, 64), nn.ELU())
    self.gaze_head = nn.Sequential(nn.Linear(192, 64), nn.ELU(), nn.Linear(64, 2))
    self.query_mlp = _MLP512(192, 96)
    self.mha = nn.MultiheadAttention(96, num_heads=num_heads, batch_first=True)
    self.output_mlp = _MLP512(288, output_dim)

  def crop(self, mapping: torch.Tensor, gaze: torch.Tensor) -> torch.Tensor:
    batch, length, width, channels = mapping.shape
    cx = (self.crop_x - 1) / 2.0 + gaze[:, 0] * float(length - self.crop_x)
    cy = (self.crop_y - 1) / 2.0 + gaze[:, 1] * float(width - self.crop_y)
    ox = (cx - (self.crop_x - 1) / 2.0).round().long().clamp(0, length - self.crop_x)
    oy = (cy - (self.crop_y - 1) / 2.0).round().long().clamp(0, width - self.crop_y)
    u = torch.arange(self.crop_x, device=mapping.device)
    v = torch.arange(self.crop_y, device=mapping.device)
    rows = ox[:, None] + u[None, :]
    cols = oy[:, None] + v[None, :]
    indices = (rows[:, :, None] * width + cols[:, None, :]).reshape(batch, -1)
    hard = mapping.reshape(batch, length * width, channels).gather(
      1, indices[:, :, None].expand(-1, -1, channels),
    ).view(batch, self.crop_x, self.crop_y, channels)
    # Export/inference needs only original heights; train-time backward uses bilinear sampling.
    if torch.jit.is_scripting():
      return hard
    if not torch.is_grad_enabled():
      return hard
    uf, vf = u.to(mapping.dtype), v.to(mapping.dtype)
    px = cx[:, None, None] + (uf - (self.crop_x - 1) / 2.0)[None, :, None]
    py = cy[:, None, None] + (vf - (self.crop_y - 1) / 2.0)[None, None, :]
    grid = torch.stack([
      (2.0 * py / (width - 1) - 1.0).expand(batch, self.crop_x, self.crop_y),
      (2.0 * px / (length - 1) - 1.0).expand(batch, self.crop_x, self.crop_y),
    ], dim=-1)
    soft = _bilinear_crop(mapping.permute(0, 3, 1, 2), grid).permute(0, 2, 3, 1)
    return hard + (soft - soft.detach())

  def encode(self, prop: torch.Tensor, mapping: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    global_features = self.global_cnn(mapping.permute(0, 3, 1, 2))
    global_feat = self.global_proj(global_features.flatten(2).max(dim=2).values)
    prop_feat = self.prop_encoder(prop)
    gaze = self.gaze_head(torch.cat([global_feat, prop_feat], dim=-1)).sigmoid()
    patch = self.crop(mapping, gaze)
    fc = self.fc_part(patch)
    cnn = self.cnn_part(patch[..., 2:].permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
    local = self.local_proj(torch.cat([fc, cnn], dim=-1)).flatten(1, 2)
    query = self.query_mlp(torch.cat([prop_feat, global_feat], dim=-1)).unsqueeze(1)
    if torch.jit.is_scripting():
      attention, _ = self.mha(query, local, local, need_weights=False)
    else:
      attention = _attention(self.mha, query, local)
    return torch.cat([attention.squeeze(1), prop_feat, global_feat], dim=-1), gaze

  def forward(self, prop: torch.Tensor, mapping: torch.Tensor) -> torch.Tensor:
    latent, _ = self.encode(prop, mapping)
    return self.output_mlp(latent)


class AME2GazeTeacherModel(AME2TeacherModel):
  """Gaze teacher using the existing RSL distribution and checkpoint/export interface."""

  def __init__(
    self, obs, obs_groups, obs_set, output_dim, num_heads=32, distribution_cfg=None,
    crop_x=12, crop_y=10, roi_coef=0.01, roi_margin=0.45,
  ):
    self.crop_x, self.crop_y = crop_x, crop_y
    self.roi_coef, self.roi_margin = roi_coef, roi_margin
    super().__init__(obs, obs_groups, obs_set, output_dim, num_heads, distribution_cfg)

  def _make_network(self, output_dim: int, num_heads: int) -> nn.Module:
    length, width, channels = self.map_shape
    if not (0 < self.crop_x <= length and 0 < self.crop_y <= width) or min(length, width) < 7:
      raise ValueError("Gaze crop must fit the map, whose sides must fit the global CNN (>=7 cells)")
    return _GazeNetwork(self.prop_dim, channels, output_dim, num_heads, self.crop_x, self.crop_y)

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    latent, self._last_gaze_r = self.network.encode(
      obs["teacher_prop"].detach(), obs["teacher_mapping"].detach(),
    )
    return latent

  def auxiliary_loss(self) -> torch.Tensor:
    excess = (self._last_gaze_r - 0.5).abs() - self.roi_margin
    return self.roi_coef * excess.clamp(min=0.0).square().mean()


class _MoENetwork(nn.Module):
  def __init__(self, obs_dim: int, output_dim: int, num_experts: int) -> None:
    super().__init__()
    self.experts = nn.ModuleList([_MLP512(obs_dim, output_dim) for _ in range(num_experts)])
    self.router = _MLP512(obs_dim, num_experts)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    weights = self.router(x).softmax(dim=-1)
    outputs = torch.stack([expert(x) for expert in self.experts], dim=1)
    return (weights.unsqueeze(-1) * outputs).sum(dim=1)


class AME2MoECritic(MLPModel):
  """Soft MoE value model with one output per reward group, as in AME-2."""

  def __init__(
    self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str,
    output_dim: int, num_experts: int = 16,
  ) -> None:
    nn.Module.__init__(self)
    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    self.obs_normalization = False
    self.obs_normalizer = nn.Identity()
    self.distribution = None
    self.mlp = _MoENetwork(self.obs_dim, output_dim, num_experts)

  def get_latent(self, obs: TensorDict, masks=None, hidden_state=None) -> torch.Tensor:
    return torch.cat([obs[key].detach() for key in self.obs_groups], dim=-1)


class _TeacherExport(nn.Module):
  def __init__(self, model: AME2TeacherModel) -> None:
    super().__init__()
    self.network = copy.deepcopy(model.network)

  def forward(self, teacher_prop: torch.Tensor, teacher_mapping: torch.Tensor) -> torch.Tensor:
    return self.network(teacher_prop, teacher_mapping)

  @torch.jit.export
  def reset(self) -> None:
    pass


class _TeacherOnnx(_TeacherExport):
  is_recurrent = False

  def __init__(self, model: AME2TeacherModel, verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.prop_dim = model.prop_dim
    self.map_shape = model.map_shape

  def get_dummy_inputs(self) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.zeros(1, self.prop_dim), torch.zeros(1, *self.map_shape)

  @property
  def input_names(self) -> list[str]:
    return ["teacher_prop", "teacher_mapping"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
