"""AME actor/critic models."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories


def _select_token_points(
  terrain_points: torch.Tensor,
  target_height: int,
  target_width: int,
) -> torch.Tensor:
  if terrain_points.shape[1] == target_height and terrain_points.shape[2] == target_width:
    return terrain_points

  row_index = torch.linspace(
    0,
    terrain_points.shape[1] - 1,
    target_height,
    device=terrain_points.device,
  ).round().long()
  col_index = torch.linspace(
    0,
    terrain_points.shape[2] - 1,
    target_width,
    device=terrain_points.device,
  ).round().long()
  sampled_rows = terrain_points.index_select(1, row_index)
  return sampled_rows.index_select(2, col_index)


class _AmeTorchModel(nn.Module):
  """TorchScript export wrapper for the AME policy."""

  def __init__(self, model: "AmeBaseModel") -> None:
    super().__init__()
    self.proprio_normalizer = copy.deepcopy(model.proprio_normalizer)
    self.query_proj = copy.deepcopy(model.query_proj)
    self.local_encoder = copy.deepcopy(model.local_encoder)
    self.attention = copy.deepcopy(model.attention)
    self.global_encoder = copy.deepcopy(model.global_encoder)
    self.query_projector = copy.deepcopy(model.query_projector)
    self.mlp = copy.deepcopy(model.mlp)
    self.deterministic_output = (
      model.distribution.as_deterministic_output_module()
      if model.distribution is not None
      else nn.Identity()
    )
    self.map_latent_dim = model.map_latent_dim
    self.terrain_input_mode = model.terrain_input_mode
    self.concat_coords_post_cnn = model.concat_coords_post_cnn
    self.attach_global = model.attach_global

  def forward(
    self,
    proprio: torch.Tensor,
    terrain_points: torch.Tensor | None = None,
  ) -> torch.Tensor:
    if terrain_points is None or terrain_points.numel() == 0:
      proprio = self.proprio_normalizer(proprio)
      return self.deterministic_output(self.mlp(proprio))

    proprio = self.proprio_normalizer(proprio)
    terrain_input = terrain_points[..., 2:3] if self.terrain_input_mode == "z" else terrain_points
    terrain_input = terrain_input.permute(0, 3, 1, 2)
    encoded = self.local_encoder(terrain_input)
    attention_points = _select_token_points(
      terrain_points,
      target_height=int(encoded.shape[2]),
      target_width=int(encoded.shape[3]),
    )
    encoded = encoded.permute(0, 2, 3, 1).reshape(terrain_points.shape[0], -1, encoded.shape[1])
    if self.concat_coords_post_cnn:
      coords = attention_points.reshape(terrain_points.shape[0], -1, 3)
      local_features = torch.cat([coords, encoded], dim=-1)
    else:
      local_features = encoded

    query = self.query_proj(proprio)
    global_feature = None
    if self.attach_global and self.global_encoder is not None and self.query_projector is not None:
      global_feature = self.global_encoder(local_features).amax(dim=1)
      query = self.query_projector(torch.cat([query, global_feature], dim=-1))

    attention_out, _ = self.attention(
      query.unsqueeze(1),
      local_features,
      local_features,
      need_weights=False,
    )
    latent = torch.cat([proprio, attention_out.squeeze(1)], dim=-1)
    if global_feature is not None:
      latent = torch.cat([latent, global_feature], dim=-1)
    return self.deterministic_output(self.mlp(latent))

  @torch.jit.export
  def reset(self) -> None:
    pass


class _AmeOnnxModel(_AmeTorchModel):
  """ONNX export wrapper for the AME policy."""

  def __init__(self, model: "AmeBaseModel", verbose: bool) -> None:
    super().__init__(model)
    self.verbose = verbose
    self.proprio_dim = model.proprio_dim
    self.terrain_shape = model.terrain_shape

  def get_dummy_inputs(self):
    if self.terrain_shape == (0, 0):
      return (torch.zeros(1, self.proprio_dim),)
    return (
      torch.zeros(1, self.proprio_dim),
      torch.zeros(1, *self.terrain_shape, 3),
    )

  @property
  def input_names(self) -> list[str]:
    if self.terrain_shape == (0, 0):
      return ["obs"]
    return ["proprio", "terrain_points"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class AmeBaseModel(nn.Module):
  """Attention-based map encoder used by AME."""

  is_recurrent: bool = False

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict[str, Any] | None = None,
    map_latent_dim: int = 64,
    num_attention_heads: int = 16,
    terrain_key: str = "terrain_points",
    proprio_obs_groups: tuple[str, ...] | list[str] = (),
    terrain_obs_group: str | None = None,
    attention_dump_key: str = "attention_weights",
    terrain_input_mode: str = "z",
    concat_coords_post_cnn: bool = True,
    cnn_downsample: bool = False,
    attach_global: bool = False,
    encoder_variant: str = "paper",
    cnns: dict[str, nn.Module | None] | None = None,
    **_: Any,
  ) -> None:
    super().__init__()

    self.map_latent_dim = map_latent_dim
    self.num_attention_heads = num_attention_heads
    self.terrain_key = terrain_key
    self.proprio_obs_groups = tuple(proprio_obs_groups)
    self.terrain_obs_group = terrain_obs_group
    self.attention_dump_key = attention_dump_key
    self.terrain_input_mode = terrain_input_mode
    self.concat_coords_post_cnn = concat_coords_post_cnn
    self.cnn_downsample = cnn_downsample
    self.attach_global = attach_global
    self.encoder_variant = encoder_variant
    self.nested_obs_mode = False
    self.terrain_obs_key: str | None = None

    self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
    self.obs_key = self.obs_groups[0]

    self.obs_normalization = obs_normalization
    if obs_normalization:
      self.proprio_normalizer = EmpiricalNormalization(self.proprio_dim)
    else:
      self.proprio_normalizer = nn.Identity()

    if distribution_cfg is not None:
      distribution_cfg = dict(distribution_cfg)
      dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore[assignment]
      self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
      mlp_output_dim = self.distribution.input_dim
    else:
      self.distribution = None
      mlp_output_dim = output_dim

    self._local_encoder_output_dim = (
      map_latent_dim - 3 if concat_coords_post_cnn else map_latent_dim
    )
    if self.flat_obs_mode:
      self.query_proj = None
      self.local_encoder = None
      self.attention = None
      self.global_encoder = None
      self.query_projector = None
      self.cnns = None
      self.shared_encoder = False
      mlp_input_dim = self.proprio_dim
    else:
      if cnns is None:
        self.query_proj = nn.Linear(self.proprio_dim, map_latent_dim)
        self.local_encoder = self._build_local_encoder()
        self.attention = nn.MultiheadAttention(
          embed_dim=map_latent_dim,
          num_heads=num_attention_heads,
          batch_first=True,
        )
        if attach_global:
          self.global_encoder = MLP(map_latent_dim, map_latent_dim, (256, 128), activation)
          self.query_projector = nn.Linear(map_latent_dim * 2, map_latent_dim)
        else:
          self.global_encoder = None
          self.query_projector = None
        self.cnns = {
          "query_proj": self.query_proj,
          "local_encoder": self.local_encoder,
          "attention": self.attention,
          "global_encoder": self.global_encoder,
          "query_projector": self.query_projector,
        }
        self.shared_encoder = False
      else:
        shared_cnns: dict[str, nn.Module | None] = {}

        shared_query_proj = cnns.get("query_proj")
        if (
          isinstance(shared_query_proj, nn.Linear)
          and shared_query_proj.in_features == self.proprio_dim
          and shared_query_proj.out_features == map_latent_dim
        ):
          self.query_proj = shared_query_proj
          shared_cnns["query_proj"] = self.query_proj
        else:
          # Actor/Critic proprio dimensions can differ; keep query projection unshared in that case.
          self.query_proj = nn.Linear(self.proprio_dim, map_latent_dim)

        local_encoder = cnns.get("local_encoder")
        attention = cnns.get("attention")
        if local_encoder is None or attention is None:
          raise ValueError("Shared AME encoders must provide `local_encoder` and `attention`.")
        self.local_encoder = local_encoder
        self.attention = attention
        shared_cnns["local_encoder"] = self.local_encoder
        shared_cnns["attention"] = self.attention

        if attach_global:
          shared_global_encoder = cnns.get("global_encoder")
          shared_query_projector = cnns.get("query_projector")
          if shared_global_encoder is not None and shared_query_projector is not None:
            self.global_encoder = shared_global_encoder
            self.query_projector = shared_query_projector
            shared_cnns["global_encoder"] = self.global_encoder
            shared_cnns["query_projector"] = self.query_projector
          else:
            self.global_encoder = MLP(map_latent_dim, map_latent_dim, (256, 128), activation)
            self.query_projector = nn.Linear(map_latent_dim * 2, map_latent_dim)
        else:
          self.global_encoder = None
          self.query_projector = None

        self.cnns = shared_cnns
        self.shared_encoder = True
      mlp_input_dim = self.proprio_dim + map_latent_dim + (map_latent_dim if attach_global else 0)
    self.mlp = MLP(mlp_input_dim, mlp_output_dim, hidden_dims, activation)
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

    self.attention_shape = self.terrain_shape
    self.last_attention_weights: torch.Tensor | None = None
    self.last_attention_points: torch.Tensor | None = None

  def _build_local_encoder(self) -> nn.Sequential:
    input_channels = 1 if self.terrain_input_mode == "z" else 3
    if self.cnn_downsample:
      return nn.Sequential(
        nn.Conv2d(input_channels, 16, kernel_size=5, padding=2, stride=2),
        nn.ELU(),
        nn.Conv2d(16, self._local_encoder_output_dim, kernel_size=3, padding=1),
        nn.ELU(),
      )
    return nn.Sequential(
      nn.Conv2d(input_channels, 16, kernel_size=5, padding=2),
      nn.ELU(),
      nn.Conv2d(16, self._local_encoder_output_dim, kernel_size=5, padding=2),
      nn.ELU(),
    )

  def _encode_terrain(self, terrain_points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    assert self.local_encoder is not None
    terrain_input = terrain_points[..., 2:3] if self.terrain_input_mode == "z" else terrain_points
    terrain_input = terrain_input.permute(0, 3, 1, 2)
    encoded = self.local_encoder(terrain_input)
    attention_points = _select_token_points(
      terrain_points,
      target_height=int(encoded.shape[2]),
      target_width=int(encoded.shape[3]),
    )
    encoded = encoded.permute(0, 2, 3, 1).reshape(
      terrain_points.shape[0],
      -1,
      self._local_encoder_output_dim,
    )
    self.attention_shape = (int(attention_points.shape[1]), int(attention_points.shape[2]))
    if self.concat_coords_post_cnn:
      coords = attention_points.reshape(terrain_points.shape[0], -1, 3)
      return torch.cat([coords, encoded], dim=-1), attention_points
    return encoded, attention_points

  def _latent_from_tensors(
    self,
    proprio: torch.Tensor,
    terrain_points: torch.Tensor,
  ) -> torch.Tensor:
    proprio = self.proprio_normalizer(proprio)
    local_features, attention_points = self._encode_terrain(terrain_points)
    self.last_attention_points = attention_points.detach()

    assert self.query_proj is not None
    assert self.attention is not None

    query = self.query_proj(proprio)
    global_feature = None
    if self.attach_global and self.global_encoder is not None and self.query_projector is not None:
      global_feature = self.global_encoder(local_features).amax(dim=1)
      query = self.query_projector(torch.cat([query, global_feature], dim=-1))

    attention_out, attention_weights = self.attention(
      query.unsqueeze(1),
      local_features,
      local_features,
      need_weights=True,
      average_attn_weights=False,
    )
    self.last_attention_weights = attention_weights.mean(dim=1).squeeze(1).detach()
    latent = torch.cat([proprio, attention_out.squeeze(1)], dim=-1)
    if global_feature is not None:
      latent = torch.cat([latent, global_feature], dim=-1)
    return latent

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
    latent = self.get_latent(obs, masks, hidden_state)
    mlp_output = self.mlp(latent)
    if self.distribution is not None:
      if stochastic_output:
        self.distribution.update(mlp_output)
        return self.distribution.sample()
      return self.distribution.deterministic_output(mlp_output)
    return mlp_output

  def get_latent(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state=None,
  ) -> torch.Tensor:
    if self.flat_obs_mode:
      return self.proprio_normalizer(obs[self.obs_key])

    del masks, hidden_state
    proprio, terrain_points = self._extract_inputs(obs)
    return self._latent_from_tensors(proprio, terrain_points)

  def reset(self, dones: torch.Tensor | None = None, hidden_state=None) -> None:
    del dones, hidden_state

  def get_hidden_state(self):
    return None

  def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
    del dones

  @property
  def output_mean(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.mean

  @property
  def output_std(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.std

  @property
  def output_entropy(self) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.entropy

  @property
  def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
    assert self.distribution is not None
    return self.distribution.params

  def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.log_prob(outputs)

  def get_kl_divergence(
    self,
    old_params: tuple[torch.Tensor, ...],
    new_params: tuple[torch.Tensor, ...],
  ) -> torch.Tensor:
    assert self.distribution is not None
    return self.distribution.kl_divergence(old_params, new_params)

  def update_normalization(self, obs: TensorDict) -> None:
    if self.obs_normalization:
      proprio, _ = self._extract_inputs(obs)
      self.proprio_normalizer.update(proprio)  # type: ignore[operator]

  def as_jit(self) -> nn.Module:
    return _AmeTorchModel(self)

  def as_onnx(self, verbose: bool) -> nn.Module:
    return _AmeOnnxModel(self, verbose)

  def parameters(self, recurse: bool = True):
    if not self.shared_encoder or self.cnns is None:
      yield from super().parameters(recurse=recurse)
      return
    shared_param_ids = {
      id(param)
      for module in self.cnns.values()
      if module is not None
      for param in module.parameters(recurse=recurse)
    }
    for param in super().parameters(recurse=recurse):
      if id(param) not in shared_param_ids:
        yield param

  def export_attention_metadata(
    self,
    path: str,
    filename: str = "attention_metadata.json",
  ) -> None:
    metadata = {
      "proprio_keys": list(self.proprio_keys),
      "proprio_dim": self.proprio_dim,
      "terrain_key": self.terrain_obs_key or self.terrain_key,
      "terrain_shape": list(self.terrain_shape),
      "attention_shape": list(self.attention_shape),
      "map_latent_dim": self.map_latent_dim,
      "num_attention_heads": self.num_attention_heads,
      "encoder_variant": self.encoder_variant,
      "terrain_input_mode": self.terrain_input_mode,
      "concat_coords_post_cnn": self.concat_coords_post_cnn,
      "cnn_downsample": self.cnn_downsample,
      "attach_global": self.attach_global,
    }
    output_path = Path(path)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / filename, "w", encoding="utf-8") as file:
      json.dump(metadata, file, indent=2, sort_keys=True)

  def _get_obs_dim(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
  ) -> tuple[list[str], int]:
    active_obs_groups = obs_groups[obs_set]
    if len(active_obs_groups) == 1:
      nested = obs[active_obs_groups[0]]
      if isinstance(nested, TensorDict):
        self.nested_obs_mode = True
        self.flat_obs_mode = False
        if self.terrain_key not in nested.keys():
          raise ValueError(f"Missing terrain observation `{self.terrain_key}`.")

        self.proprio_keys = tuple(key for key in nested.keys() if key != self.terrain_key)
        if len(self.proprio_keys) == 0:
          raise ValueError("AME observations must contain at least one proprioceptive term.")
        self.proprio_dim = sum(int(nested[key].shape[-1]) for key in self.proprio_keys)

        terrain_points = nested[self.terrain_key]
        if terrain_points.ndim != 4 or terrain_points.shape[-1] != 3:
          raise ValueError(
            "AME terrain points must have shape [batch, height, width, 3], "
            f"got {tuple(terrain_points.shape)}."
          )
        self.terrain_obs_key = self.terrain_key
        self.terrain_shape = (int(terrain_points.shape[1]), int(terrain_points.shape[2]))
        return active_obs_groups, self.proprio_dim + self.map_latent_dim

      if nested.ndim != 2:
        raise ValueError(
          "AME flat fallback only supports [batch, obs_dim] tensors, "
          f"got {tuple(nested.shape)}."
        )
      self.flat_obs_mode = True
      self.proprio_keys = (active_obs_groups[0],)
      self.proprio_dim = int(nested.shape[-1])
      self.terrain_shape = (0, 0)
      return active_obs_groups, self.proprio_dim

    if self.proprio_obs_groups or self.terrain_obs_group is not None:
      proprio_keys = list(self.proprio_obs_groups)
      terrain_keys = [self.terrain_obs_group] if self.terrain_obs_group is not None else []
      expected_groups = set(proprio_keys + terrain_keys)
      unexpected_groups = set(active_obs_groups) - expected_groups
      if unexpected_groups:
        raise ValueError(
          "AME model received unexpected observation groups. "
          f"Expected {sorted(expected_groups)}, got extras {sorted(unexpected_groups)}."
        )
    else:
      proprio_keys = []
      terrain_keys = []
      for obs_group in active_obs_groups:
        value = obs[obs_group]
        if isinstance(value, TensorDict):
          raise ValueError(
            "AME models do not support nested observation groups mixed with top-level groups."
          )
        if value.ndim == 2:
          proprio_keys.append(obs_group)
          continue
        if value.ndim == 4 and value.shape[-1] == 3:
          terrain_keys.append(obs_group)
          continue
        raise ValueError(
          "AME observations must be 2D proprio tensors or 4D terrain tensors, "
          f"got {tuple(value.shape)} for `{obs_group}`."
        )

    proprio_dim = 0
    terrain_shape: tuple[int, int] | None = None
    for obs_group in proprio_keys:
      if obs_group not in active_obs_groups:
        raise ValueError(f"Configured proprio observation group `{obs_group}` is not active.")
      value = obs[obs_group]
      if isinstance(value, TensorDict) or value.ndim != 2:
        raise ValueError(
          f"Configured proprio observation group `{obs_group}` must be a 2D tensor."
        )
      proprio_dim += int(value.shape[-1])
    for obs_group in terrain_keys:
      if obs_group not in active_obs_groups:
        raise ValueError(f"Configured terrain observation group `{obs_group}` is not active.")
      value = obs[obs_group]
      if isinstance(value, TensorDict) or value.ndim != 4 or value.shape[-1] != 3:
        raise ValueError(
          f"Configured terrain observation group `{obs_group}` must have shape [B, H, W, 3]."
        )
      terrain_shape = (int(value.shape[1]), int(value.shape[2]))

    if len(proprio_keys) == 0 or len(terrain_keys) != 1 or terrain_shape is None:
      raise ValueError(
        "AME models require at least one proprio group and exactly one terrain group. "
        f"Got proprio={proprio_keys}, terrain={terrain_keys}."
      )

    self.flat_obs_mode = False
    self.proprio_keys = tuple(proprio_keys)
    self.proprio_dim = proprio_dim
    self.terrain_obs_key = terrain_keys[0]
    self.terrain_shape = terrain_shape
    return active_obs_groups, self.proprio_dim + self.map_latent_dim

  def _extract_inputs(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor]:
    if self.flat_obs_mode:
      obs_tensor = obs[self.obs_key]
      return obs_tensor, torch.zeros(obs_tensor.shape[0], 0, 0, 3, device=obs_tensor.device)
    if self.nested_obs_mode:
      nested = obs[self.obs_key]
      if not isinstance(nested, TensorDict):
        raise ValueError("AME models require nested actor/critic observations.")
      proprio = torch.cat([nested[key] for key in self.proprio_keys], dim=-1)
      terrain_points = nested[self.terrain_key]
      return proprio, terrain_points

    proprio = torch.cat([obs[key] for key in self.proprio_keys], dim=-1)
    if self.terrain_obs_key is None:
      raise ValueError("AME model terrain observation key is not configured.")
    terrain_points = obs[self.terrain_obs_key]
    return proprio, terrain_points


class AmeActorModel(AmeBaseModel):
  """AME actor model."""


class AmeCriticModel(AmeBaseModel):
  """AME critic model."""
