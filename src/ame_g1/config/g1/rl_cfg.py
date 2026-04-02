"""RSL-RL runner config for AME on Unitree G1."""

from __future__ import annotations

from dataclasses import dataclass, field

from mjlab.rl import (
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from ame_g1.config.variants import DEFAULT_VARIANT, AmePhase, AmeVariant, resolve_task_spec


@dataclass
class AmeModelCfg(RslRlModelCfg):
  """Custom actor/critic config for the AME attention model."""

  map_latent_dim: int = 64
  num_attention_heads: int = 16
  proprio_key: str = "proprio"
  terrain_key: str = "terrain_points"
  proprio_obs_groups: tuple[str, ...] = ()
  terrain_obs_group: str | None = None
  attention_dump_key: str = "attention_weights"
  terrain_input_mode: str = "z"
  concat_coords_post_cnn: bool = True
  cnn_downsample: bool = False
  attach_global: bool = False
  encoder_variant: str = "paper"


@dataclass
class AmePpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  """PPO config with the symmetry extension enabled for AME-1."""

  symmetry_cfg: dict = field(default_factory=dict)


@dataclass
class AmeRunnerCfg(RslRlOnPolicyRunnerCfg):
  """Runner config specialized for AME."""

  actor: AmeModelCfg = field(
    default_factory=lambda: AmeModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      class_name="ame_g1.learning.actor_critic:AmeActorModel",
    )
  )
  critic: AmeModelCfg = field(
    default_factory=lambda: AmeModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      distribution_cfg=None,
      class_name="ame_g1.learning.actor_critic:AmeCriticModel",
    )
  )


def _variant_model_cfg(
  *,
  variant: AmeVariant,
  use_global_context: bool,
  proprio_obs_groups: tuple[str, ...],
  terrain_obs_group: str,
  distribution_cfg,
  class_name: str,
) -> AmeModelCfg:
  if variant == "paper":
    return AmeModelCfg(
      hidden_dims=(512, 256, 128),
      activation="elu",
      obs_normalization=False,
      proprio_obs_groups=proprio_obs_groups,
      terrain_obs_group=terrain_obs_group,
      distribution_cfg=distribution_cfg,
      class_name=class_name,
      terrain_input_mode="z",
      concat_coords_post_cnn=True,
      cnn_downsample=False,
      attach_global=False,
      encoder_variant="paper",
    )
  return AmeModelCfg(
    hidden_dims=(512, 256, 128),
    activation="elu",
    obs_normalization=False,
    proprio_obs_groups=proprio_obs_groups,
    terrain_obs_group=terrain_obs_group,
    distribution_cfg=distribution_cfg,
    class_name=class_name,
    terrain_input_mode="xyz",
    concat_coords_post_cnn=False,
    cnn_downsample=True,
    attach_global=use_global_context,
    encoder_variant="g1",
  )


def unitree_g1_ame_ppo_runner_cfg(
  phase: AmePhase,
  variant: AmeVariant = DEFAULT_VARIANT,
  use_global_context: bool = False,
) -> AmeRunnerCfg:
  """Create RL config for the AME training phases."""

  if variant == "paper" and use_global_context:
    raise ValueError("`paper` variant does not support global-context encoder mode.")
  entropy_coef = 0.005 if phase == "base" else 0.002
  task_spec = resolve_task_spec(variant, phase)
  return AmeRunnerCfg(
    actor=_variant_model_cfg(
      variant=variant,
      use_global_context=use_global_context,
      proprio_obs_groups=("actor_proprio",),
      terrain_obs_group="actor_terrain",
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "scalar",
      },
      class_name="ame_g1.learning.actor_critic:AmeActorModel",
    ),
    critic=_variant_model_cfg(
      variant=variant,
      use_global_context=use_global_context,
      proprio_obs_groups=("critic_proprio",),
      terrain_obs_group="critic_terrain",
      distribution_cfg=None,
      class_name="ame_g1.learning.actor_critic:AmeCriticModel",
    ),
    algorithm=AmePpoAlgorithmCfg(
      num_learning_epochs=5,
      num_mini_batches=3,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      entropy_coef=entropy_coef,
      desired_kl=0.01,
      max_grad_norm=1.0,
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      normalize_advantage_per_mini_batch=False,
      optimizer="adam",
      share_cnn_encoders=True,
      symmetry_cfg={
        "use_data_augmentation": True,
        "use_mirror_loss": False,
        "mirror_loss_coeff": 0.0,
        "data_augmentation_func": "ame_g1.learning.symmetry:augment_g1_symmetry",
      },
    ),
    experiment_name=task_spec.experiment_name,
    logger="tensorboard",
    num_steps_per_env=24,
    max_iterations=15000 if phase == "base" else 3200,
    obs_groups={
      "actor": ("actor_proprio", "actor_terrain"),
      "critic": ("critic_proprio", "critic_terrain"),
    },
    save_interval=100,
    wandb_project="ame_g1",
  )
