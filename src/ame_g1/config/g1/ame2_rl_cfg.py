"""Pinned AME-2 teacher recipe; community AME training defaults stay separate."""
from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import RslRlOnPolicyRunnerCfg


@dataclass
class TeacherActorCfg:
  class_name: str = "ame_g1.learning.ame2_models:AME2TeacherModel"
  num_heads: int = 32
  distribution_cfg: dict = field(default_factory=lambda: {
    "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log",
  })


@dataclass
class GazeTeacherActorCfg(TeacherActorCfg):
  class_name: str = "ame_g1.learning.ame2_models:AME2GazeTeacherModel"
  crop_x: int = 12
  crop_y: int = 10
  roi_coef: float = 0.01
  roi_margin: float = 0.45


@dataclass
class TeacherCriticCfg:
  class_name: str = "ame_g1.learning.ame2_models:AME2MoECritic"
  num_experts: int = 16


@dataclass
class TeacherPpoCfg:
  class_name: str = "ame_g1.learning.ame2_ppo:AME2PPO"
  optimizer: Literal["muon"] = "muon"
  use_mixed_precision: bool = True
  multi_head_critic: bool = True
  rnd_cfg: None = None  # Required by the RSL logger; this recipe does not use RND.
  num_learning_epochs: int = 5
  num_mini_batches: int = 4
  learning_rate: float = .001
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  gamma: float = .99
  lam: float = .95
  entropy_coef: float = .004
  ent_coef_decay: float = .9999
  min_ent_coef: float = .001
  renorm_max_value_loss: float = 100.
  desired_kl: float = .01
  max_grad_norm: float = 1.
  value_loss_coef: float = 1.
  use_clipped_value_loss: bool = True
  clip_param: float = .2
  normalize_advantage_per_mini_batch: bool = False
  symmetry_cfg: dict = field(default_factory=lambda: {
    "use_data_augmentation": True, "use_mirror_loss": False,
    "for_critic_only": True,
    "data_augmentation_func": "ame_g1.mdp.ame2_symmetry:ame2_g1_symmetry_augmentation",
  })


@dataclass
class Ame2TeacherRunnerCfg(RslRlOnPolicyRunnerCfg):
  actor: TeacherActorCfg = field(default_factory=TeacherActorCfg)
  critic: TeacherCriticCfg = field(default_factory=TeacherCriticCfg)
  algorithm: TeacherPpoCfg = field(default_factory=TeacherPpoCfg)
  obs_groups: dict = field(default_factory=lambda: {
    "actor": ("teacher_mapping", "teacher_prop"), "critic": ("critic",),
  })
  num_steps_per_env: int = 24
  max_iterations: int = 5000
  save_interval: int = 100
  experiment_name: str = "ame2_g1_teacher"
  logger: Literal["wandb", "tensorboard"] = "tensorboard"
  upload_model: bool = False
  onnx_export_policy: Literal["never", "final_only", "every_save"] = "final_only"


@dataclass
class Ame2GazeTeacherRunnerCfg(Ame2TeacherRunnerCfg):
  actor: GazeTeacherActorCfg = field(default_factory=GazeTeacherActorCfg)
  experiment_name: str = "ame2_g1_gaze_teacher"
