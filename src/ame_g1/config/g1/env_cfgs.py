"""Environment builders for AME on Unitree G1."""

from __future__ import annotations

from copy import deepcopy
import math

from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, GridPatternCfg, RayCastSensorCfg
from mjlab.tasks.velocity.config.g1.env_cfgs import unitree_g1_rough_env_cfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from ame_g1 import mdp as ame_mdp
from ame_g1.config.variants import DEFAULT_VARIANT, AmeStage, AmeVariant, resolve_task_spec
from ame_g1.terrains import make_stage_terrain_generator

_ROBOT_CFG = SceneEntityCfg("robot")
_TORSO_CFG = SceneEntityCfg("robot", body_names=["torso_link"])
_ANKLES_CFG = SceneEntityCfg(
  "robot",
  body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
)
_ARMS_CFG = SceneEntityCfg(
  "robot",
  joint_names=[
    ".*shoulder.*",
    ".*elbow.*",
    ".*wrist.*",
  ],
)
_BODY_CHAIN_CFG = SceneEntityCfg(
  "robot",
  body_names=[
    "pelvis",
    "torso_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  ],
)
_FEET_CFG = SceneEntityCfg("robot", site_names=["left_foot", "right_foot"])
_STANCE_JOINTS_CFG = SceneEntityCfg(
  "robot",
  joint_names=[
    ".*hip.*",
    ".*knee.*",
    ".*ankle.*",
    ".*waist.*",
  ],
)


def _make_observations(
  stage: AmeStage,
  variant: AmeVariant,
  play: bool,
) -> dict[str, ObservationGroupCfg]:
  stage2_train = stage == "stage2" and not play

  actor_proprio_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
      noise=Unoise(n_min=-0.5, n_max=0.5),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=vel_mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=vel_mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=vel_mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=vel_mdp.last_action),
    "command": ObservationTermCfg(
      func=vel_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  actor_terrain_terms = {
    "terrain_points": ObservationTermCfg(
      func=ame_mdp.terrain_points,
      params={
        "sensor_name": "terrain_scan",
        "height_noise_range": (-0.03, 0.03) if stage2_train else None,
        "apply_drift": stage2_train,
        "clip_height_range": (-1.2, 0.0),
      },
    ),
  }
  critic_proprio_terms = {
    "base_lin_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "base_ang_vel": ObservationTermCfg(
      func=vel_mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=vel_mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=vel_mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=vel_mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=vel_mdp.last_action),
    "command": ObservationTermCfg(
      func=vel_mdp.generated_commands,
      params={"command_name": "twist"},
    ),
  }
  critic_terrain_terms = {
    "terrain_points": ObservationTermCfg(
      func=ame_mdp.terrain_points,
      params={
        "sensor_name": "terrain_scan",
        "clip_height_range": (-1.2, 0.0),
      },
    ),
  }
  return {
    "actor_proprio": ObservationGroupCfg(
      terms=actor_proprio_terms,
      concatenate_terms=True,
      enable_corruption=stage2_train,
    ),
    "actor_terrain": ObservationGroupCfg(
      terms=actor_terrain_terms,
      concatenate_terms=True,
      enable_corruption=stage2_train,
    ),
    "critic_proprio": ObservationGroupCfg(
      terms=critic_proprio_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
    "critic_terrain": ObservationGroupCfg(
      terms=critic_terrain_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }


def _make_rewards(stage: AmeStage):
  rewards = {
    "track_linear_velocity": RewardTermCfg(
      func=ame_mdp.track_body_linear_velocity,
      weight=5.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.25),
        "asset_cfg": deepcopy(_TORSO_CFG),
      },
    ),
    "track_angular_velocity": RewardTermCfg(
      func=ame_mdp.track_body_angular_velocity,
      weight=3.0,
      params={
        "command_name": "twist",
        "std": math.sqrt(0.5),
        "asset_cfg": deepcopy(_TORSO_CFG),
      },
    ),
    "termination": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "action_rate": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-5.0e-3),
    "joint_acc": RewardTermCfg(
      func=vel_mdp.joint_acc_l2,
      weight=-1.0e-6,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "joint_torques": RewardTermCfg(
      func=vel_mdp.joint_torques_l2,
      weight=-2.0e-5,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "joint_pos_limits": RewardTermCfg(
      func=vel_mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "body_linear_velocity": RewardTermCfg(
      func=ame_mdp.body_linear_velocity_penalty,
      weight=-5.0e-2,
      params={"asset_cfg": deepcopy(_TORSO_CFG)},
    ),
    "body_ang_vel": RewardTermCfg(
      func=ame_mdp.body_angular_velocity_penalty_b,
      weight=-5.0e-2,
      params={"asset_cfg": deepcopy(_TORSO_CFG)},
    ),
    "foot_contact_forces": RewardTermCfg(
      func=ame_mdp.foot_contact_force_penalty,
      weight=-2.5e-5,
      params={"sensor_name": "feet_ground_contact", "force_threshold": 700.0},
    ),
    "foot_slip": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-1.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": deepcopy(_FEET_CFG),
      },
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=ame_mdp.joint_deviation_threshold,
      weight=-0.5,
      params={"threshold": 0.25, "asset_cfg": deepcopy(_ARMS_CFG)},
    ),
    "no_fly": RewardTermCfg(
      func=ame_mdp.no_fly_penalty,
      weight=-5.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "straight_body": RewardTermCfg(
      func=ame_mdp.body_projected_gravity_l2,
      weight=-3.0,
      params={"asset_cfg": deepcopy(_BODY_CHAIN_CFG)},
    ),
    "ankle_orientation": RewardTermCfg(
      func=ame_mdp.ankle_orientation_penalty,
      weight=-0.5,
      params={
        "asset_cfg": deepcopy(_ANKLES_CFG),
        "sensor_name": "terrain_scan",
        "elevation_threshold": 0.18,
        "reduced_scale": 0.1,
      },
    ),
  }
  if stage == "stage2":
    rewards["standing_joint_pos"] = RewardTermCfg(
      func=ame_mdp.standing_joint_position_penalty,
      weight=-0.1,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": deepcopy(_STANCE_JOINTS_CFG),
      },
    )
    rewards["standing_joint_vel"] = RewardTermCfg(
      func=ame_mdp.standing_joint_velocity_penalty,
      weight=-0.5,
      params={
        "command_name": "twist",
        "command_threshold": 0.1,
        "asset_cfg": deepcopy(_STANCE_JOINTS_CFG),
      },
    )
  return rewards


def _make_terminations():
  return {
    "time_out": TerminationTermCfg(func=vel_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=envs_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "torso_contact": TerminationTermCfg(
      func=vel_mdp.illegal_contact,
      params={"sensor_name": "torso_ground_contact", "force_threshold": 10.0},
    ),
  }


def _make_stage_events(cfg, stage: AmeStage, play: bool) -> dict[str, EventTermCfg]:
  events = {
    "reset_base": cfg.events["reset_base"],
    "reset_robot_joints": cfg.events["reset_robot_joints"],
  }
  if play:
    events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )
    return events
  if stage == "stage2":
    events["push_robot"] = cfg.events["push_robot"]
    events["foot_friction"] = cfg.events["foot_friction"]
    events["encoder_bias"] = cfg.events["encoder_bias"]
    events["base_com"] = cfg.events["base_com"]
    events["torso_mass"] = EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
        "ranges": (-1.0, 3.0),
        "operation": "add",
      },
    )
    events["torso_inertia"] = EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=["torso_link"]),
        "alpha_range": (-0.08, 0.08),
        "t_range": (-0.01, 0.01),
      },
    )
    events["map_drift"] = EventTermCfg(
      mode="reset",
      func=ame_mdp.resample_map_scan_drift,
      params={"std_xy": 0.02},
    )
  return events


def _ensure_torso_contact_sensor(cfg) -> None:
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  if "torso_ground_contact" in sensor_names:
    return
  torso_contact_cfg = ContactSensorCfg(
    name="torso_ground_contact",
    primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )
  cfg.scene.sensors = (cfg.scene.sensors or ()) + (torso_contact_cfg,)


def _configure_terrain_scan(cfg, variant: AmeVariant) -> None:
  resolution = 0.1 if variant == "paper" else 0.05
  for sensor in cfg.scene.sensors or ():
    if sensor.name != "terrain_scan":
      continue
    assert isinstance(sensor, RayCastSensorCfg)
    sensor.pattern = GridPatternCfg(size=(1.6, 1.0), resolution=resolution)
    sensor.ray_alignment = "yaw"
    return
  raise KeyError("terrain_scan sensor is required for AME terrain observations.")


def unitree_g1_ame_env_cfg(
  stage: AmeStage,
  variant: AmeVariant = DEFAULT_VARIANT,
  play: bool = False,
):
  """Build the AME environment on top of mjlab's Unitree G1 rough task."""

  cfg = deepcopy(unitree_g1_rough_env_cfg(play=play))
  task_spec = resolve_task_spec(variant, stage)
  cfg.seed = 42
  cfg.episode_length_s = 20.0 if not play else int(1e9)
  cfg.scene.num_envs = task_spec.play_num_envs if play else task_spec.train_num_envs
  cfg.scene.terrain.terrain_generator = make_stage_terrain_generator(stage=stage, play=play)  # type: ignore[union-attr]
  cfg.scene.terrain.max_init_terrain_level = 5  # type: ignore[union-attr]
  _configure_terrain_scan(cfg, variant)
  _ensure_torso_contact_sensor(cfg)
  cfg.observations = _make_observations(stage=stage, variant=variant, play=play)
  cfg.rewards = _make_rewards(stage=stage)
  cfg.terminations = _make_terminations()
  cfg.events = _make_stage_events(cfg, stage=stage, play=play)
  if play:
    cfg.curriculum = {}
  else:
    cfg.scene.terrain.terrain_generator.curriculum = True  # type: ignore[union-attr]
  return cfg
