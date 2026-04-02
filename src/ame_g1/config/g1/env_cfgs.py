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
from ame_g1.config.variants import DEFAULT_VARIANT, AmePhase, AmeVariant, resolve_task_spec
from ame_g1.terrains import make_phase_terrain_generator

_ROBOT_CFG = SceneEntityCfg("robot")
_TORSO_CFG = SceneEntityCfg("robot", body_names=["torso_link"])
_ANKLES_CFG = SceneEntityCfg(
  "robot",
  body_names=["left_ankle_roll_link", "right_ankle_roll_link"],
)
_ARMS_CFG = SceneEntityCfg(
  "robot",
  joint_names=[
    ".*_shoulder_.*_joint",
    ".*_elbow_joint",
    ".*_wrist_.*",
  ],
)
_HIP_CFG = SceneEntityCfg(
  "robot",
  joint_names=[".*_hip_yaw_joint", ".*_hip_roll_joint"],
)
_WAIST_CFG = SceneEntityCfg(
  "robot",
  joint_names=["waist.*"],
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
  phase: AmePhase,
  variant: AmeVariant,
  play: bool,
) -> dict[str, ObservationGroupCfg]:
  finetune_train = phase == "finetune" and not play

  actor_proprio_terms = {
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
        "height_noise_range": (-0.03, 0.03) if finetune_train else None,
        "apply_drift": finetune_train,
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
      enable_corruption=finetune_train,
    ),
    "actor_terrain": ObservationGroupCfg(
      terms=actor_terrain_terms,
      concatenate_terms=True,
      enable_corruption=finetune_train,
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


def _make_rewards(phase: AmePhase):
  rewards = {
    "termination_penalty": RewardTermCfg(func=envs_mdp.is_terminated, weight=-200.0),
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=ame_mdp.track_lin_vel_xy_yaw_frame_exp,
      weight=2.0,
      params={
        "command_name": "twist",
        "std": 0.25,
      },
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=ame_mdp.track_ang_vel_z_world_exp,
      weight=3.0,
      params={
        "command_name": "twist",
        "std": 0.25,
      },
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=ame_mdp.ang_vel_xy_l2,
      weight=-0.05,
    ),
    "undesired_contacts": RewardTermCfg(
      func=ame_mdp.undesired_contacts,
      weight=-1.0,
      params={"sensor_name": "nonfoot_ground_contact", "threshold": 1.0},
    ),
    "dof_torques_l2": RewardTermCfg(
      func=vel_mdp.joint_torques_l2,
      weight=-1.5e-7,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "dof_acc_l2": RewardTermCfg(
      func=vel_mdp.joint_acc_l2,
      weight=-1.25e-7,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "dof_vel_l2": RewardTermCfg(
      func=vel_mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "dof_pos_limits": RewardTermCfg(
      func=vel_mdp.joint_pos_limits,
      weight=-1.0,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "dof_torques_limits": RewardTermCfg(
      func=ame_mdp.applied_torque_limits,
      weight=-0.01,
      params={"asset_cfg": deepcopy(_ROBOT_CFG)},
    ),
    "action_rate_l2": RewardTermCfg(func=vel_mdp.action_rate_l2, weight=-0.01),
    "flat_orientation_l2": RewardTermCfg(
      func=vel_mdp.flat_orientation_l2,
      weight=-2.0,
    ),
    "feet_air_time": RewardTermCfg(
      func=ame_mdp.feet_air_time_positive_biped,
      weight=0.25,
      params={
        "command_name": "twist",
        "sensor_name": "feet_ground_contact",
        "threshold": 0.6,
      },
    ),
    "feet_air_time_variance": RewardTermCfg(
      func=ame_mdp.air_time_variance_penalty,
      weight=-0.7,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_slide": RewardTermCfg(
      func=vel_mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "twist",
        "command_threshold": 0.05,
        "asset_cfg": deepcopy(_FEET_CFG),
      },
    ),
    "feet_stumble": RewardTermCfg(
      func=ame_mdp.feet_stumble,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "feet_too_near": RewardTermCfg(
      func=ame_mdp.feet_too_near,
      weight=-1.0,
      params={"threshold": 0.2, "asset_cfg": deepcopy(_ANKLES_CFG)},
    ),
    "joint_coordination": RewardTermCfg(
      func=ame_mdp.joint_coordination_rel,
      weight=-0.2,
      params={
        "asset_cfg": deepcopy(_ROBOT_CFG),
        "coord_joints": [
          ["left_hip_pitch_joint", "right_shoulder_pitch_joint"],
          ["right_hip_pitch_joint", "left_shoulder_pitch_joint"],
        ],
        "coord_signs": [
          [1.0, 1.0],
          [1.0, 1.0],
        ],
      },
    ),
    "joint_deviation_hip": RewardTermCfg(
      func=ame_mdp.joint_deviation_l1,
      weight=-0.1,
      params={"asset_cfg": deepcopy(_HIP_CFG)},
    ),
    "joint_deviation_arms": RewardTermCfg(
      func=ame_mdp.joint_deviation_l1,
      weight=-0.3,
      params={"asset_cfg": deepcopy(_ARMS_CFG)},
    ),
    "joint_deviation_waists": RewardTermCfg(
      func=ame_mdp.joint_deviation_l1,
      weight=-1.0,
      params={"asset_cfg": deepcopy(_WAIST_CFG)},
    ),
  }
  if phase == "finetune":
    rewards["dof_torques_limits"].weight = -0.05
    rewards["action_rate_l2"].weight = -0.05
    rewards["flat_orientation_l2"].weight = -5.0
    rewards["feet_air_time"].weight = 0.5
    rewards["feet_air_time_variance"].weight = -2.0
    rewards["feet_slide"].weight = -0.3
    rewards["feet_stumble"].weight = -5.0
    rewards["feet_too_near"].weight = -5.0
    rewards["joint_coordination"].weight = -0.5
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


def _make_phase_events(cfg, phase: AmePhase, play: bool) -> dict[str, EventTermCfg]:
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
  if phase == "finetune":
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


def _ensure_contact_sensors(cfg) -> None:
  sensor_names = {sensor.name for sensor in (cfg.scene.sensors or ())}
  sensors = list(cfg.scene.sensors or ())
  if "torso_ground_contact" not in sensor_names:
    sensors.append(
      ContactSensorCfg(
        name="torso_ground_contact",
        primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="none",
        num_slots=1,
        history_length=4,
      )
    )
  if "nonfoot_ground_contact" not in sensor_names:
    sensors.append(
      ContactSensorCfg(
        name="nonfoot_ground_contact",
        primary=ContactMatch(mode="body", pattern="(?!.*ankle.*).*", entity="robot"),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        history_length=4,
      )
    )
  cfg.scene.sensors = tuple(sensors)


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
  phase: AmePhase,
  variant: AmeVariant = DEFAULT_VARIANT,
  play: bool = False,
):
  """Build the AME environment on top of mjlab's Unitree G1 rough task."""

  cfg = deepcopy(unitree_g1_rough_env_cfg(play=play))
  task_spec = resolve_task_spec(variant, phase)
  cfg.seed = 42
  cfg.episode_length_s = 20.0 if not play else int(1e9)
  cfg.scene.num_envs = task_spec.play_num_envs if play else task_spec.train_num_envs
  cfg.scene.terrain.terrain_generator = make_phase_terrain_generator(phase=phase, play=play)  # type: ignore[union-attr]
  cfg.scene.terrain.max_init_terrain_level = 5  # type: ignore[union-attr]
  _configure_terrain_scan(cfg, variant)
  _ensure_contact_sensors(cfg)
  cfg.observations = _make_observations(phase=phase, variant=variant, play=play)
  cfg.rewards = _make_rewards(phase=phase)
  cfg.terminations = _make_terminations()
  cfg.events = _make_phase_events(cfg, phase=phase, play=play)
  if play:
    cfg.curriculum = {}
  else:
    cfg.scene.terrain.terrain_generator.curriculum = True  # type: ignore[union-attr]
  return cfg
