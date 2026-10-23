# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
"""AME-2 teacher task adapted to mjlab's G1-29 asset.

Reference: leggedrobotics/ame2_minimal@8beb9a6, ame2/tasks/ame2_env_cfg.py.
Simulator and robot differences are recorded in docs/stage2-preparation.md.
"""
from copy import deepcopy
import math

import mujoco

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp
from mjlab.envs.mdp import dr
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, ObjRef
from mjlab.sim import SimulationCfg, MujocoCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg
from mjlab.viewer import ViewerConfig

from ame_g1.config.g1.ame2_robot import make_ame2_g1_cfg, ame2_joint_velocity_limits
from ame_g1.mdp.ame2_actions import Ame2DelayedJointPositionActionCfg
from ame_g1.mdp.ame2_commands import SafeTerrainBasedPose2dCommandCfg
from ame_g1.mdp import ame2_observations as obs
from ame_g1.mdp import ame2_rewards as rewards
from ame_g1.mdp import ame2_terminations as terminations
from ame_g1.mdp.ame2_events import randomize_mass_with_inertia, reset_joints_by_scale
from ame_g1.mdp.ame2_curriculum import ame2_terrain_levels_goalreaching, ame2_teacher_schedule
from ame_g1.mdp.ame2_scan import AME2RayCastSensorCfg
from ame_g1.terrains.ame2_config import make_ame2_terrain_generator


def _contact(name, bodies, secondary=None):
  return ContactSensorCfg(
    name=name, primary=ContactMatch(mode="body", pattern=bodies, entity="robot"),
    secondary=secondary, fields=("found", "force"), reduce="netforce",
    history_length=4, track_air_time=True,
  )


def _observations(play):
  prop = {
    "base_lin_vel": ObservationTermCfg(
      func=obs.ame2_base_lin_vel,
      noise=UniformNoiseCfg(n_min=-0.1, n_max=0.1),
    ),
    "base_ang_vel": ObservationTermCfg(
      func=obs.ame2_base_ang_vel,
      noise=UniformNoiseCfg(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=obs.ame2_projected_gravity,
      noise=UniformNoiseCfg(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=UniformNoiseCfg(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=UniformNoiseCfg(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=obs.ame2_last_action),
    "pose_commands": ObservationTermCfg(
      func=obs.ame2_commands_obs,
      params={'command_name': 'base_pose', 'dist_thr': 2.0},
    ),
  }
  critic = deepcopy(prop)
  critic["pose_commands"].params["dist_thr"] = None
  critic.update(
    {
      'time_left': ObservationTermCfg(func=obs.ame2_time_left),
      'contact_states': ObservationTermCfg(func=obs.ame2_contact_states, params={'sensor_name': 'contact_forces'}),
      'height_scan': ObservationTermCfg(
        func=obs.ame2_height_scan,
        params={'sensor_name': 'height_scanner_critic', 'offset': 0.74},
        clip=(-1.5, 1.5),
      ),
    },
  )
  return {
    "teacher_prop": ObservationGroupCfg(
      terms=prop,
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "teacher_mapping": ObservationGroupCfg(
      terms={
        "teacher_mapping": ObservationTermCfg(
          func=obs.ame2_teacher_mapping,
          params={
            "sensor_name": "height_scanner",
            "offset": 0.74,
            "custom_noise": (0., 0., 0.) if play else (.001, .001, .05),
          },
        ),
      },
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(terms=critic, concatenate_terms=True, enable_corruption=False),
  }


def make_ame2_gaze_teacher_env_cfg(play=False):
  """Gaze's 23 controlled joints and 82-D proprioception, in native joint order.

  Wrists retain the native asset's PD control at the default position. Importing
  official weights requires an explicit joint-order conversion, not direct load.
  """
  cfg = make_ame2_teacher_env_cfg(play=play)
  spec = cfg.scene.entities["robot"].spec_fn()
  joints = tuple(j.name for j in spec.joints
                 if j.type != mujoco.mjtJoint.mjJNT_FREE and "wrist" not in j.name)
  cfg.actions["joint_pos"].actuator_names = joints
  for group in ("teacher_prop", "critic"):
    for term in ("joint_pos", "joint_vel"):
      cfg.observations[group].terms[term].params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=joints,
      )
  return cfg


def make_ame2_teacher_env_cfg(play=False):
  robot_cfg = make_ame2_g1_cfg()
  # Joint names come from the retained XML, preserving its actual ordering.
  robot_spec = robot_cfg.spec_fn()
  joint_names = [
    joint.name for joint in robot_spec.joints
    if joint.type != mujoco.mjtJoint.mjJNT_FREE
  ]
  torso = SceneEntityCfg("robot", body_names=("torso_link",))
  feet = SceneEntityCfg("robot", body_names=(".*ankle_roll_link",))
  all_bodies = SceneEntityCfg("robot", body_names=(".*",))
  sensors = (
    AME2RayCastSensorCfg(
      name='height_scanner',
      frame=ObjRef(type='body', name='pelvis', entity='robot'),
      include_geom_groups=(0,),
      drift_range=(0.0, 0.0) if play else (-0.04, 0.04),
    ),
    AME2RayCastSensorCfg(
      name='height_scanner_critic',
      frame=ObjRef(type='body', name='pelvis', entity='robot'),
      include_geom_groups=(0,),
      drift_range=(0.0, 0.0),
    ),
    _contact("contact_forces", ".*"),
    _contact("feet_contact", ".*ankle_roll_link"),
    _contact("nonfoot_contact", r"^(?!.*ankle).*"),
    _contact("torso_pelvis_contact", r"^(torso_link|pelvis)$"),
    _contact(
      'self_col_knee',
      'left_knee_link',
      ContactMatch(mode='body', pattern='right_knee_link', entity='robot'),
    ),
    _contact(
      'self_col_ankle',
      'left_ankle_roll_link',
      ContactMatch(mode='body', pattern='right_ankle_roll_link', entity='robot'),
    ),
    _contact("self_col_torso", r".*(shoulder_roll|shoulder_yaw|elbow|wrist_pitch)_link",
             ContactMatch(mode="body",pattern="torso_link",entity="robot")),
  )
  reward_terms = {
    "ame2_position_tracking": RewardTermCfg(
      func=rewards.ame2_position_tracking,
      weight=100.0,
      params={'command_name': 'base_pose', 'T': 4.0, 'std': 2.0},
    ),
    "ame2_heading_tracking": RewardTermCfg(
      func=rewards.ame2_heading_tracking,
      weight=50.0,
      params={'command_name': 'base_pose', 'T': 2.0, 'dist_thr': 0.5},
    ),
    "ame2_move2goal": RewardTermCfg(
      func=rewards.ame2_move2goal,
      weight=5.0,
      params={'command_name': 'base_pose', 'cos_thr': 0.5, 'dist_thr': 0.5, 'asset_cfg': torso},
    ),
    "ame2_standatgoal": RewardTermCfg(
      func=rewards.ame2_standatgoal,
      weight=5.0,
      params={'command_name': 'base_pose', 'sensor_name': 'feet_contact'},
    ),
    "early_termination_penalty": RewardTermCfg(func=rewards.ame2_is_terminated_term,weight=-500.),
    "undesired_spinning": RewardTermCfg(
      func=rewards.ame2_undesired_spinning,
      weight=-1.0,
      params={'asset_cfg': torso, 'yaw_rate_thr': 3.0},
    ),
    "undesired_leaping": RewardTermCfg(
      func=rewards.ame2_undesired_leaping,
      weight=-1.0,
      params={
        'sensor_name': 'feet_contact',
        'height_scanner_name': 'height_scanner_critic',
        'start_time': 0.2,
      },
    ),
    "undesired_non_foot_contacts": RewardTermCfg(
      func=rewards.ame2_undesired_non_foot_contacts,
      weight=-1.0,
      params={'sensor_name': 'nonfoot_contact'},
    ),
    "undesired_stumbling": RewardTermCfg(
      func=rewards.ame2_undesired_stumbling,
      weight=-1.0,
      params={'sensor_name': 'contact_forces'},
    ),
    "undesired_slippage": RewardTermCfg(
      func=rewards.ame2_undesired_slippage,
      weight=-1.0,
      params={'sensor_name': 'contact_forces', 'asset_cfg': all_bodies, 'slip_vel_thr': 0.3},
    ),
    **{
      f"undesired_self_col_{part}": RewardTermCfg(
        func=rewards.ame2_undesired_torso_collision if part == 'torso' else rewards.ame2_undesired_self_collision,
        weight=-1.0,
        params={'sensor_name': f'self_col_{part}'},
      ) for part in ("knee", "torso", "ankle")
    },
    "base_roll_rate": RewardTermCfg(
      func=rewards.ame2_base_roll_rate_l2,
      weight=-0.1,
      params={'asset_cfg': torso},
    ),
    "joint_regularization": RewardTermCfg(func=rewards.ame2_joint_regularization_l2,weight=-.001),
    "action_smoothness": RewardTermCfg(func=rewards.ame2_action_rate_l2,weight=-.01),
    "link_contact_forces": RewardTermCfg(
      func=rewards.ame2_link_contact_forces_l2,
      weight=-1e-05,
      params={'sensor_name': 'contact_forces', 'robot_weight': 400.0},
    ),
    "body_lin_acc_l1": RewardTermCfg(
      func=rewards.ame2_body_lin_acc_l1,
      weight=-0.001,
      params={'asset_cfg': all_bodies},
    ),
    "joint_pos_limits": RewardTermCfg(func=rewards.ame2_joint_pos_limits,weight=-1000.),
    "joint_vel_limits": RewardTermCfg(
      func=rewards.ame2_joint_vel_limits,
      weight=-1.0,
      params={'velocity_limits': ame2_joint_velocity_limits(joint_names)},
    ),
    "joint_torque_limits": RewardTermCfg(func=rewards.ame2_joint_torque_limits,weight=-1.),
    "ame2_optional_uppershaping": RewardTermCfg(
      func=rewards.ame2_optional_uppershaping,
      weight=-1.0,
      params={
        'asset_cfg': SceneEntityCfg(
          'robot',
          joint_names=('.*_shoulder_(pitch|roll|yaw)_joint', '.*_elbow_joint'),
        ),
        'height_scanner_name': 'height_scanner_critic',
      },
    ),
    "ame2_optional_foot_shaping": RewardTermCfg(
      func=rewards.ame2_optional_foot_yaw,
      weight=-1.0,
      params={'asset_cfg': feet, 'torso_cfg': torso},
    ),
  }
  events = {
    "reset_base": EventTermCfg(func=mdp.reset_root_state_uniform,mode="reset",params={
      "pose_range":{"x":(-.2,.2),"y":(-.2,.2),"yaw":(-math.pi/4,math.pi/4)},
      "velocity_range":{key:(-.5,.5) for key in ("x","y","z","roll","pitch","yaw")},
    }),
    "reset_robot_joints": EventTermCfg(func=reset_joints_by_scale,mode="reset"),
  }
  if not play:
    events.update(
      {
        'physics_material': EventTermCfg(
          func=dr.geom_friction,
          mode='startup',
          params={
            'asset_cfg': SceneEntityCfg('robot', geom_names=('.*',)),
            'ranges': (0.3, 1.0),
            'operation': 'abs',
          },
        ),
        'add_base_mass': EventTermCfg(
          func=randomize_mass_with_inertia,
          mode='startup',
          params={'asset_cfg': torso, 'ranges': (-1.0, 3.0)},
        ),
        'base_com': EventTermCfg(
          func=dr.body_com_offset,
          mode='startup',
          params={
            'asset_cfg': torso,
            'ranges': {0: (-0.02, 0.02), 1: (-0.02, 0.02), 2: (-0.04, 0.04)},
            'operation': 'add',
          },
        ),
        'actuator_gains': EventTermCfg(
          func=dr.pd_gains,
          mode='startup',
          params={'kp_range': (0.85, 1.15), 'kd_range': (0.85, 1.15)},
        ),
        'armature': EventTermCfg(
          func=dr.joint_armature,
          mode='startup',
          params={'ranges': (0.85, 1.15), 'operation': 'scale'},
        ),
      },
    )
  curriculum = {} if play else {"terrain_levels":CurriculumTermCfg(
    func=ame2_terrain_levels_goalreaching,
    params={'command_name': 'base_pose'},
  )}
  if not play:
    curriculum["teacher_schedule"] = CurriculumTermCfg(func=ame2_teacher_schedule)
  cfg = ManagerBasedRlEnvCfg(
    decimation=4,
    episode_length_s=16.0,
    scene=SceneCfg(
      num_envs=4096 if not play else 1,
      entities={'robot': robot_cfg},
      sensors=sensors,
      terrain=TerrainEntityCfg(
        terrain_type='generator',
        terrain_generator=make_ame2_terrain_generator(num_rows=2 if play else 10),
        max_init_terrain_level=1 if play else 5,
      ),
    ),
    sim=SimulationCfg(mujoco=MujocoCfg(timestep=0.005), nconmax=70, contact_sensor_maxmatch=500),
    observations=_observations(play),
    actions={
      'joint_pos': Ame2DelayedJointPositionActionCfg(
        entity_name='robot',
        actuator_names=('.*',),
        delay_range=(0, 0) if play else (0, 1),
      ),
    },
    commands={'base_pose': SafeTerrainBasedPose2dCommandCfg(resampling_time_range=(16.01, 16.01))},
    rewards=reward_terms,
    events=events,
    curriculum=curriculum,
    terminations={
      'time_out': TerminationTermCfg(func=mdp.time_out, time_out=True),
      'early_gravity': TerminationTermCfg(func=terminations.ame2_bad_orientation, params={'start_time': 0.2}),
      'early_hip_acc': TerminationTermCfg(
        func=terminations.ame2_body_lin_acc_out_of_limit,
        params={
          'threshold': 100.0,
          'asset_cfg': SceneEntityCfg('robot', body_names=('.*hip_pitch_link',)),
          'foot_sensor_name': 'feet_contact',
          'start_time': 1.0,
        },
      ),
      'early_illegal_contact': TerminationTermCfg(
        func=terminations.ame2_illegal_contact,
        params={'threshold': 400.0, 'sensor_name': 'torso_pelvis_contact', 'start_time': 1.0},
      ),
      'early_stagnation': TerminationTermCfg(
        func=terminations.ame2_stagnation,
        params={
          'distance_threshold': 0.5,
          'goal_distance_threshold': 1.0,
          'time_window': 5.0,
          'command_name': 'base_pose',
        },
      ),
    },
    viewer=ViewerConfig(body_name='torso_link'),
  )
  return cfg
