# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Community AME G1 controls on mjlab's G1-29 geometry.

The default pose, PD gains, armature and effort limits follow
AME_Locomotion's UNITREE_G1_29DOF_CFG. MuJoCo geometry and contacts remain
mjlab's; this is not a claim of complete PhysX dynamics equivalence.
"""

from copy import deepcopy

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import get_g1_robot_cfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def make_community_g1_cfg() -> EntityCfg:
  cfg = deepcopy(get_g1_robot_cfg())
  # Joint expressions, stiffness, damping, effort limit.
  motors = (
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint"), 100., 2., 88.),
    (("waist_yaw_joint",), 200., 5., 88.),
    ((".*_hip_roll_joint",), 100., 2., 139.),
    ((".*_knee_joint",), 150., 4., 139.),
    ((".*_shoulder_.*_joint", ".*_elbow_joint", ".*_wrist_roll_joint"), 40., 10., 25.),
    ((".*_ankle_.*_joint",), 40., 2., 25.),
    (("waist_roll_joint", "waist_pitch_joint"), 40., 5., 25.),
    ((".*_wrist_pitch_joint", ".*_wrist_yaw_joint"), 40., 10., 5.),
  )
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=tuple(
      BuiltinPositionActuatorCfg(
        target_names_expr=names, stiffness=kp, damping=kd,
        effort_limit=effort, armature=.01,
      )
      for names, kp, kd, effort in motors
    ),
    soft_joint_pos_limit_factor=.9,
  )
  cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0., 0., .8),
    joint_pos={
      ".*_hip_pitch_joint": -.1, ".*_knee_joint": .3,
      ".*_ankle_pitch_joint": -.2, ".*_shoulder_pitch_joint": .3,
      "left_shoulder_roll_joint": .25, "right_shoulder_roll_joint": -.25,
      ".*_elbow_joint": .97, "left_wrist_roll_joint": .15,
      "right_wrist_roll_joint": -.15,
    },
    joint_vel={".*": 0.},
  )
  return cfg
