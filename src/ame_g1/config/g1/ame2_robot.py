# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AME-2 motor recipe adapted to mjlab's G1-29 model.

The 23 non-wrist joints use the gains, armatures and DC motor torque-speed limits
from leggedrobotics/ame2_minimal (8beb9a6), assets/unitree.py. The reference asset
is g1_lidar.usd; this task retains mjlab's g1_29dof_rev_1_0 XML masses, inertias,
collision geometry and self-collision exclusions. They are not asserted identical
to the USD, and this configuration is not an official-checkpoint-compatible robot.

The six extra wrist DOFs keep mjlab's native position controllers: roll uses its
5020 gains/armature and 25 Nm effort limit; pitch/yaw use its 4010 gains/armature
and 5 Nm effort limit. They start at zero and remain policy-controlled. No wrist
motor values are inferred from the reference, which defines none.
"""

from copy import deepcopy
import re

from mjlab.actuator import DcMotorActuatorCfg
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import (
  G1_ACTUATOR_4010,
  G1_ACTUATOR_5020,
  ACTUATOR_4010,
  ACTUATOR_5020,
  get_g1_robot_cfg,
)
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def make_ame2_g1_cfg() -> EntityCfg:
  """Return an independent G1-29 entity with the reference non-wrist controls."""
  cfg = deepcopy(get_g1_robot_cfg())
  # Each row: joint expressions, stiffness, damping, effort, no-load speed, armature.
  motor_specs = (
    ((".*_hip_pitch_joint", ".*_hip_yaw_joint", "waist_yaw_joint"), 88., 2., 88., 32., 0.01017752004),
    ((".*_hip_roll_joint",), 139., 3., 88., 32., 0.025101925),
    ((".*_knee_joint",), 139., 3., 139., 20., 0.025101925),
    (("waist_roll_joint", "waist_pitch_joint"), 50., 1., 50., 37., 0.00721945),
    ((".*_ankle_pitch_joint", ".*_ankle_roll_joint"), 35., 0.8, 50., 37., 0.00721945),
    ((".*_shoulder_pitch_joint",), 35., 0.8, 25., 37., 0.003609725),
    ((".*_shoulder_roll_joint", ".*_shoulder_yaw_joint"), 40., 1., 25., 37., 0.003609725),
    ((".*_elbow_joint",), 40., 1., 25., 37., 0.003597),
  )
  motors = tuple(
    DcMotorActuatorCfg(
      target_names_expr=names, stiffness=kp, damping=kd,
      effort_limit=effort, saturation_effort=1.25 * effort,
      velocity_limit=speed, armature=armature, frictionloss=0.0, viscous_damping=0.0,
    )
    for names, kp, kd, effort, speed, armature in motor_specs
  )
  wrist_roll = deepcopy(G1_ACTUATOR_5020)
  wrist_roll.target_names_expr = (".*_wrist_roll_joint",)
  wrist_pitch_yaw = deepcopy(G1_ACTUATOR_4010)
  cfg.articulation = EntityArticulationInfoCfg(
    actuators=(*motors, wrist_roll, wrist_pitch_yaw), soft_joint_pos_limit_factor=0.9,
  )
  cfg.init_state = EntityCfg.InitialStateCfg(
    pos=(0., 0., 0.78),
    joint_pos={
      ".*_hip_pitch_joint": -0.30,
      ".*_hip_roll_joint": 0.0,
      ".*_hip_yaw_joint": 0.0,
      ".*_knee_joint": 0.62,
      ".*_ankle_pitch_joint": -0.33,
      ".*_ankle_roll_joint": 0.0,
      ".*_elbow_joint": 0.30,
      "waist_.*_joint": 0.0,
      "left_shoulder_roll_joint": 0.16,
      "left_shoulder_pitch_joint": 0.35,
      "right_shoulder_roll_joint": -0.16,
      "right_shoulder_pitch_joint": 0.35,
      ".*_shoulder_yaw_joint": 0.0,
      ".*_wrist_.*_joint": 0.0,
    },
    joint_vel={".*": 0.0},
  )
  return cfg


def ame2_joint_velocity_limits(joint_names) -> tuple[float, ...]:
  """Motor no-load speeds in the requested joint order, for the velocity penalty.

  The six native position-controlled wrists do not enforce a torque-speed curve;
  their limits here are the 5020/4010 motor specifications supplied by mjlab.
  All other values come from the DC control configurations themselves.
  """
  motors = make_ame2_g1_cfg().articulation.actuators
  limits = []
  for name in joint_names:
    motor = next(
      actuator for actuator in motors
      if any(re.fullmatch(pattern, name) for pattern in actuator.target_names_expr)
    )
    if isinstance(motor, DcMotorActuatorCfg):
      limits.append(motor.velocity_limit)
    elif name.endswith("_wrist_roll_joint"):
      limits.append(ACTUATOR_5020.velocity_limit)
    else:
      limits.append(ACTUATOR_4010.velocity_limit)
  return tuple(limits)
