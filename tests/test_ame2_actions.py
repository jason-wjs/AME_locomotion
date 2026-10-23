"""Physics-step action timing and the explicit G1-29 motor adaptation."""

from types import SimpleNamespace
import unittest

import torch
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.actuator.dc_actuator import dc_motor_clip
from mjlab.entity import Entity

from ame_g1.config.g1.ame2_robot import ame2_joint_velocity_limits, make_ame2_g1_cfg
from ame_g1.mdp.ame2_actions import Ame2DelayedJointPositionActionCfg


class FakeRobot:
  def __init__(self):
    self.data = SimpleNamespace(
      default_joint_pos=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
      encoder_bias=torch.zeros(2, 2),
    )
    self.target = None

  def find_joints_by_actuator_names(self, names):
    return [0, 1], ["left_joint", "right_joint"]

  def set_joint_position_target(self, target, joint_ids):
    self.target = target.clone()


class TeacherActionTests(unittest.TestCase):
  def test_delay_advances_on_physics_substeps_and_partial_reset_is_local(self):
    robot = FakeRobot()
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
    action = Ame2DelayedJointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), delay_range=(1, 1),
    ).build(env)
    neutral = robot.data.default_joint_pos
    commands = torch.tensor([[4., -8.], [8., 4.]])
    before = action.action_history.clone()
    action.process_actions(commands)
    torch.testing.assert_close(action.action_history, before)
    action.apply_actions()
    torch.testing.assert_close(robot.target, neutral)
    # No new policy action: the second physics substep must still advance latency.
    action.apply_actions()
    torch.testing.assert_close(robot.target, neutral + 0.25 * commands)
    other_history = action.action_history[1].clone()
    action.reset(torch.tensor([0]))
    torch.testing.assert_close(action.action_history[0], neutral[0].expand(2, -1))
    torch.testing.assert_close(action.action_history[1], other_history)
    torch.testing.assert_close(action.raw_action[0], torch.zeros(2))
    torch.testing.assert_close(action.raw_action[1], commands[1])
    self.assertEqual(action.delay_steps.tolist(), [1, 1])

  def test_zero_delay_clips_processed_target_and_applies_encoder_bias(self):
    robot = FakeRobot()
    robot.data.encoder_bias.fill_(0.05)
    env = SimpleNamespace(num_envs=2, device="cpu", scene={"robot": robot})
    action = Ame2DelayedJointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), delay_range=(0, 0),
    ).build(env)
    action.process_actions(torch.tensor([[1000., -1000.], [0., 0.]]))
    action.apply_actions()
    expected = torch.tensor([[100., -100.], [0.3, 0.4]]) - 0.05
    torch.testing.assert_close(robot.target, expected)
    action.reset(slice(None))
    torch.testing.assert_close(action._processed_actions, robot.data.default_joint_pos)

  def test_robot_compiles_with_all_joints_controlled_and_reference_dc_limits(self):
    cfg = make_ame2_g1_cfg()
    self.assertEqual(cfg.init_state.pos, (0., 0., 0.78))
    robot = Entity(cfg)
    model = robot.compile()
    self.assertEqual(model.nu, 29)
    self.assertEqual(model.nq, 36)
    self.assertEqual(len(set(model.actuator_trnid[:, 0].tolist())), 29)
    self.assertTrue((model.actuator_forcerange[:, 0] < 0).all())
    self.assertTrue((model.actuator_forcerange[:, 1] > 0).all())
    for actuator in cfg.articulation.actuators:
      for name in robot.find_joints(actuator.target_names_expr)[1]:
        self.assertEqual(model.actuator(name).forcerange.tolist(), [-actuator.effort_limit, actuator.effort_limit])
    dof = model.jnt_dofadr[model.joint("left_hip_roll_joint").id]
    self.assertAlmostEqual(model.dof_armature[dof], 0.025101925)
    self.assertEqual(sum(isinstance(a, DcMotorActuatorCfg) for a in cfg.articulation.actuators), 8)
    hip_roll = next(a for a in cfg.articulation.actuators if a.target_names_expr == (".*_hip_roll_joint",))
    self.assertEqual((hip_roll.stiffness, hip_roll.damping, hip_roll.effort_limit), (139., 3., 88.))
    clipped = dc_motor_clip(
      torch.tensor([1000., 1000., -1000., -1000.]),
      torch.tensor(110.), torch.tensor(32.), torch.tensor(88.),
      torch.tensor([0., 32., 0., -32.]),
    )
    torch.testing.assert_close(clipped, torch.tensor([88., 0., -88., 0.]))
    # Returned configs must not mutate mjlab's shared defaults or another instance.
    cfg.init_state.joint_pos[".*_knee_joint"] = 99
    self.assertEqual(make_ame2_g1_cfg().init_state.joint_pos[".*_knee_joint"], 0.62)

  def test_action_order_is_joint_order_despite_regrouped_motors(self):
    robot = Entity(make_ame2_g1_cfg())
    robot.compile()
    # Use real Entity joint/actuator resolution and target setter with CPU buffers.
    robot._data = SimpleNamespace(
      default_joint_pos=torch.zeros(1, 29), encoder_bias=torch.zeros(1, 29),
      joint_pos_target=torch.zeros(1, 29),
    )
    env = SimpleNamespace(num_envs=1, device="cpu", scene={"robot": robot})
    action = Ame2DelayedJointPositionActionCfg(
      entity_name="robot", actuator_names=(".*",), delay_range=(0, 0),
    ).build(env)
    self.assertEqual(tuple(action.target_names), robot.joint_names)
    self.assertEqual(action.target_ids.tolist(), list(range(29)))
    knee_index = robot.joint_names.index("right_knee_joint")
    command = torch.zeros(1, 29)
    command[0, knee_index] = 3.0
    action.process_actions(command)
    action.apply_actions()
    torch.testing.assert_close(robot.data.joint_pos_target, command * 0.25)
    self.assertEqual(robot.data.joint_pos_target[0, knee_index].item(), 0.75)
    limits = dict(zip(robot.joint_names, ame2_joint_velocity_limits(robot.joint_names)))
    self.assertEqual(limits["left_hip_roll_joint"], 32.0)
    self.assertEqual(limits["right_knee_joint"], 20.0)
    self.assertEqual(limits["left_wrist_roll_joint"], 37.0)
    self.assertEqual(limits["right_wrist_pitch_joint"], 22.0)


if __name__ == "__main__":
  unittest.main()
