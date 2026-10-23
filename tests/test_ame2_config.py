"""Compile the teacher's native robot/sensors and validate MDP wiring on CPU."""

import inspect
import unittest
from types import SimpleNamespace

import torch

import mjlab
from mjlab.scene import Scene
from mjlab.managers.scene_entity_config import SceneEntityCfg

from ame_g1.config.g1.ame2_env_cfg import make_ame2_teacher_env_cfg
from ame_g1.mdp.ame2_events import randomize_mass_with_inertia


class TeacherConfigTests(unittest.TestCase):
  def test_native_scene_and_all_term_arguments_resolve(self):
    cfg = make_ame2_teacher_env_cfg()
    cfg.scene.num_envs = 1
    # Terrain families have independent geometry tests. Here compile every robot
    # and contact sensor without building the full training terrain atlas.
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    scene = Scene(cfg.scene, device="cpu")
    model = scene.compile()
    robot = scene["robot"]
    self.assertEqual(robot.num_joints, 29)
    self.assertEqual(model.nu, 29)
    self.assertEqual(len(cfg.rewards["joint_vel_limits"].params["velocity_limits"]), 29)
    for group in cfg.observations.values():
      for term in group.terms.values():
        inspect.signature(term.func).bind(None, **term.params)
    for terms, extra_args in ((cfg.rewards, ()), (cfg.terminations, ()),
                              (cfg.events, (None,)), (cfg.curriculum, (None,))):
      for name, term in terms.items():
        with self.subTest(term=name):
          for value in term.params.values():
            if isinstance(value, SceneEntityCfg):
              value.resolve(scene)
          if inspect.isclass(term.func):
            inspect.signature(term.func.__call__).bind(None, None, *extra_args, **term.params)
          else:
            inspect.signature(term.func).bind(None, *extra_args, **term.params)
    # Sensor axes must agree with body velocities used by the slip term.
    self.assertEqual(scene["contact_forces"].primary_names, list(robot.body_names))
    self.assertEqual(list(cfg.observations["critic"].terms), [
      "base_lin_vel", "base_ang_vel", "projected_gravity", "joint_pos", "joint_vel",
      "actions", "pose_commands", "time_left", "contact_states", "height_scan",
    ])

  def test_payload_randomization_scales_inertia_once_and_only_selected_envs(self):
    defaults = {"body_mass":torch.tensor([0.,5.,8.]), "body_inertia":torch.tensor([[0.,0.,0.],[1.,2.,3.],[4.,5.,6.]])}
    model = SimpleNamespace(body_mass=defaults["body_mass"].repeat(3,1), body_inertia=defaults["body_inertia"].repeat(3,1,1))
    robot = SimpleNamespace(data=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([1,2]))))
    env = SimpleNamespace(num_envs=3, device="cpu", scene={"robot":robot}, sim=SimpleNamespace(model=model, get_default_field=defaults.__getitem__))
    asset = SceneEntityCfg("robot", body_ids=[0])
    for _ in range(2):
      randomize_mass_with_inertia(env, torch.tensor([1]), (2.,2.), asset)
    torch.testing.assert_close(model.body_mass[1], torch.tensor([0.,7.,8.]))
    torch.testing.assert_close(model.body_inertia[1,1], defaults["body_inertia"][1] * 1.4)
    torch.testing.assert_close(model.body_mass[[0,2]], defaults["body_mass"].repeat(2,1))
    self.assertIn("body_mass", randomize_mass_with_inertia.model_fields)
    self.assertIn("body_inertia", randomize_mass_with_inertia.model_fields)

  def test_play_config_is_independent_and_disables_training_randomization(self):
    train = make_ame2_teacher_env_cfg()
    play = make_ame2_teacher_env_cfg(play=True)
    self.assertEqual(play.curriculum, {})
    self.assertEqual(set(play.events), {"reset_base", "reset_robot_joints"})
    self.assertEqual(play.actions["joint_pos"].delay_range, (0, 0))
    self.assertTrue(train.observations["teacher_prop"].enable_corruption)
    self.assertFalse(play.observations["teacher_prop"].enable_corruption)
    train.commands["base_pose"].cone_angle_deg = 11.
    self.assertEqual(play.commands["base_pose"].cone_angle_deg, 45.)


if __name__ == "__main__":
  unittest.main()
