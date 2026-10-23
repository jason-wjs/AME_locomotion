from __future__ import annotations

import unittest

from mjlab.sensor import GridPatternCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import DEFAULT_VARIANT, iter_task_specs, resolve_task_spec
from ame_g1.terrains import make_phase_terrain_generator


def _terrain_scan_cfg(env_cfg):
  for sensor in env_cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      return sensor
  raise AssertionError("terrain_scan sensor not found")


class VariantConfigTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    register_ame_tasks()

  def test_default_variant_is_g1(self) -> None:
    self.assertEqual(DEFAULT_VARIANT, "g1")

  def test_ame_command_and_scan_contract(self) -> None:
    cfg = load_env_cfg(resolve_task_spec("g1", "base").task_id)
    self.assertEqual(cfg.commands["twist"].rel_forward_envs, 0.0)
    self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x, (0.0, 1.5))
    self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y, (0.0, 0.0))
    self.assertNotIn("command_vel", cfg.curriculum)
    sensors = {sensor.name: sensor for sensor in cfg.scene.sensors}
    self.assertEqual(sensors["terrain_scan"].include_geom_groups, (0,))
    self.assertEqual(sensors["terrain_scan"].frame.name, "torso_link")
    self.assertEqual(cfg.actions["joint_pos"].scale, 0.25)
    self.assertNotIn("foot_height_scan", sensors)
    for group in ("actor_proprio", "critic_proprio"):
      self.assertEqual(cfg.observations[group].terms["base_ang_vel"].scale, 0.2)
      self.assertEqual(cfg.observations[group].terms["joint_vel"].scale, 0.05)

  def test_all_variant_phase_pairs_are_defined(self) -> None:
    specs = list(iter_task_specs())
    self.assertEqual(
      {(spec.variant, spec.phase) for spec in specs},
      {
        ("ame2-gaze", "base"),
        ("ame2", "base"),
        ("paper", "base"),
        ("paper", "finetune"),
        ("g1", "base"),
        ("g1", "finetune"),
      },
    )
    self.assertEqual(len({spec.task_id for spec in specs}), 6)
    self.assertEqual(len({spec.experiment_name for spec in specs}), 6)

  def test_variant_specific_env_and_model_cfgs_match_plan(self) -> None:
    paper = resolve_task_spec("paper", "base")
    g1 = resolve_task_spec("g1", "base")

    paper_env = load_env_cfg(paper.task_id)
    g1_env = load_env_cfg(g1.task_id)
    self.assertEqual(paper_env.scene.num_envs, 4096)
    self.assertEqual(g1_env.scene.num_envs, 1792)
    self.assertEqual(g1_env.sim.broadphase, "nxn")
    self.assertEqual(g1_env.scene.terrain.max_init_terrain_level, 0)
    course = g1_env.scene.terrain.terrain_generator
    self.assertEqual((course.num_rows, course.num_cols, course.size), (10, 20, (8., 8.)))
    self.assertEqual(
      [terrain.family for terrain in course.sub_terrains.values()],
      ["stairs"] * 2 + ["inverse_stairs"] * 2 + ["grid"] * 2
      + ["rough"] * 2 + ["slope"] * 2 + ["inverse_slope"] * 2
      + ["stones"] * 4 + ["gaps"] * 4,
    )

    paper_scan = _terrain_scan_cfg(paper_env)
    g1_scan = _terrain_scan_cfg(g1_env)
    self.assertIsInstance(paper_scan.pattern, GridPatternCfg)
    self.assertIsInstance(g1_scan.pattern, GridPatternCfg)
    assert isinstance(paper_scan.pattern, GridPatternCfg)
    assert isinstance(g1_scan.pattern, GridPatternCfg)
    self.assertEqual(paper_scan.pattern.resolution, 0.1)
    self.assertEqual(g1_scan.pattern.resolution, 0.05)

    paper_rl = load_rl_cfg(paper.task_id)
    g1_rl = load_rl_cfg(g1.task_id)
    self.assertEqual(g1_rl.algorithm.num_mini_batches, 8)

    self.assertEqual(paper_rl.actor.encoder_variant, "paper")
    self.assertEqual(paper_rl.actor.terrain_input_mode, "z")
    self.assertTrue(paper_rl.actor.concat_coords_post_cnn)
    self.assertFalse(paper_rl.actor.cnn_downsample)
    self.assertFalse(paper_rl.actor.attach_global)

    self.assertEqual(g1_rl.actor.encoder_variant, "g1")
    self.assertEqual(g1_rl.actor.terrain_input_mode, "xyz")
    self.assertFalse(g1_rl.actor.concat_coords_post_cnn)
    self.assertTrue(g1_rl.actor.cnn_downsample)
    self.assertFalse(g1_rl.actor.attach_global)

  def test_finetune_training_terrain_budget_matches_base(self) -> None:
    base_generator = make_phase_terrain_generator("base", play=False)
    finetune_generator = make_phase_terrain_generator("finetune", play=False)

    base_tiles = base_generator.num_rows * base_generator.num_cols
    finetune_tiles = finetune_generator.num_rows * finetune_generator.num_cols

    self.assertEqual(base_tiles, 40)
    self.assertEqual(finetune_tiles, base_tiles)


if __name__ == "__main__":
  unittest.main()
