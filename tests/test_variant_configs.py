from __future__ import annotations

import unittest

from mjlab.sensor import GridPatternCfg
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import DEFAULT_VARIANT, iter_task_specs, resolve_task_spec


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

  def test_all_variant_stage_pairs_are_defined(self) -> None:
    specs = list(iter_task_specs())
    self.assertEqual(
      {(spec.variant, spec.stage) for spec in specs},
      {
        ("paper", "stage1"),
        ("paper", "stage2"),
        ("g1", "stage1"),
        ("g1", "stage2"),
      },
    )
    self.assertEqual(len({spec.task_id for spec in specs}), 4)
    self.assertEqual(len({spec.experiment_name for spec in specs}), 4)

  def test_variant_specific_env_and_model_cfgs_match_plan(self) -> None:
    paper = resolve_task_spec("paper", "stage1")
    g1 = resolve_task_spec("g1", "stage1")

    paper_env = load_env_cfg(paper.task_id)
    g1_env = load_env_cfg(g1.task_id)
    self.assertEqual(paper_env.scene.num_envs, 4096)
    self.assertEqual(g1_env.scene.num_envs, 4096)

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

    self.assertEqual(paper_rl.actor.encoder_variant, "paper")
    self.assertEqual(paper_rl.actor.terrain_input_mode, "z")
    self.assertTrue(paper_rl.actor.concat_coords_post_cnn)
    self.assertFalse(paper_rl.actor.cnn_downsample)
    self.assertFalse(paper_rl.actor.attach_global)

    self.assertEqual(g1_rl.actor.encoder_variant, "g1")
    self.assertEqual(g1_rl.actor.terrain_input_mode, "xyz")
    self.assertFalse(g1_rl.actor.concat_coords_post_cnn)
    self.assertTrue(g1_rl.actor.cnn_downsample)
    self.assertTrue(g1_rl.actor.attach_global)


if __name__ == "__main__":
  unittest.main()
