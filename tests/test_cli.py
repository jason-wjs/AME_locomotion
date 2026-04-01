from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import unittest

import mjlab
from mjlab.scripts.train import TrainConfig
import tyro

from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import resolve_task_spec


class AmeCliTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    register_ame_tasks()

  def test_train_help_renders_with_symmetry_cfg_defaults(self) -> None:
    default = TrainConfig.from_task(resolve_task_spec("g1", "stage1").task_id)
    stdout = StringIO()
    stderr = StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
      with self.assertRaises(SystemExit) as cm:
        tyro.cli(
          TrainConfig,
          args=["--help"],
          default=default,
          config=mjlab.TYRO_FLAGS,
        )
    self.assertEqual(cm.exception.code, 0)
    rendered = stdout.getvalue() + stderr.getvalue()
    self.assertIn("--agent.algorithm.symmetry-cfg.use-data-augmentation", rendered)

  def test_train_config_accepts_gpu_id_list_literal(self) -> None:
    default = TrainConfig.from_task(resolve_task_spec("g1", "stage1").task_id)
    cfg = tyro.cli(
      TrainConfig,
      args=["--gpu-ids", "[0]"],
      default=default,
      config=mjlab.TYRO_FLAGS,
    )
    self.assertEqual(cfg.gpu_ids, [0])


if __name__ == "__main__":
  unittest.main()
