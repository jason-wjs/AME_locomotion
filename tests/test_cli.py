from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
import json
import unittest
from unittest.mock import patch

import mjlab
from mjlab.scripts.train import TrainConfig
import tyro
from mujoco_warp._src import collision_convex, collision_flex, types

from ame_g1.cli import train as train_cli
from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import resolve_task_spec
from ame_g1.learning import runner as training_runner


class AmeCliTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    register_ame_tasks()

  def test_train_help_renders_with_symmetry_cfg_defaults(self) -> None:
    default = TrainConfig.from_task(resolve_task_spec("g1", "base").task_id)
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
    default = TrainConfig.from_task(resolve_task_spec("g1", "base").task_id)
    cfg = tyro.cli(
      TrainConfig,
      args=["--gpu-ids", "[0]"],
      default=default,
      config=mjlab.TYRO_FLAGS,
    )
    self.assertEqual(cfg.gpu_ids, [0])

  def test_train_preflight_reports_effective_capacity_without_starting_training(self) -> None:
    modules = (types, collision_convex, collision_flex)
    for module in modules:
      patcher = patch.object(module, "MJ_MAX_EPAHORIZON", 24)
      patcher.start()
      self.addCleanup(patcher.stop)
    stdout = StringIO()
    with (
      patch("sys.argv", ["ame-train", "--variant", "ame2-gaze", "--preflight", "--gpu-ids", "[0]"]),
      patch.object(train_cli, "launch_ame_training", side_effect=AssertionError("Training launched")) as launch,
      patch.object(training_runner, "ManagerBasedRlEnv", side_effect=AssertionError("Environment created")) as env,
      patch.object(training_runner, "select_gpus", side_effect=AssertionError("GPU selected")) as select_gpus,
      patch.object(training_runner, "load_runner_cls", side_effect=AssertionError("Runner loaded")) as load_runner,
      redirect_stdout(stdout),
    ):
      train_cli.main()
    report = json.loads(stdout.getvalue())
    self.assertEqual(report["task_id"], "AME2-G1-Gaze-Teacher")
    self.assertEqual(report["effective_collision_capacities"], {module.__name__: 64 for module in modules})
    self.assertEqual(report["scope"], "configuration_only_not_physics_validation")
    launch.assert_not_called()
    env.assert_not_called()
    select_gpus.assert_not_called()
    load_runner.assert_not_called()

  def test_train_without_preflight_preserves_launch(self) -> None:
    with (
      patch("sys.argv", ["ame-train", "--variant", "ame2-gaze", "--gpu-ids", "[0]"]),
      patch.object(train_cli, "launch_ame_training") as launch,
    ):
      train_cli.main()
    launch.assert_called_once()
    self.assertEqual(launch.call_args.kwargs["task_spec"].task_id, "AME2-G1-Gaze-Teacher")
    self.assertEqual(launch.call_args.kwargs["args"].gpu_ids, [0])


if __name__ == "__main__":
  unittest.main()
