from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import unittest


class CliModuleTests(unittest.TestCase):
  def test_train_module_is_invokable_via_python_m(self) -> None:
    ame_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    result = subprocess.run(
      [
        sys.executable,
        "-m",
        "ame_g1.cli.train",
        "--variant",
        "g1",
        "--phase",
        "base",
        "--global-context",
        "off",
        "--gpu-ids",
        "None",
        "--env.scene.num-envs",
        "1",
        "--agent.run-name",
        "cli_entry_smoke",
        "--agent.max-iterations",
        "0",
        "--agent.logger",
        "tensorboard",
      ],
      cwd=ame_root,
      env=env,
      capture_output=True,
      text=True,
      check=False,
    )
    self.assertEqual(result.returncode, 0, msg=result.stderr)
    self.assertIn("[INFO] Training with:", result.stdout)

  def test_non_train_module_entrypoints_expose_help(self) -> None:
    ame_root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    cases = {
      "ame_g1.cli.play": "--checkpoint-file",
      "ame_g1.cli.eval": "--num-episodes",
      "ame_g1.cli.plot_attention": "--dump-file",
    }

    for module_name, expected_flag in cases.items():
      with self.subTest(module=module_name):
        result = subprocess.run(
          [sys.executable, "-m", module_name, "--help"],
          cwd=ame_root,
          env=env,
          capture_output=True,
          text=True,
          check=False,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn(expected_flag, result.stdout)


if __name__ == "__main__":
  unittest.main()
