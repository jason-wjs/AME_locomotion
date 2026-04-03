from __future__ import annotations

from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

from ame_g1.config.g1.rl_cfg import unitree_g1_ame_ppo_runner_cfg
from ame_g1.learning.runner import AmeOnPolicyRunner


class _FakeObs:
  def to(self, _device: str):
    return self


class _FakeEnv:
  def __init__(self) -> None:
    self.device = "cpu"
    self.unwrapped = SimpleNamespace(common_step_counter=0)

  def get_observations(self):
    return _FakeObs()

  def step(self, _actions):
    return _FakeObs(), torch.zeros(1), torch.zeros(1), {}


class _FakeAlg:
  def __init__(self) -> None:
    self.actor = SimpleNamespace(last_attention_weights=None, last_attention_points=None)
    self.critic = SimpleNamespace()
    self.learning_rate = 1.0e-3

  def train_mode(self) -> None:
    return None

  def act(self, _obs):
    return torch.zeros(1, 1)

  def process_env_step(self, _obs, _rewards, _dones, _extras) -> None:
    return None

  def compute_returns(self, _obs) -> None:
    return None

  def update(self):
    return {}

  def get_policy(self):
    return SimpleNamespace(output_std=torch.zeros(1))

  def save(self):
    return {"model_state_dict": {}}


class _FakeLogger:
  def __init__(self, log_dir: str) -> None:
    self.writer = object()
    self.log_dir = log_dir

  def init_logging_writer(self) -> None:
    return None

  def process_env_step(self, _rewards, _dones, _extras, _intrinsic_rewards) -> None:
    return None

  def log(self, **_kwargs) -> None:
    return None

  def stop_logging_writer(self) -> None:
    return None


class AmeRunnerSavePolicyTests(unittest.TestCase):
  def test_default_onnx_export_policy_is_final_only(self) -> None:
    cfg = unitree_g1_ame_ppo_runner_cfg(phase="finetune", variant="g1")
    self.assertEqual(cfg.onnx_export_policy, "final_only")

  def test_resume_does_not_immediately_periodically_save(self) -> None:
    runner = object.__new__(AmeOnPolicyRunner)
    with tempfile.TemporaryDirectory() as tmpdir:
      runner.env = _FakeEnv()
      runner.alg = _FakeAlg()
      runner.logger = _FakeLogger(tmpdir)
      runner.cfg = {
        "num_steps_per_env": 1,
        "algorithm": {},
        "save_interval": 100,
      }
      runner.device = "cpu"
      runner.is_distributed = False
      runner.current_learning_iteration = 9400
      runner.save = mock.Mock()

      runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)

      self.assertEqual(runner.save.call_count, 1)
      self.assertTrue(runner.save.call_args.args[0].endswith("model_9400.pt"))

  def test_save_can_skip_export_artifacts(self) -> None:
    runner = object.__new__(AmeOnPolicyRunner)
    runner.env = _FakeEnv()
    runner.alg = _FakeAlg()
    runner.logger = SimpleNamespace(writer=None, save_model=mock.Mock())
    runner.cfg = {"upload_model": False}
    runner.current_learning_iteration = 123
    runner.export_policy_to_onnx = mock.Mock()
    runner.alg.actor.export_attention_metadata = mock.Mock()
    runner.alg.critic.export_attention_metadata = mock.Mock()

    with tempfile.TemporaryDirectory() as tmpdir:
      checkpoint_path = Path(tmpdir) / "model_123.pt"
      AmeOnPolicyRunner.save(runner, str(checkpoint_path), export_artifacts=False)

      self.assertTrue(checkpoint_path.exists())
      runner.export_policy_to_onnx.assert_not_called()
      runner.alg.actor.export_attention_metadata.assert_not_called()
      runner.alg.critic.export_attention_metadata.assert_not_called()


if __name__ == "__main__":
  unittest.main()
