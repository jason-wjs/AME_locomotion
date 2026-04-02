from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import resolve_task_spec


def _make_env(variant: str, phase: str) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  task_id = resolve_task_spec(variant, phase).task_id
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return env, wrapped


def _make_runner(variant: str, phase: str, wrapped: RslRlVecEnvWrapper):
  task_id = resolve_task_spec(variant, phase).task_id
  runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
  agent_cfg = asdict(load_rl_cfg(task_id))
  return runner_cls(wrapped, agent_cfg, device="cpu")


class AmeSmokeTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    register_ame_tasks()

  def test_installed_mjlab_is_used(self) -> None:
    import mjlab

    mjlab_path = Path(mjlab.__file__).resolve()
    repo_root = Path(__file__).resolve().parents[2]
    self.assertIn("site-packages", str(mjlab_path))
    self.assertFalse(str(mjlab_path).startswith(str(repo_root / "mjlab")))

  def test_phase_observation_shapes(self) -> None:
    expected_shapes = {"paper": (17, 11, 3), "g1": (33, 21, 3)}
    for variant in ("paper", "g1"):
      for phase in ("base", "finetune"):
        env, wrapped = _make_env(variant, phase)
        try:
          obs = wrapped.get_observations()
          self.assertEqual(
            set(obs.keys()),
            {"actor_proprio", "actor_terrain", "critic_proprio", "critic_terrain"},
          )
          self.assertEqual(tuple(obs["actor_proprio"].shape), (1, 96))
          self.assertEqual(tuple(obs["critic_proprio"].shape), (1, 99))
          self.assertEqual(tuple(obs["actor_terrain"].shape), (1, *expected_shapes[variant]))
          self.assertEqual(tuple(obs["critic_terrain"].shape), (1, *expected_shapes[variant]))
          obs, _, _, _ = wrapped.step(torch.zeros(1, wrapped.num_actions))
          self.assertEqual(tuple(obs["actor_terrain"].shape), (1, *expected_shapes[variant]))
        finally:
          env.close()

  def test_actor_emits_attention_weights(self) -> None:
    for variant in ("paper", "g1"):
      env, wrapped = _make_env(variant, "base")
      try:
        runner = _make_runner(variant, "base", wrapped)
        obs = wrapped.get_observations().to("cpu")
        _ = runner.alg.act(obs)
        attention = runner.alg.actor.last_attention_weights
        attention_points = runner.alg.actor.last_attention_points
        self.assertIsNotNone(attention)
        self.assertIsNotNone(attention_points)
        assert attention is not None
        assert attention_points is not None
        self.assertEqual(tuple(attention.shape), (1, 17 * 11))
        self.assertEqual(tuple(attention_points.shape), (1, 17, 11, 3))
      finally:
        env.close()

  def test_base_symmetry_update_smoke(self) -> None:
    env, wrapped = _make_env("g1", "base")
    try:
      runner = _make_runner("g1", "base", wrapped)
      runner.cfg["num_steps_per_env"] = 3
      self.assertIsNotNone(runner.alg.symmetry)
      runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    finally:
      env.close()

  def test_base_checkpoint_loads_into_finetune_runner(self) -> None:
    for variant in ("paper", "g1"):
      env_base, wrapped_base = _make_env(variant, "base")
      env_finetune, wrapped_finetune = _make_env(variant, "finetune")
      try:
        runner_base = _make_runner(variant, "base", wrapped_base)
        runner_finetune = _make_runner(variant, "finetune", wrapped_finetune)
        with tempfile.TemporaryDirectory() as tmpdir:
          checkpoint_path = Path(tmpdir) / "model_0.pt"
          runner_base.save(str(checkpoint_path))
          runner_finetune.load(str(checkpoint_path), strict=True, map_location="cpu")

        actor_key = "query_proj.weight"
        critic_key = "mlp.0.weight"
        self.assertTrue(
          torch.equal(
            runner_base.alg.actor.state_dict()[actor_key],
            runner_finetune.alg.actor.state_dict()[actor_key],
          )
        )
        self.assertTrue(
          torch.equal(
            runner_base.alg.critic.state_dict()[critic_key],
            runner_finetune.alg.critic.state_dict()[critic_key],
          )
        )
      finally:
        env_base.close()
        env_finetune.close()


if __name__ == "__main__":
  unittest.main()
