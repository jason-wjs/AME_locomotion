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


def _make_env(variant: str, stage: str) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  task_id = resolve_task_spec(variant, stage).task_id
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)
  env_cfg.scene.num_envs = 1
  env = ManagerBasedRlEnv(cfg=env_cfg, device="cpu", render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return env, wrapped


def _make_runner(variant: str, stage: str, wrapped: RslRlVecEnvWrapper):
  task_id = resolve_task_spec(variant, stage).task_id
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

  def test_stage_observation_shapes(self) -> None:
    expected_shapes = {"paper": (17, 11, 3), "g1": (33, 21, 3)}
    for variant in ("paper", "g1"):
      for stage in ("stage1", "stage2"):
        env, wrapped = _make_env(variant, stage)
        try:
          obs = wrapped.get_observations()
          self.assertEqual(
            set(obs.keys()),
            {"actor_proprio", "actor_terrain", "critic_proprio", "critic_terrain"},
          )
          self.assertEqual(tuple(obs["actor_proprio"].shape), (1, 99))
          self.assertEqual(tuple(obs["critic_proprio"].shape), (1, 99))
          self.assertEqual(tuple(obs["actor_terrain"].shape), (1, *expected_shapes[variant]))
          self.assertEqual(tuple(obs["critic_terrain"].shape), (1, *expected_shapes[variant]))
          obs, _, _, _ = wrapped.step(torch.zeros(1, wrapped.num_actions))
          self.assertEqual(tuple(obs["actor_terrain"].shape), (1, *expected_shapes[variant]))
        finally:
          env.close()

  def test_actor_emits_attention_weights(self) -> None:
    for variant in ("paper", "g1"):
      env, wrapped = _make_env(variant, "stage1")
      try:
        runner = _make_runner(variant, "stage1", wrapped)
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

  def test_stage1_symmetry_update_smoke(self) -> None:
    env, wrapped = _make_env("g1", "stage1")
    try:
      runner = _make_runner("g1", "stage1", wrapped)
      runner.cfg["num_steps_per_env"] = 3
      self.assertIsNotNone(runner.alg.symmetry)
      runner.learn(num_learning_iterations=1, init_at_random_ep_len=False)
    finally:
      env.close()

  def test_stage1_checkpoint_loads_into_stage2_runner(self) -> None:
    for variant in ("paper", "g1"):
      env_stage1, wrapped_stage1 = _make_env(variant, "stage1")
      env_stage2, wrapped_stage2 = _make_env(variant, "stage2")
      try:
        runner_stage1 = _make_runner(variant, "stage1", wrapped_stage1)
        runner_stage2 = _make_runner(variant, "stage2", wrapped_stage2)
        with tempfile.TemporaryDirectory() as tmpdir:
          checkpoint_path = Path(tmpdir) / "model_0.pt"
          runner_stage1.save(str(checkpoint_path))
          runner_stage2.load(str(checkpoint_path), strict=True, map_location="cpu")

        actor_key = "query_proj.weight"
        critic_key = "mlp.0.weight"
        self.assertTrue(
          torch.equal(
            runner_stage1.alg.actor.state_dict()[actor_key],
            runner_stage2.alg.actor.state_dict()[actor_key],
          )
        )
        self.assertTrue(
          torch.equal(
            runner_stage1.alg.critic.state_dict()[critic_key],
            runner_stage2.alg.critic.state_dict()[critic_key],
          )
        )
      finally:
        env_stage1.close()
        env_stage2.close()


if __name__ == "__main__":
  unittest.main()
