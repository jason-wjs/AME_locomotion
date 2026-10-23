"""Small real-simulator teacher check; select the test device explicitly."""
from dataclasses import asdict
import os
from pathlib import Path
import tempfile
import unittest

import torch
import mjlab
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg

from ame_g1.config import register
from ame_g1.learning.runner import AmeOnPolicyRunner
from ame_g1.learning.ame2_wrapper import RewardComponentVecEnv


@unittest.skipUnless(os.environ.get("AME_TEST_DEVICE"), "set AME_TEST_DEVICE for simulator integration")
class TeacherIntegrationTests(unittest.TestCase):
  task_id = "AME2-G1-Teacher"
  action_dim = 29

  def test_rollout_update_reset_and_resume(self):
    register()
    device = os.environ["AME_TEST_DEVICE"]
    cfg = load_env_cfg(self.task_id)
    cfg.scene.num_envs = 4
    terrain = cfg.scene.terrain.terrain_generator
    terrain.num_rows = 1
    terrain.num_cols = 1
    terrain.sub_terrains = {"rough": terrain.sub_terrains["rough"]}
    cfg.scene.terrain.max_init_terrain_level = 0
    env = ManagerBasedRlEnv(cfg, device=device)
    try:
      wrapped = RewardComponentVecEnv(env)
      obs = wrapped.get_observations()
      self.assertEqual(tuple(obs["teacher_mapping"].shape), (4, 26, 15, 3))
      self.assertEqual(tuple(obs["teacher_prop"].shape), (4, 3 * self.action_dim + 13))
      self.assertEqual(len(wrapped.reward_component_names), 23)
      actions = torch.zeros(4, self.action_dim, device=device)
      env.episode_length_buf[0] = env.max_episode_length - 1
      obs, reward, done, extras = wrapped.step(actions)
      self.assertTrue(done[0])
      torch.testing.assert_close(extras["reward_components"].sum(-1), reward)
      for value in obs.values():
        self.assertTrue(torch.isfinite(value).all())
      other_pos = env.scene["robot"].data.root_link_pos_w[1:].clone()
      env.reset(env_ids=torch.tensor([0], device=device))
      torch.testing.assert_close(env.scene["robot"].data.root_link_pos_w[1:], other_pos)
      agent = asdict(load_rl_cfg(self.task_id))
      agent["num_steps_per_env"] = 6
      agent["algorithm"]["num_learning_epochs"] = 1
      agent["algorithm"]["num_mini_batches"] = 1
      runner = AmeOnPolicyRunner(wrapped, agent, device=device)
      before = next(runner.alg.actor.parameters()).detach().clone()
      runner.learn(2)
      self.assertFalse(torch.equal(before, next(runner.alg.actor.parameters())))
      for module in (runner.alg.actor, runner.alg.critic):
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))
      with tempfile.TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "teacher.pt"
        runner.save(str(checkpoint), export_artifacts=False)
        entropy = runner.alg.entropy_coef
        obs = wrapped.get_observations()
        with torch.inference_mode():
          expected = runner.get_inference_policy()(obs).clone()
        resumed = AmeOnPolicyRunner(wrapped, agent, device=device)
        resumed.load(str(checkpoint))
        self.assertEqual(resumed.alg.entropy_coef, entropy)
        with torch.inference_mode():
          torch.testing.assert_close(resumed.get_inference_policy()(obs), expected)
        resumed.learn(1)
        resumed.export_policy_to_jit(directory)
        exported = torch.jit.load(str(Path(directory) / "policy.pt"), map_location=device)
        with torch.inference_mode():
          torch.testing.assert_close(exported(obs["teacher_prop"], obs["teacher_mapping"]), resumed.get_inference_policy()(obs))
      print("AME2_INTEGRATION_PASS", {key: tuple(value.shape) for key, value in obs.items()})
    finally:
      env.close()


class GazeIntegrationTests(TeacherIntegrationTests):
  task_id = "AME2-G1-Gaze-Teacher"
  action_dim = 23


if __name__ == "__main__":
  unittest.main()
