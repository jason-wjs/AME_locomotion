"""Teacher learning contracts: component targets, symmetry and exact resume."""

import copy
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from ame_g1.learning.ame2_ppo import AME2PPO
from ame_g1.learning.ame2_storage import RewardComponentStorage


def make_algorithm(**kwargs):
  obs = TensorDict({"policy": torch.randn(2, 3)}, batch_size=[2])
  groups = {"actor": ["policy"], "critic": ["policy"]}
  actor = MLPModel(
    obs, groups, "actor", 2, hidden_dims=(8,),
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": 0.5},
  )
  critic = MLPModel(obs, groups, "critic", 2, hidden_dims=(8,))
  storage = RewardComponentStorage(2, 2, obs, [2], 2)
  options = dict(
    reward_component_names=("progress", "effort"), num_learning_epochs=1,
    num_mini_batches=1, use_mixed_precision=False, schedule="fixed",
    entropy_coef=0.01, ent_coef_decay=0.5, min_ent_coef=0.001,
  )
  options.update(kwargs)
  return AME2PPO(actor, critic, storage, **options), obs


def collect(algorithm, obs):
  with torch.inference_mode():
    for index in range(2):
      algorithm.act(obs)
      components = torch.tensor([[1.0, -0.1], [0.5, -0.2]]) * (index + 1)
      algorithm.process_env_step(
        obs, components.sum(-1), torch.zeros(2), {"reward_components": components},
      )
    algorithm.compute_returns(obs)


class TeacherPPOTests(unittest.TestCase):
  def test_component_gae_sums_to_scalar_gae(self):
    algorithm, obs = make_algorithm(normalize_advantage_per_mini_batch=True, gamma=0.9, lam=0.8)
    st = algorithm.storage
    st.rewards.copy_(torch.tensor([[[1., -0.2], [2., -0.3]], [[3., -0.4], [4., -0.5]]]))
    st.values.copy_(torch.tensor([[[0.5, -0.1], [1., -0.2]], [[0.7, -0.2], [1.2, -0.1]]]))
    st.dones[1, 1] = 1
    last = torch.tensor([[0.8, -0.2], [1.1, -0.1]])
    with patch.object(algorithm.critic, "forward", return_value=last):
      algorithm.compute_returns(obs)
    scalar_adv = torch.zeros(2)
    expected = torch.zeros(2, 2)
    for step in (1, 0):
      next_values = last.sum(-1) if step == 1 else st.values[1].sum(-1)
      mask = 1 - st.dones[step, :, 0].float()
      delta = st.rewards[step].sum(-1) + mask * 0.9 * next_values - st.values[step].sum(-1)
      scalar_adv = delta + mask * 0.9 * 0.8 * scalar_adv
      expected[step] = scalar_adv
    torch.testing.assert_close(st.advantages[..., 0], expected)
    torch.testing.assert_close(st.returns[1, 1], st.rewards[1, 1])
    self.assertEqual(st.returns.shape, (2, 2, 2))
    self.assertEqual(st.advantages.shape, (2, 2, 1))

  def test_timeout_bootstraps_each_head_without_mutating_extras(self):
    algorithm, obs = make_algorithm(gamma=0.9)
    components = torch.tensor([[1., -0.1], [2., -0.2]])
    original = components.clone()
    with torch.inference_mode():
      algorithm.act(obs)
      values = algorithm.transition.values.clone()
      algorithm.process_env_step(
        obs, components.sum(-1), torch.ones(2),
        {"reward_components": components, "time_outs": torch.tensor([True, False])},
      )
    expected = original.clone()
    expected[0] += 0.9 * values[0]
    torch.testing.assert_close(algorithm.storage.rewards[0], expected)
    torch.testing.assert_close(components, original)

  def test_update_only_augments_critic_and_resume_preserves_next_update(self):
    def mirror(*, obs, actions, env, for_critic):
      self.assertTrue(for_critic)
      self.assertIsNone(actions)
      reflected = obs.clone()
      reflected["policy"] = -reflected["policy"]
      return torch.cat([obs, reflected]), None

    symmetry = dict(
      for_critic_only=True, use_data_augmentation=True, use_mirror_loss=False,
      data_augmentation_func=mirror, _env=None,
    )
    algorithm, obs = make_algorithm(symmetry_cfg=symmetry)
    collect(algorithm, obs)
    actor_sizes, critic_sizes = [], []
    ah = algorithm.actor.register_forward_pre_hook(lambda _, args: actor_sizes.append(args[0].batch_size[0]))
    ch = algorithm.critic.register_forward_pre_hook(lambda _, args: critic_sizes.append(args[0].batch_size[0]))
    loss = algorithm.update()
    ah.remove()
    ch.remove()
    self.assertEqual(actor_sizes, [4])
    self.assertEqual(critic_sizes, [8])
    self.assertEqual(loss["entropy_coef"], 0.005)
    self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in loss.values()))
    self.assertTrue(any("momentum_buffer" in state for state in algorithm.optimizer.state.values()))
    self.assertTrue(any("exp_avg" in state for state in algorithm.optimizer.state.values()))

    algorithm.learning_rate = 0.0002
    for group in algorithm.optimizer.param_groups:
      group["lr"] = algorithm.learning_rate
    checkpoint = copy.deepcopy(algorithm.save())
    resumed, _ = make_algorithm(symmetry_cfg=symmetry)
    self.assertTrue(resumed.load(checkpoint))
    self.assertEqual(resumed.learning_rate, 0.0002)
    self.assertEqual(resumed.entropy_coef, 0.005)
    for model in (algorithm, resumed):
      torch.manual_seed(51)
      collect(model, obs)
      model.update()
    for expected, actual in zip(algorithm.actor.parameters(), resumed.actor.parameters()):
      torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(algorithm.critic.parameters(), resumed.critic.parameters()):
      torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    bad = copy.deepcopy(checkpoint)
    bad["ame2_training_state"]["reward_component_names"] = ("effort", "progress")
    with self.assertRaisesRegex(ValueError, "order"):
      resumed.load(bad)

  def test_bfloat16_update_keeps_parameters_float32(self):
    algorithm, obs = make_algorithm(use_mixed_precision=True)
    collect(algorithm, obs)
    losses = algorithm.update()
    self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in losses.values()))
    self.assertTrue(all(p.dtype == torch.float32 for p in algorithm.actor.parameters()))


if __name__ == "__main__":
  unittest.main()
