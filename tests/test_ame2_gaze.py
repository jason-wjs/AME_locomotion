"""Gaze gradients, serialization and the actual PPO auxiliary-loss path."""
import io
import unittest

import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel

from ame_g1.learning.ame2_models import AME2GazeTeacherModel
from ame_g1.learning.ame2_ppo import AME2PPO
from ame_g1.learning.ame2_storage import RewardComponentStorage
from ame_g1.cli.import_gaze import convert_actor_weights


GROUPS = {"actor": ["teacher_prop", "teacher_mapping"], "critic": ["critic"]}


def inputs():
  torch.manual_seed(43)
  return TensorDict({
    "teacher_prop": torch.randn(2, 82),
    "teacher_mapping": torch.randn(2, 26, 15, 3),
    "critic": torch.randn(2, 8),
  }, batch_size=[2])


def actor_for(obs):
  return AME2GazeTeacherModel(obs, GROUPS, "actor", 23, distribution_cfg={
    "class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log",
  })


class GazeTests(unittest.TestCase):
  def test_explicit_joint_conversion_preserves_actions_and_distribution(self):
    obs = inputs()
    source = actor_for(obs)
    source_joints = [f"joint_{i}" for i in range(23)]
    permutation = torch.randperm(23)
    native_joints = [source_joints[i] for i in permutation]
    state = {**source.network.state_dict(), "log_std": torch.linspace(-.5, .5, 23)}
    native = actor_for(obs)
    native.load_state_dict(convert_actor_weights(state, source_joints, native_joints), strict=True)
    native_obs = obs.clone()
    for start in (9, 32, 55):
      native_obs["teacher_prop"][:, start:start + 23] = obs["teacher_prop"][:, start + permutation]
    with torch.inference_mode():
      expected = source(obs)[:, permutation]
      native(native_obs, stochastic_output=True)
      torch.testing.assert_close(native.output_mean, expected)
      torch.testing.assert_close(native.output_std, state["log_std"][permutation].exp().expand(2, -1))

  def test_gaze_learns_through_crop_and_survives_save_and_export(self):
    obs = inputs()
    obs["teacher_prop"].requires_grad_()
    obs["teacher_mapping"].requires_grad_()
    actor = actor_for(obs)
    result = actor(obs)
    result.square().sum().backward()
    self.assertIsNone(obs["teacher_prop"].grad)
    self.assertIsNone(obs["teacher_mapping"].grad)
    grad = actor.network.gaze_head[-1].weight.grad
    self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
    saved = io.BytesIO()
    torch.save(actor.state_dict(), saved)
    saved.seek(0)
    restored = actor_for(obs)
    restored.load_state_dict(torch.load(saved, weights_only=True), strict=True)
    # Export after a gradient-enabled forward must not try to deepcopy its transient graph.
    exported = torch.jit.script(actor.as_jit().eval())
    with torch.inference_mode():
      torch.testing.assert_close(restored(obs), result)
      torch.testing.assert_close(exported(obs["teacher_prop"], obs["teacher_mapping"]), result)

  def test_ppo_roi_loss_moves_gaze_away_from_boundary_without_policy_advantage(self):
    obs = inputs()
    actor = actor_for(obs)
    with torch.no_grad():
      actor.network.gaze_head[-1].weight.zero_()
      actor.network.gaze_head[-1].bias.fill_(4.0)
    critic = MLPModel(obs, GROUPS, "critic", 2, hidden_dims=(8,))
    storage = RewardComponentStorage(2, 2, obs, [23], 2)
    alg = AME2PPO(
      actor, critic, storage, reward_component_names=("a", "b"),
      num_learning_epochs=1, num_mini_batches=1, use_mixed_precision=False,
      schedule="fixed", entropy_coef=0.0, value_loss_coef=0.0,
    )
    with torch.inference_mode():
      for _ in range(2):
        alg.act(obs)
        alg.process_env_step(obs, torch.zeros(2), torch.zeros(2),
                             {"reward_components": torch.zeros(2, 2)})
      alg.compute_returns(obs)
      storage.advantages.zero_()
      actor(obs)
      before = actor.auxiliary_loss().item()
    losses = alg.update()
    self.assertGreater(losses["auxiliary"], 0)
    with torch.inference_mode():
      actor(obs)
      self.assertLess(actor.auxiliary_loss().item(), before)
    imported = {"ame2_import": {"inference_only": True}, "actor_state_dict": actor.state_dict()}
    alg.load(imported, load_cfg={"actor": True})
    with self.assertRaisesRegex(ValueError, "cannot resume native training"):
      alg.load(imported)
