"""Teacher inference/export and multi-reward value semantics."""

import io
import math
import unittest

import torch
from tensordict import TensorDict

from ame_g1.learning.ame2_models import AME2MoECritic, AME2TeacherModel


def _observations():
  torch.manual_seed(7)
  return TensorDict({
    "teacher_prop": torch.randn(3, 17),
    "teacher_mapping": torch.randn(3, 5, 4, 4),
    "critic": torch.randn(3, 23),
  }, batch_size=[3])


GROUPS = {"actor": ["teacher_prop", "teacher_mapping"], "critic": ["critic"]}


class TeacherModelTests(unittest.TestCase):
  def test_teacher_distribution_resume_and_jit_agree_with_inference(self):
    obs = _observations()
    cfg = {"class_name": "GaussianDistribution", "init_std": 0.7, "std_type": "log"}
    actor = AME2TeacherModel(obs, GROUPS, "actor", 6, distribution_cfg=cfg)
    mean = actor(obs)
    actions = actor(obs, stochastic_output=True)
    torch.testing.assert_close(actor.output_mean, mean)
    torch.testing.assert_close(actor.output_std, torch.full_like(mean, 0.7))
    expected = torch.distributions.Normal(mean, 0.7).log_prob(actions).sum(-1)
    torch.testing.assert_close(actor.get_output_log_prob(actions), expected)
    params = actor.output_distribution_params
    torch.testing.assert_close(actor.get_kl_divergence(params, params), torch.zeros(3))

    # Distribution parameters must survive the same state_dict checkpoint path as weights.
    with torch.no_grad():
      actor.distribution.log_std_param.add_(0.2)
    checkpoint = io.BytesIO()
    torch.save(actor.state_dict(), checkpoint)
    checkpoint.seek(0)
    restored = AME2TeacherModel(obs, GROUPS, "actor", 6, distribution_cfg=cfg)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored(obs, stochastic_output=True)
    torch.testing.assert_close(restored.output_mean, mean)
    torch.testing.assert_close(restored.output_std, torch.full_like(mean, 0.7 * math.exp(0.2)))
    exported = torch.jit.script(restored.as_jit().eval())
    with torch.no_grad():
      torch.testing.assert_close(exported(obs["teacher_prop"], obs["teacher_mapping"]), mean)


  def test_teacher_gradients_stop_at_observations_but_reach_both_map_paths(self):
    obs = _observations()
    obs["teacher_prop"].requires_grad_()
    obs["teacher_mapping"].requires_grad_()
    actor = AME2TeacherModel(obs, GROUPS, "actor", 6)
    actor(obs).square().sum().backward()
    assert obs["teacher_prop"].grad is None
    assert obs["teacher_mapping"].grad is None
    for parameter in (actor.network.fc_part[0].weight, actor.network.cnn_part[0].weight,
                      actor.network.mha.in_proj_weight, actor.network.global_pool_mlp.net[0].weight):
      assert parameter.grad is not None
      assert torch.isfinite(parameter.grad).all()
      assert parameter.grad.abs().sum() > 0


  def test_moe_mixes_each_reward_head_and_trains_router(self):
    obs = _observations()
    critic = AME2MoECritic(obs, GROUPS, "critic", output_dim=3, num_experts=2)
    # Input-independent experts isolate the mixture semantics from MLP implementation.
    with torch.no_grad():
      for parameter in critic.parameters():
        parameter.zero_()
      critic.mlp.experts[0].net[-1].bias.copy_(torch.tensor([1.0, 3.0, -2.0]))
      critic.mlp.experts[1].net[-1].bias.copy_(torch.tensor([5.0, -1.0, 6.0]))
      critic.mlp.router.net[-1].bias.copy_(torch.tensor([0.0, math.log(3.0)]))
    value = critic(obs)
    torch.testing.assert_close(value, torch.tensor([[4.0, 0.0, 4.0]]).expand(3, -1))
    value[:, 0].sum().backward()
    assert critic.mlp.router.net[-1].bias.grad.abs().sum() > 0
    for expert in critic.mlp.experts:
      assert expert.net[-1].bias.grad[0] != 0
    exported = torch.jit.script(critic.as_jit())
    torch.testing.assert_close(exported(obs["critic"]), value)
