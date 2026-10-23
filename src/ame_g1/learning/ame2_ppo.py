# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""AME-2 teacher PPO on RSL-RL 5.4.2.

Adapted from leggedrobotics/ame2_minimal, commit 8beb9a6, rsl_rl/algorithms/ppo.py.
The existing RSL runner/model interfaces remain intact. This extension implements
component value targets, critic-only symmetry and the official optimizer recipe.
"""

from itertools import chain

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_callable, resolve_obs_groups

from ame_g1.learning.ame2_muon import Muon
from ame_g1.learning.ame2_storage import RewardComponentStorage


class AME2PPO(PPO):
  """Feedforward, single-device teacher; actor gradients use summed advantages."""

  def __init__(
    self, actor, critic, storage, *, reward_component_names,
    optimizer="muon", use_mixed_precision=True, ent_coef_decay=0.9999,
    min_ent_coef=0.001, renorm_max_value_loss=100.0, multi_head_critic=True,
    symmetry_cfg=None, **kwargs,
  ):
    if actor.is_recurrent or critic.is_recurrent:
      raise ValueError("AME-2 teacher PPO requires feedforward actor and critic.")
    if kwargs.get("rnd_cfg") or kwargs.get("multi_gpu_cfg"):
      raise ValueError("This teacher implementation supports one device without RND.")
    if not multi_head_critic:
      raise ValueError("AME-2 teacher uses one critic head per reward component.")
    if optimizer != "muon":
      raise ValueError("The AME-2 reference recipe uses the hybrid Muon optimizer.")
    if symmetry_cfg and (
      not symmetry_cfg.get("for_critic_only", False)
      or not symmetry_cfg.get("use_data_augmentation", False)
      or symmetry_cfg.get("use_mirror_loss", False)
    ):
      raise ValueError("Teacher symmetry must augment only the critic, without mirror loss.")
    super().__init__(actor, critic, storage, optimizer="adam", symmetry_cfg=None, **kwargs)
    self.optimizer = Muon(chain(actor.parameters(), critic.parameters()), lr=self.learning_rate)
    self.reward_component_names = tuple(reward_component_names)
    if len(self.reward_component_names) != storage.num_reward_components:
      raise ValueError("Reward names must match the critic/storage head count.")
    self.num_critic_heads = storage.num_reward_components
    self.use_mixed_precision = use_mixed_precision
    self.ent_coef_decay = ent_coef_decay
    self.min_ent_coef = min_ent_coef
    self.renorm_max_value_loss = renorm_max_value_loss
    self.critic_symmetry = dict(symmetry_cfg) if symmetry_cfg else None
    if self.critic_symmetry:
      self.critic_symmetry["data_augmentation_func"] = resolve_callable(
        self.critic_symmetry["data_augmentation_func"]
      )

  def process_env_step(self, obs, rewards, dones, extras):
    self.actor.update_normalization(obs)
    self.critic.update_normalization(obs)
    components = extras["reward_components"]
    if components.shape != (self.storage.num_envs, self.num_critic_heads):
      raise ValueError("reward_components must have shape (num_envs, num_reward_components).")
    self.transition.rewards = components.to(self.device).clone()
    self.transition.dones = dones
    if "time_outs" in extras:
      self.transition.rewards += (
        self.gamma * self.transition.values * extras["time_outs"].to(self.device).view(-1, 1)
      )
    self.storage.add_transition(self.transition)
    self.transition.clear()
    self.actor.reset(dones)
    self.critic.reset(dones)

  @staticmethod
  def _normalize_advantages(advantages):
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)

  def compute_returns(self, obs):
    st = self.storage
    last_values = self.critic(obs).detach()
    advantage = 0
    for step in reversed(range(st.num_transitions_per_env)):
      next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
      not_terminal = 1.0 - st.dones[step].float()
      delta = st.rewards[step] + not_terminal * self.gamma * next_values - st.values[step]
      advantage = delta + not_terminal * self.gamma * self.lam * advantage
      st.returns[step] = advantage + st.values[step]
    st.advantages = (st.returns - st.values).sum(dim=-1, keepdim=True)
    if not self.normalize_advantage_per_mini_batch:
      st.advantages = self._normalize_advantages(st.advantages)

  def update(self):
    totals = {"value": 0.0, "surrogate": 0.0, "entropy": 0.0}
    auxiliary_loss = getattr(self.actor, "auxiliary_loss", None)
    if auxiliary_loss is not None:
      totals["auxiliary"] = 0.0
    generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        batch.advantages = self._normalize_advantages(batch.advantages)
      critic_obs = batch.observations
      old_values, returns = batch.values, batch.returns
      if self.critic_symmetry:
        critic_obs, _ = self.critic_symmetry["data_augmentation_func"](
          env=self.critic_symmetry["_env"], obs=batch.observations,
          actions=None, for_critic=True,
        )
        original_size = batch.observations.batch_size[0]
        if critic_obs.batch_size[0] % original_size:
          raise ValueError("Critic symmetry must produce complete copies of the original batch.")
        copies = critic_obs.batch_size[0] // original_size
        old_values = old_values.repeat(copies, 1)
        returns = returns.repeat(copies, 1)

      with torch.amp.autocast(
        device_type=torch.device(self.device).type, enabled=self.use_mixed_precision,
        dtype=torch.bfloat16,
      ):
        self.actor(batch.observations, stochastic_output=True)
        log_prob = self.actor.get_output_log_prob(batch.actions)
        values = self.critic(critic_obs)
        entropy = self.actor.output_entropy
        if self.desired_kl is not None and self.schedule == "adaptive":
          with torch.inference_mode():
            kl = self.actor.get_kl_divergence(
              batch.old_distribution_params, self.actor.output_distribution_params,
            ).mean()
            if kl > self.desired_kl * 2:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif 0 < kl < self.desired_kl / 2:
              self.learning_rate = min(1e-3, self.learning_rate * 1.5)
            for group in self.optimizer.param_groups:
              group["lr"] = self.learning_rate

        ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
        advantages = batch.advantages.squeeze(-1)
        surrogate_loss = torch.maximum(
          -advantages * ratio,
          -advantages * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
        ).mean()
        value_errors = (values - returns).square()
        if self.use_clipped_value_loss:
          clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
          value_errors = torch.maximum(value_errors, (clipped - returns).square())
        value_loss = value_errors.mean()
        if self.renorm_max_value_loss is not None:
          # Same detached scaling as the reference; clamping the loss itself would erase gradients.
          value_loss = value_loss * (
            self.renorm_max_value_loss / value_loss.detach().clamp_min(self.renorm_max_value_loss)
          )
        loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()
        if auxiliary_loss is not None:
          auxiliary = auxiliary_loss()
          loss = loss + auxiliary

      self.optimizer.zero_grad()
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()
      totals["value"] += value_loss.item()
      totals["surrogate"] += surrogate_loss.item()
      totals["entropy"] += entropy.mean().item()
      if auxiliary_loss is not None:
        totals["auxiliary"] += auxiliary.item()

    count = self.num_learning_epochs * self.num_mini_batches
    self.storage.clear()
    if self.entropy_coef > self.min_ent_coef:
      self.entropy_coef = max(self.entropy_coef * self.ent_coef_decay, self.min_ent_coef)
    return {**{key: value / count for key, value in totals.items()}, "entropy_coef": self.entropy_coef}

  def save(self):
    return {
      **super().save(),
      "ame2_training_state": {
        "entropy_coef": self.entropy_coef,
        "reward_component_names": self.reward_component_names,
      },
    }

  def load(self, loaded_dict, load_cfg=None, strict=True):
    if "ame2_import" in loaded_dict:
      if load_cfg != {"actor": True}:
        raise ValueError("Imported official Gaze weights are actor-only; they cannot resume native training.")
      return super().load(loaded_dict, load_cfg, strict)
    training_state = loaded_dict["ame2_training_state"]
    if (load_cfg is None or load_cfg.get("critic")) and (
      tuple(training_state["reward_component_names"]) != self.reward_component_names
    ):
      raise ValueError("Checkpoint reward-component order differs from this task.")
    resume_iteration = super().load(loaded_dict, load_cfg, strict)
    if load_cfg is None or load_cfg.get("optimizer"):
      self.entropy_coef = training_state["entropy_coef"]
    return resume_iteration

  @classmethod
  def construct_algorithm(cls, obs, env, cfg, device):
    algorithm_cfg = dict(cfg["algorithm"])
    algorithm_cfg.pop("class_name")
    actor_cfg, critic_cfg = dict(cfg["actor"]), dict(cfg["critic"])
    actor_cls = resolve_callable(actor_cfg.pop("class_name"))
    critic_cls = resolve_callable(critic_cfg.pop("class_name"))
    groups = resolve_obs_groups(obs, cfg["obs_groups"], ["actor", "critic"])
    cfg["obs_groups"] = groups
    names = tuple(env.reward_component_names)
    actor = actor_cls(obs, groups, "actor", env.num_actions, **actor_cfg).to(device)
    critic = critic_cls(obs, groups, "critic", len(names), **critic_cfg).to(device)
    storage = RewardComponentStorage(
      env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], len(names), device,
    )
    if algorithm_cfg.get("symmetry_cfg"):
      algorithm_cfg["symmetry_cfg"] = {**algorithm_cfg["symmetry_cfg"], "_env": env}
    algorithm = cls(
      actor, critic, storage, reward_component_names=names, device=device,
      multi_gpu_cfg=cfg.get("multi_gpu"), **algorithm_cfg,
    )
    algorithm.compile(cfg.get("torch_compile_mode"))
    return algorithm
