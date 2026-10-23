"""Expose the ordered reward components used by the teacher value heads."""
from mjlab.rl import RslRlVecEnvWrapper


class RewardComponentVecEnv(RslRlVecEnvWrapper):
  @property
  def reward_component_names(self):
    return tuple(self.unwrapped.reward_manager.active_terms)

  def step(self, actions):
    obs, rewards, dones, extras = super().step(actions)
    # mjlab 1.6 stores weighted reward rates, whereas the scalar return includes dt.
    # reset() preserves _step_reward, including rows that ended on this transition.
    extras["reward_components"] = (
      self.unwrapped.reward_manager._step_reward * self.unwrapped.step_dt
    )
    return obs, rewards, dones, extras
