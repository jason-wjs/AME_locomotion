"""AME evaluation entrypoint."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Literal

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.lab_api.math import quat_apply_inverse, yaw_quat

import torch
import tyro

import mjlab

from ame_g1 import assert_installed_mjlab
from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import DEFAULT_VARIANT, resolve_task_spec
from ame_g1.learning.ame2_wrapper import RewardComponentVecEnv
from ame_g1.learning.runner import (
  load_trained_policy,
  resolve_checkpoint,
  resolve_device,
)
from ame_g1.physics_guard import install_physics_overflow_guard
from ame_g1.simulation import configure_collision_capacity


@dataclass(frozen=True)
class EvalArgs:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  num_envs: int = 32
  num_episodes: int = 64
  device: str | None = None


@dataclass(frozen=True)
class TaskArgs:
  variant: Literal["paper", "g1", "ame2", "ame2-gaze"] = DEFAULT_VARIANT
  phase: Literal["base", "finetune"] = "base"
  global_context: Literal["on", "off"] = "off"


def _terrain_names(env) -> list[str]:
  terrain = env.scene.terrain
  assert terrain is not None
  generator = terrain.cfg.terrain_generator
  assert generator is not None
  return list(generator.sub_terrains.keys())


def _terrain_name_for_type(terrain_names: list[str], terrain_type: int) -> str:
  return terrain_names[min(max(int(terrain_type), 0), len(terrain_names) - 1)]


def main() -> None:
  assert_installed_mjlab(mjlab)
  register_ame_tasks()
  task_args, remaining_args = tyro.cli(
    TaskArgs,
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )
  args = tyro.cli(
    EvalArgs,
    args=remaining_args,
    prog=(
      f"ame-eval --variant {task_args.variant} "
      f"--phase {task_args.phase} --global-context {task_args.global_context}"
    ),
    config=mjlab.TYRO_FLAGS,
  )

  if task_args.variant == "paper" and task_args.global_context == "on":
    raise ValueError("`--global-context on` is not supported for `--variant paper`.")
  use_global_context = task_args.global_context == "on"
  task_spec = resolve_task_spec(task_args.variant, task_args.phase)
  device = resolve_device(args.device)
  checkpoint_path = resolve_checkpoint(
    task_spec,
    checkpoint_file=args.checkpoint_file,
    wandb_run_path=args.wandb_run_path,
    wandb_checkpoint_name=args.wandb_checkpoint_name,
  )
  env_cfg = load_env_cfg(task_spec.task_id, play=True)
  # Interactive play can be unbounded; evaluation uses the training episode horizon.
  env_cfg.episode_length_s = load_env_cfg(task_spec.task_id).episode_length_s
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.auto_reset = False
  capacities = configure_collision_capacity(task_spec)
  env = ManagerBasedRlEnv(env_cfg, device=device)
  install_physics_overflow_guard(env.sim, task_spec, capacities)
  wrapper_cls = RewardComponentVecEnv if task_spec.variant in ("ame2", "ame2-gaze") else RslRlVecEnvWrapper
  wrapped_env = wrapper_cls(env, clip_actions=load_rl_cfg(task_spec.task_id).clip_actions)
  _, policy = load_trained_policy(
    task_spec,
    wrapped_env,
    device,
    checkpoint_path,
    use_global_context=use_global_context,
  )

  obs = wrapped_env.get_observations()
  robot = wrapped_env.unwrapped.scene["robot"]
  terrain = wrapped_env.unwrapped.scene.terrain
  assert terrain is not None
  terrain_names = _terrain_names(wrapped_env.unwrapped)
  episode_start_xy = robot.data.root_link_pos_w[:, :2].clone()
  terrain_size = terrain.cfg.terrain_generator.size[0]  # type: ignore[union-attr]
  success_by_terrain: dict[str, dict[str, int]] = defaultdict(
    lambda: {"success": 0, "failure": 0, "stuck": 0}
  )
  episode_returns = torch.zeros(args.num_envs, device=device)
  tracking_error_sums = torch.zeros(args.num_envs, device=device)
  episode_steps = torch.zeros(args.num_envs, device=device)
  completed_returns: list[float] = []
  completed_tracking: list[float] = []
  completed = 0

  while completed < args.num_episodes:
    with torch.inference_mode():
      actions = policy(obs)
      terrain_types = terrain.terrain_types.clone()
      if task_spec.variant not in ("ame2", "ame2-gaze"):
        commands = env.command_manager.get_command("twist").clone()
      obs, rewards, dones, _ = wrapped_env.step(actions)
      # auto_reset=False retains the actual final state, including a last-step fall.
      if task_spec.variant in ("ame2", "ame2-gaze"):
        goal = env.command_manager.get_term("base_pose")
        distance = (goal.pos_command_w[:, :2] - robot.data.root_link_pos_w[:, :2]).norm(dim=1)
        success = (distance < .5) & (goal.command[:, 3].abs() < .5)
      else:
        distance = (robot.data.root_link_pos_w[:, :2] - episode_start_xy).norm(dim=1)
        success = distance > terrain_size * .5
        velocity = quat_apply_inverse(
          yaw_quat(robot.data.root_link_quat_w), robot.data.root_link_lin_vel_w
        )
        tracking_error_sums += (commands[:, :2] - velocity[:, :2]).norm(dim=1)
      success &= ~env.reset_terminated
      episode_returns += rewards
      episode_steps += 1
      done_ids = torch.nonzero(dones, as_tuple=False).flatten()
      if len(done_ids) == 0:
        continue
      for env_id in done_ids.tolist():
        terrain_name = _terrain_name_for_type(terrain_names, int(terrain_types[env_id].item()))
        timed_out = bool(env.reset_time_outs[env_id])
        if bool(success[env_id]):
          success_by_terrain[terrain_name]["success"] += 1
        elif timed_out and not bool(env.reset_terminated[env_id]):
          success_by_terrain[terrain_name]["stuck"] += 1
        else:
          success_by_terrain[terrain_name]["failure"] += 1
        completed_returns.append(float(episode_returns[env_id].item()))
        completed_tracking.append(float((tracking_error_sums[env_id] / episode_steps[env_id]).item()))
        episode_returns[env_id] = 0.0
        tracking_error_sums[env_id] = 0.0
        episode_steps[env_id] = 0.0
        completed += 1
        if completed >= args.num_episodes:
          break
      env.reset(env_ids=done_ids)
      episode_start_xy[done_ids] = robot.data.root_link_pos_w[done_ids, :2]
      obs = wrapped_env.get_observations()

  avg_return = sum(completed_returns) / max(len(completed_returns), 1)
  avg_tracking = sum(completed_tracking) / max(len(completed_tracking), 1)
  print(f"checkpoint={checkpoint_path}")
  print(f"episodes={completed}")
  print(f"avg_episode_return={avg_return:.4f}")
  if task_spec.variant not in ("ame2", "ame2-gaze"):
    print(f"avg_tracking_error_mps={avg_tracking:.4f}")
  for terrain_name in terrain_names:
    stats = success_by_terrain[terrain_name]
    total = sum(stats.values())
    if total == 0:
      continue
    print(
      f"{terrain_name}: success={stats['success']} failure={stats['failure']} "
      f"stuck={stats['stuck']} total={total}"
    )

  env.close()


if __name__ == "__main__":
  main()
