"""AME play entrypoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

import mjlab
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

from ame_g1 import assert_installed_mjlab
from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import DEFAULT_VARIANT, resolve_task_spec
from ame_g1.learning.runner import (
  load_trained_policy,
  make_wrapped_play_env,
  resolve_checkpoint,
  resolve_device,
)


@dataclass(frozen=True)
class AmePlayArgs:
  checkpoint_file: str | None = None
  wandb_run_path: str | None = None
  wandb_checkpoint_name: str | None = None
  num_envs: int = 1
  num_steps: int = 400
  device: str | None = None
  viewer: bool = False
  viewer_backend: str = "auto"
  dump_file: str | None = None
  dump_every: int = 1


@dataclass(frozen=True)
class TaskArgs:
  variant: Literal["paper", "g1"] = DEFAULT_VARIANT
  phase: Literal["base", "finetune"] = "base"
  global_context: Literal["on", "off"] = "off"


def _resolve_viewer_backend(requested: str) -> str:
  if requested != "auto":
    return requested
  has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
  return "native" if has_display else "viser"


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
    AmePlayArgs,
    args=remaining_args,
    prog=(
      f"ame-play --variant {task_args.variant} "
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
  env, wrapped_env = make_wrapped_play_env(task_spec, device=device, num_envs=args.num_envs)
  _, policy = load_trained_policy(
    task_spec,
    wrapped_env,
    device,
    checkpoint_path,
    use_global_context=use_global_context,
  )

  if args.viewer:
    viewer_backend = _resolve_viewer_backend(args.viewer_backend)
    if viewer_backend == "native":
      NativeMujocoViewer(wrapped_env, policy).run()
    elif viewer_backend == "viser":
      ViserPlayViewer(wrapped_env, policy).run()
    else:
      raise RuntimeError(f"Unsupported viewer backend: {viewer_backend}")
    env.close()
    return

  obs = wrapped_env.get_observations()
  robot = wrapped_env.unwrapped.scene["robot"]
  dump_steps: list[dict[str, torch.Tensor | None]] = []
  for step in range(args.num_steps):
    with torch.inference_mode():
      actions = policy(obs)
      if args.dump_file is not None and step % max(args.dump_every, 1) == 0:
        attention = getattr(policy, "last_attention_weights", None)
        attention_points = getattr(policy, "last_attention_points", None)
        dump_steps.append(
          {
            "terrain_points": obs["actor_terrain"].detach().cpu(),
            "attention_points": attention_points.detach().cpu() if attention_points is not None else None,
            "attention_weights": attention.detach().cpu() if attention is not None else None,
            "base_pos_w": robot.data.root_link_pos_w.detach().cpu(),
            "base_quat_w": robot.data.root_link_quat_w.detach().cpu(),
            "actions": actions.detach().cpu(),
          }
        )
      obs, _, _, _ = wrapped_env.step(actions)

  if args.dump_file is not None:
    output_path = Path(args.dump_file).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
      {
        "variant": task_args.variant,
        "phase": task_args.phase,
        "global_context": task_args.global_context,
        "checkpoint": str(checkpoint_path),
        "encoder_variant": getattr(policy, "encoder_variant", None),
        "attach_global": getattr(policy, "attach_global", None),
        "downsampled_shape": list(getattr(policy, "attention_shape", ())),
        "steps": dump_steps,
      },
      output_path,
    )
    print(f"[INFO] Saved attention dump to {output_path}")

  env.close()


if __name__ == "__main__":
  main()
