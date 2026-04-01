"""AME-specific on-policy runner hooks and launch helpers."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
import time
from typing import cast

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.scripts.train import TrainConfig
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.velocity.rl.runner import VelocityOnPolicyRunner
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import add_wandb_tags
from mjlab.utils.wrappers import VideoRecorder
from tensordict import TensorDictBase

from ame_g1.config.variants import AmeTaskSpec, DEFAULT_VARIANT, resolve_task_spec


class AmeOnPolicyRunner(VelocityOnPolicyRunner):
  """Thin runner wrapper for AME-owned save hooks."""

  env: RslRlVecEnvWrapper

  @staticmethod
  def _check_nan_value(name: str, value) -> None:
    if isinstance(value, TensorDictBase):
      for key, child in value.items():
        AmeOnPolicyRunner._check_nan_value(f"{name}.{key}", child)
      return
    if isinstance(value, torch.Tensor) and torch.isnan(value).any():
      raise ValueError(
        f"The observation '{name}' returned by the environment contains NaN values. "
        "This usually indicates a bug in the environment's step() or reset() function."
      )

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf,
        high=int(self.env.max_episode_length),
      )

    obs = self.env.get_observations().to(self.device)
    self.alg.train_mode()

    if self.is_distributed:
      print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
      self.alg.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.alg.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          rnd_cfg = self.cfg["algorithm"].get("rnd_cfg")
          if self.cfg.get("check_for_nan", True):
            self._check_nan_value("obs", obs)
            if torch.isnan(rewards).any():
              raise ValueError("The rewards returned by the environment contain NaN values.")
            if torch.isnan(dones).any():
              raise ValueError("The dones returned by the environment contain NaN values.")
          obs, rewards, dones = (
            obs.to(self.device),
            rewards.to(self.device),
            dones.to(self.device),
          )
          self.alg.process_env_step(obs, rewards, dones, extras)
          intrinsic_rewards = self.alg.intrinsic_rewards if rnd_cfg is not None else None
          self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

        stop = time.time()
        collect_time = stop - start
        start = stop
        self.alg.compute_returns(obs)

      loss_dict = self.alg.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        learning_rate=self.alg.learning_rate,
        action_std=self.alg.get_policy().output_std,
        rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"].get("rnd_cfg") is not None else None,
      )

      if self.logger.writer is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore[arg-type]

    if self.logger.writer is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))  # type: ignore[arg-type]
      self.logger.stop_logging_writer()

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    if self.cfg["upload_model"] and getattr(self.logger, "writer", None) is not None:
      self.logger.save_model(path, self.current_learning_iteration)
    policy_path = Path(path).resolve().parent
    os.makedirs(policy_path, exist_ok=True)
    filename = policy_path.name + ".onnx"
    try:
      self.export_policy_to_onnx(str(policy_path), filename)
    except Exception as error:
      print(f"[WARN] ONNX export failed (training continues): {error}")
    actor = getattr(self.alg, "actor", None)
    critic = getattr(self.alg, "critic", None)
    if actor is not None and hasattr(actor, "export_attention_metadata"):
      actor.export_attention_metadata(str(policy_path))
    if critic is not None and hasattr(critic, "export_attention_metadata"):
      critic.export_attention_metadata(
        str(policy_path),
        filename="critic_attention_metadata.json",
      )
    metadata = {
      "runner_cls": type(self).__name__,
      "actor_cls": type(actor).__name__ if actor is not None else None,
      "critic_cls": type(critic).__name__ if critic is not None else None,
    }
    with open(policy_path / "ame_runner_metadata.json", "w", encoding="utf-8") as file:
      json.dump(metadata, file, indent=2, sort_keys=True)


def resolve_device(device: str | None = None) -> str:
  if device is not None:
    return device
  return "cuda:0" if torch.cuda.is_available() else "cpu"


def resolve_checkpoint(
  task_spec: AmeTaskSpec,
  checkpoint_file: str | None,
  wandb_run_path: str | None,
  wandb_checkpoint_name: str | None,
) -> Path:
  agent_cfg = load_rl_cfg(task_spec.task_id)
  log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
  if checkpoint_file is not None:
    path = Path(checkpoint_file).expanduser().resolve()
    if not path.exists():
      raise FileNotFoundError(f"Checkpoint file not found: {path}")
    return path
  if wandb_run_path is None:
    raise ValueError(
      "Either `checkpoint_file` or `wandb_run_path` must be provided for trained policies."
    )
  path, _ = get_wandb_checkpoint_path(
    log_root_path,
    Path(wandb_run_path),
    wandb_checkpoint_name,
  )
  return path


def make_wrapped_play_env(
  task_spec: AmeTaskSpec,
  device: str,
  num_envs: int | None = None,
) -> tuple[ManagerBasedRlEnv, RslRlVecEnvWrapper]:
  env_cfg = load_env_cfg(task_spec.task_id, play=True)
  agent_cfg = load_rl_cfg(task_spec.task_id)
  if num_envs is not None:
    env_cfg.scene.num_envs = num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  return env, wrapped


def load_trained_policy(
  task_spec: AmeTaskSpec,
  env: RslRlVecEnvWrapper,
  device: str,
  checkpoint_path: Path,
):
  agent_cfg = load_rl_cfg(task_spec.task_id)
  runner_cls = load_runner_cls(task_spec.task_id) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), device=device)
  runner.load(
    str(checkpoint_path),
    load_cfg={"actor": True},
    strict=True,
    map_location=device,
  )
  policy = runner.get_inference_policy(device=device)
  return runner, policy


def _sanitize_yaml_value(value):
  if isinstance(value, dict):
    sanitized: dict = {}
    for key, child in value.items():
      if isinstance(key, str) and key.startswith("_"):
        continue
      sanitized[key] = _sanitize_yaml_value(child)
    return sanitized
  if isinstance(value, list):
    return [_sanitize_yaml_value(child) for child in value]
  if isinstance(value, tuple):
    return tuple(_sanitize_yaml_value(child) for child in value)
  if isinstance(value, Path):
    return str(value)
  if isinstance(value, (str, int, float, bool, type(None))):
    return value
  return repr(value)


def run_ame_train(
  task_spec: AmeTaskSpec,
  cfg: TrainConfig,
  log_dir: Path,
  checkpoint_file: Path | None = None,
) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None
  is_tracking_task = "motion" in cfg.env.commands and isinstance(
    cfg.env.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task:
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if motion_cmd.motion_file and Path(motion_cmd.motion_file).exists():
      print(f"[INFO] Using local motion file: {motion_cmd.motion_file}")
    elif cfg.registry_name:
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      raise ValueError(
        "For tracking tasks, provide either:\n"
        "  --registry-name your-org/motions/motion-name\n"
        "  --env.commands.motion.motion-file /path/to/motion.npz"
      )

  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

  env = ManagerBasedRlEnv(
    cfg=cfg.env,
    device=device,
    render_mode="rgb_array" if cfg.video else None,
  )

  log_root_path = log_dir.parent
  resume_path: Path | None = checkpoint_file
  if resume_path is None and cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path,
        Path(cfg.wandb_run_path),
        cfg.wandb_checkpoint_name,
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      resume_path = get_checkpoint_path(
        log_root_path,
        cfg.agent.load_run,
        cfg.agent.load_checkpoint,
      )

  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_spec.task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name
  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", _sanitize_yaml_value(env_cfg))
    dump_yaml(log_dir / "params" / "agent.yaml", _sanitize_yaml_value(agent_cfg))

  runner.learn(num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True)
  env.close()


def launch_ame_training(
  task_spec: AmeTaskSpec,
  args: TrainConfig,
  checkpoint_file: Path | None = None,
) -> None:
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  selected_gpus, num_gpus = select_gpus(args.gpu_ids)
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    run_ame_train(task_spec, args, log_dir, checkpoint_file=checkpoint_file)
    return

  import torchrunx

  logging.basicConfig(level=logging.INFO)
  if "TORCHRUNX_LOG_DIR" not in os.environ:
    if args.torchrunx_log_dir is not None:
      os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
    else:
      os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

  print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
  torchrunx.Launcher(
    hostnames=["localhost"],
    workers_per_host=num_gpus,
    backend=None,
    copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
  ).run(run_ame_train, task_spec, args, log_dir, checkpoint_file)
