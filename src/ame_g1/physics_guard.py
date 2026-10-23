"""Stop on physics buffer overflow and retain bounded diagnostic state."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from ame_g1.config.variants import AmeTaskSpec

if TYPE_CHECKING:
  from mjlab.sim import Simulation


# One frame on the simulation device; no growing history or normal-step disk I/O.
_STATE_FIELDS = (
  "time", "qpos", "qvel", "act", "history", "ctrl", "qacc_warmstart",
  "qfrc_applied", "xfrc_applied", "eq_active", "mocap_pos", "mocap_quat",
  "userdata", "tree_asleep",
)
_CONTACT_FIELDS = ("worldid", "geom", "dist", "pos", "frame")


class PhysicsOverflowError(RuntimeError):
  """A physics step exceeded capacity; its results must not reach learning."""


class PhysicsOverflowGuard:
  def __init__(
    self,
    sim: Simulation,
    task_spec: AmeTaskSpec,
    output_dir: Path,
    capacities: dict[str, int],
  ) -> None:
    self.sim = sim
    self.task_spec = task_spec
    self.output_dir = output_dir
    self.capacities = dict(capacities)
    self._original_step = sim.step
    # mjlab 1.6 cannot bridge zero-length Warp vec3/quat mocap arrays.
    self._state_fields = tuple(
      name for name in _STATE_FIELDS
      if sim.mj_model.nmocap > 0 or name not in ("mocap_pos", "mocap_quat")
    )
    self._before: dict[str, torch.Tensor] = {}
    self.step_count = 0
    self._failed = False

  def step(self) -> None:
    if self._failed:
      raise PhysicsOverflowError("Simulation already failed with a physics overflow.")
    for name in self._state_fields:
      value = getattr(self.sim.data, name)
      if name not in self._before:
        self._before[name] = value.clone()
      else:
        self._before[name].copy_(value)
    self.step_count += 1
    try:
      self._original_step()
    finally:
      # Check even when another guard (e.g. NaN guard) also raises on this step.
      self._check_overflow()

  def _check_overflow(self) -> None:
    overflow = self.sim.data.overflow
    if not bool(torch.any(overflow != 0)):
      return
    self._failed = True
    flags = overflow.detach().cpu().numpy().copy()
    env_ids = np.flatnonzero(flags)
    message = (
      f"Physics buffer overflow at guarded step {self.step_count}: "
      f"{len(env_ids)} environments, first IDs={env_ids[:5].tolist()}, "
      f"flags={flags[env_ids[:5]].tolist()}."
    )
    try:
      dump_dir = self._dump(flags, env_ids)
    except Exception as error:
      # Evidence failures must never allow invalid transitions to continue.
      raise PhysicsOverflowError(f"{message} Diagnostic dump failed: {error}") from error
    raise PhysicsOverflowError(f"{message} Diagnostic snapshot: {dump_dir}")

  def _dump(self, flags: np.ndarray, env_ids: np.ndarray) -> Path:
    self.output_dir.mkdir(parents=True, exist_ok=True)
    dump_dir = Path(tempfile.mkdtemp(prefix="overflow_", dir=self.output_dir))
    selected = env_ids[:5]
    metadata = {
      "schema_version": 1,
      "task_id": self.task_spec.task_id,
      "guarded_physics_steps": self.step_count,
      "overflow_env_ids": env_ids.tolist(),
      "overflow_flags": flags[env_ids].tolist(),
      "snapshot_env_ids": selected.tolist(),
      "effective_collision_capacities": self.capacities,
      "state_fields": list(self._state_fields),
      "expanded_model_fields": sorted(self.sim.expanded_fields),
      "dump_complete": False,
      "scope": (
        "Physics diagnostics: before/after step state, base model, expanded model "
        "fields and available post-step contacts. Contacts may be incomplete after "
        "overflow. This is not a training checkpoint or a guarantee of exact replay."
      ),
    }
    metadata_path = dump_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    ids = torch.as_tensor(selected, device=self._before["qpos"].device)
    arrays = {"snapshot_env_ids": selected, "overflow": flags}
    for name in self._state_fields:
      arrays[f"before_{name}"] = self._before[name][ids].detach().cpu().numpy()
      arrays[f"after_{name}"] = getattr(self.sim.data, name)[ids].detach().cpu().numpy()
    # Domain randomization expands selected model fields per world. Retain the
    # live values as the base MJB alone does not describe those environments.
    for name in sorted(self.sim.expanded_fields):
      arrays[f"model_{name}"] = getattr(self.sim.model, name)[ids].detach().cpu().numpy()
    contact = self.sim.data.contact
    nacon = int(self.sim.data.nacon[0].item())
    contact_count = min(nacon, contact.worldid.shape[0])
    world_ids = contact.worldid[:contact_count]
    mask = torch.isin(world_ids, ids)
    for name in _CONTACT_FIELDS:
      arrays[f"contact_{name}"] = getattr(contact, name)[:contact_count][mask].detach().cpu().numpy()
    metadata["reported_contacts_all_envs"] = nacon
    metadata["retained_contacts"] = len(arrays["contact_worldid"])
    np.savez_compressed(dump_dir / "state.npz", **arrays)
    mujoco.mj_saveModel(self.sim.mj_model, str(dump_dir / "model.mjb"))
    metadata["dump_complete"] = True
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return dump_dir


def install_physics_overflow_guard(
  sim: Simulation,
  task_spec: AmeTaskSpec,
  capacities: dict[str, int],
  output_dir: Path | None = None,
) -> PhysicsOverflowGuard | None:
  """Attach to this instance only, before the first rollout or evaluation step."""
  if task_spec.variant not in ("g1", "ame2-gaze") or task_spec.phase != "base":
    return None
  if output_dir is None:
    output_dir = Path("logs") / task_spec.experiment_name / "physics_failures"
  guard = PhysicsOverflowGuard(sim, task_spec, output_dir, capacities)
  sim.step = guard.step
  return guard
