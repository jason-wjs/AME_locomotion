from __future__ import annotations

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from ame_g1.config.variants import resolve_task_spec
from ame_g1.physics_guard import PhysicsOverflowError, install_physics_overflow_guard


class FakeSimulation:
  def __init__(self) -> None:
    self.calls = 0
    self.fail = False
    self.data = SimpleNamespace(
      time=torch.zeros(2), qpos=torch.tensor([[0., 1.], [2., 3.]]),
      qvel=torch.zeros(2, 2), ctrl=torch.tensor([[1.], [4.]]),
      qacc_warmstart=torch.ones(2, 2), act=torch.zeros(2, 0),
      history=torch.zeros(2, 0), qfrc_applied=torch.zeros(2, 2),
      xfrc_applied=torch.zeros(2, 1, 6), eq_active=torch.zeros(2, 0, dtype=torch.bool),
      mocap_pos=torch.zeros(2, 0, 3), mocap_quat=torch.zeros(2, 0, 4),
      userdata=torch.zeros(2, 0), tree_asleep=torch.zeros(2, 1, dtype=torch.int32),
      overflow=torch.zeros(2, dtype=torch.int32), nacon=torch.tensor([2]),
      contact=SimpleNamespace(
        worldid=torch.tensor([0, 1, 1]), geom=torch.tensor([[0, 1], [1, 2], [8, 9]]),
        dist=torch.tensor([-.1, -.2, -999.]), pos=torch.zeros(3, 3),
        frame=torch.zeros(3, 3, 3),
      ),
    )
    self.expanded_fields = {"body_mass"}
    self.model = SimpleNamespace(body_mass=torch.tensor([[1.], [2.]]))
    self.mj_model = mujoco.MjModel.from_xml_string(
      '<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'
    )

  def step(self) -> None:
    self.calls += 1
    self.data.qpos.add_(10)
    self.data.time.add_(.002)
    if self.fail:
      self.data.overflow[1] = 256


class PhysicsGuardTests(unittest.TestCase):
  def setUp(self) -> None:
    self.temp = tempfile.TemporaryDirectory()
    self.addCleanup(self.temp.cleanup)
    self.output = Path(self.temp.name) / "failures"
    self.sim = FakeSimulation()
    self.spec = resolve_task_spec("ame2-gaze", "base")
    self.guard = install_physics_overflow_guard(
      self.sim, self.spec, {"test_capacity": 64}, self.output,
    )

  def test_healthy_step_has_no_disk_output_or_model_change(self) -> None:
    self.sim.step()
    self.sim.step()
    self.assertEqual(self.sim.calls, 2)
    self.assertFalse(self.output.exists())
    torch.testing.assert_close(self.sim.data.qpos, torch.tensor([[20., 21.], [22., 23.]]))
    torch.testing.assert_close(self.sim.model.body_mass, torch.tensor([[1.], [2.]]))

  def test_overflow_saves_last_before_and_after_state_and_stops(self) -> None:
    self.sim.step()
    self.sim.fail = True
    with self.assertRaisesRegex(PhysicsOverflowError, "flags=\\[256\\]"):
      self.sim.step()
    dump = next(self.output.iterdir())
    metadata = json.loads((dump / "metadata.json").read_text())
    self.assertTrue(metadata["dump_complete"])
    self.assertEqual(metadata["guarded_physics_steps"], 2)
    self.assertEqual(metadata["snapshot_env_ids"], [1])
    with np.load(dump / "state.npz", allow_pickle=False) as state:
      np.testing.assert_array_equal(state["overflow"], [0, 256])
      np.testing.assert_array_equal(state["before_qpos"], [[12, 13]])
      np.testing.assert_array_equal(state["after_qpos"], [[22, 23]])
      np.testing.assert_array_equal(state["before_ctrl"], [[4]])
      np.testing.assert_array_equal(state["model_body_mass"], [[2]])
      np.testing.assert_array_equal(state["contact_geom"], [[1, 2]])
    restored = mujoco.MjModel.from_binary_path(str(dump / "model.mjb"))
    self.assertEqual(restored.ngeom, self.sim.mj_model.ngeom)
    with self.assertRaisesRegex(PhysicsOverflowError, "already failed"):
      self.sim.step()
    self.assertEqual(self.sim.calls, 2)

  def test_dump_failure_still_stops_simulation(self) -> None:
    self.sim.fail = True
    with patch("ame_g1.physics_guard.np.savez_compressed", side_effect=OSError("disk full")):
      with self.assertRaisesRegex(PhysicsOverflowError, "Diagnostic dump failed: disk full"):
        self.sim.step()
    dump = next(self.output.iterdir())
    self.assertFalse(json.loads((dump / "metadata.json").read_text())["dump_complete"])
    with self.assertRaises(PhysicsOverflowError):
      self.sim.step()
    self.assertEqual(self.sim.calls, 1)

  def test_instance_isolation_and_unaffected_task(self) -> None:
    other = FakeSimulation()
    self.assertIsNone(install_physics_overflow_guard(
      other, resolve_task_spec("paper", "base"), {}, self.output,
    ))
    self.sim.fail = other.fail = True
    other.step()
    self.assertEqual(other.calls, 1)
    self.assertEqual(self.sim.calls, 0)
    self.assertFalse(self.output.exists())

  def test_simulation_exception_is_preserved_without_overflow(self) -> None:
    def fail() -> None:
      raise RuntimeError("other physics failure")
    self.guard._original_step = fail
    with self.assertRaisesRegex(RuntimeError, "other physics failure"):
      self.sim.step()
    self.assertFalse(self.output.exists())

  def test_overflow_dump_survives_another_guard_exception(self) -> None:
    original = self.guard._original_step
    def fail() -> None:
      original()
      raise RuntimeError("NaN guard")
    self.guard._original_step = fail
    self.sim.fail = True
    with self.assertRaises(PhysicsOverflowError):
      self.sim.step()
    self.assertTrue(json.loads((next(self.output.iterdir()) / "metadata.json").read_text())["dump_complete"])


class PhysicsGuardCpuIntegrationTests(unittest.TestCase):
  def test_real_bridge_step_and_injected_fault_archive(self) -> None:
    from mjlab.sim import Simulation, SimulationCfg

    model = mujoco.MjModel.from_xml_string('''
      <mujoco><worldbody><body pos="0 0 1"><freejoint/>
        <geom type="sphere" size=".1"/>
      </body></worldbody></mujoco>
    ''')
    sim = Simulation(2, SimulationCfg(nconmax=16, njmax=64), model, device="cpu")
    sim.expand_model_fields(("body_mass",))
    sim.model.body_mass[1, 1] = 7.
    # Injection tests the real bridge and archive path, not EPA collision replay.
    original_step = sim.step
    def step_then_fault() -> None:
      original_step()
      sim.data.overflow[1] = 256
    sim.step = step_then_fault
    before = sim.data.qpos[1].clone().numpy()
    with tempfile.TemporaryDirectory() as temp:
      install_physics_overflow_guard(
        sim, resolve_task_spec("ame2-gaze", "base"), {}, Path(temp),
      )
      with self.assertRaises(PhysicsOverflowError):
        sim.step()
      dump = next(Path(temp).iterdir())
      self.assertTrue(json.loads((dump / "metadata.json").read_text())["dump_complete"])
      with np.load(dump / "state.npz", allow_pickle=False) as state:
        np.testing.assert_array_equal(state["before_qpos"][0], before)
        self.assertEqual(state["model_body_mass"][0, 1], 7.)
      restored = mujoco.MjModel.from_binary_path(str(dump / "model.mjb"))
      self.assertEqual(restored.nq, model.nq)


if __name__ == "__main__":
  unittest.main()
