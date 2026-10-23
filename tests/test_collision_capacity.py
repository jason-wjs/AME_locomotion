from __future__ import annotations

import inspect
from pathlib import Path
import unittest
from unittest.mock import patch

import mujoco
import mujoco_warp as mjwarp
import warp as wp
from mujoco_warp._src import collision_convex, collision_flex, types

from ame_g1.config.variants import iter_task_specs, resolve_task_spec
from ame_g1.simulation import configure_collision_capacity


class CollisionCapacityTests(unittest.TestCase):
  def setUp(self) -> None:
    self.modules = (types, collision_convex, collision_flex)
    for module in self.modules:
      patcher = patch.object(module, "MJ_MAX_EPAHORIZON", 24)
      patcher.start()
      self.addCleanup(patcher.stop)

  def test_gaze_base_configures_all_collision_modules(self) -> None:
    report = configure_collision_capacity(resolve_task_spec("ame2-gaze", "base"))
    self.assertEqual(report, {module.__name__: 64 for module in self.modules})
    self.assertTrue(all(module.MJ_MAX_EPAHORIZON == 64 for module in self.modules))

  def test_g1_base_retains_its_capacity(self) -> None:
    report = configure_collision_capacity(resolve_task_spec("g1", "base"))
    self.assertEqual(report, {module.__name__: 64 for module in self.modules})
    self.assertTrue(all(module.MJ_MAX_EPAHORIZON == 64 for module in self.modules))

  def test_other_tasks_report_defaults_without_changing_them(self) -> None:
    for task_spec in iter_task_specs():
      if task_spec.variant in ("g1", "ame2-gaze") and task_spec.phase == "base":
        continue
      with self.subTest(task_id=task_spec.task_id):
        report = configure_collision_capacity(task_spec)
        self.assertEqual(report, {module.__name__: 24 for module in self.modules})
        self.assertTrue(all(module.MJ_MAX_EPAHORIZON == 24 for module in self.modules))

  def test_out_of_scope_task_reports_existing_values_without_resetting_them(self) -> None:
    expected = {}
    for module, capacity in zip(self.modules, (64, 48, 32)):
      module.MJ_MAX_EPAHORIZON = capacity
      expected[module.__name__] = capacity
    report = configure_collision_capacity(resolve_task_spec("g1", "finetune"))
    self.assertEqual(report, expected)
    self.assertEqual(
      {module.__name__: module.MJ_MAX_EPAHORIZON for module in self.modules}, expected,
    )


class CollisionCapacityCpuRuntimeTests(unittest.TestCase):
  def test_gaze_capacity_is_used_by_real_convex_collision(self) -> None:
    # The EPA horizon is a temporary array inside convex_narrowphase, not a
    # Data field. Observe its allocation while running a real CPU collision.
    for module in (types, collision_convex, collision_flex):
      patcher = patch.object(module, "MJ_MAX_EPAHORIZON", 24)
      patcher.start()
      self.addCleanup(patcher.stop)
    configure_collision_capacity(resolve_task_spec("ame2-gaze", "base"))
    source = Path(collision_convex.__file__)
    allocation_line = next(
      line_number
      for line_number, line in enumerate(source.read_text().splitlines(), 1)
      if "epa_horizon = wp.empty" in line
    )
    observed_allocations = []
    original_empty = wp.empty

    def observe_empty(*args, **kwargs):
      caller = inspect.currentframe().f_back
      result = original_empty(*args, **kwargs)
      if Path(caller.f_code.co_filename) == source and caller.f_lineno == allocation_line:
        observed_allocations.append((result.shape, result.device.is_cpu))
      return result

    model = mujoco.MjModel.from_xml_string("""
      <mujoco>
        <option gravity="0 0 0"/>
        <worldbody>
          <body pos="0 0 0">
            <freejoint/>
            <geom type="ellipsoid" size=".2 .15 .1"/>
          </body>
          <body pos=".25 0 0">
            <freejoint/>
            <geom type="ellipsoid" size=".2 .15 .1"/>
          </body>
        </worldbody>
      </mujoco>
    """)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    with wp.ScopedDevice("cpu"):
      warp_model = mjwarp.put_model(model)
      warp_data = mjwarp.put_data(
        model, data, nworld=1, nconmax=16, nccdmax=16, njmax=64,
      )
      with patch.object(wp, "empty", new=observe_empty):
        mjwarp.collision(warp_model, warp_data)
      wp.synchronize_device("cpu")

    self.assertEqual(observed_allocations, [((warp_data.naccdmax, 64), True)])
    self.assertGreater(int(warp_data.nacon.numpy()[0]), 0)
    self.assertEqual(warp_data.overflow.numpy().tolist(), [0])


if __name__ == "__main__":
  unittest.main()
