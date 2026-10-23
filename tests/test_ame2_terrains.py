"""Native surface checks for the AME-2 terrain migration."""

import numpy as np
import mujoco
import unittest
import trimesh
from dataclasses import replace

from mjlab.terrains.terrain_generator import FlatPatchSamplingCfg
from ame_g1.terrains.ame2_config import make_ame2_terrain_generator
from ame_g1.terrains.ame2_geometry import _MeshTerrainCfg, _add_heightfield


def _spec():
    spec = mujoco.MjSpec()
    spec.worldbody.add_body(name="terrain")
    return spec


def _ray_height(model, data, x, y):
    distance = mujoco.mj_ray(model, data, np.array([x, y, 100.]), np.array([0., 0., -1.]),
                             None, True, -1, np.zeros(1, dtype=np.int32))
    return 100. - distance if distance >= 0 else np.nan


class TestAME2Terrains(unittest.TestCase):
    def test_native_heightfield_keeps_xy_axes_and_physical_heights(self):
        spec = _spec()
        # Non-square grid and unequal slopes catch a silent x/y transposition.
        x, y = np.meshgrid(np.linspace(0, 4, 5), np.linspace(0, 2, 3), indexing="ij")
        heights = 0.2 * x + 0.05 * y - 0.3
        _add_heightfield(spec, heights, (4., 2.))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for px, py in ((0.5, 0.25), (2., 1.), (3.5, 1.5)):
            np.testing.assert_allclose(_ray_height(model, data, px, py), 0.2 * px + 0.05 * py - 0.3, atol=1e-6)


    def test_disconnected_source_mesh_does_not_bridge_collision_gap(self):
        class DisconnectedTerrain(_MeshTerrainCfg):
            @staticmethod
            def _generate(difficulty, cfg, rng):
                boxes = []
                for cx in (2., 8.):
                    mesh = trimesh.creation.box([1., 1., 1.])
                    mesh.apply_translation([cx, 5., -0.5])
                    boxes.append(mesh)
                return [trimesh.util.concatenate(boxes)], np.array([2., 5., 0.])

        spec = _spec()
        output = DisconnectedTerrain().function(1., spec, np.random.default_rng(42))
        assert len(output.geometries) == 2
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(_ray_height(model, data, 2., 5.), 0., atol=1e-6)
        np.testing.assert_allclose(_ray_height(model, data, 8., 5.), 0., atol=1e-6)
        assert np.isnan(_ray_height(model, data, 5., 5.))


    def test_protruding_component_is_clipped_before_neighboring_tiles_are_added(self):
        class ProtrudingTerrain(_MeshTerrainCfg):
            @staticmethod
            def _generate(difficulty, cfg, rng):
                mesh = trimesh.creation.box([3., 3., 1.])
                mesh.apply_translation([9.5, 5., 0.5])
                return [mesh], np.array([5., 5., 0.])

        spec = _spec()
        ProtrudingTerrain(size=(10., 10.)).function(1., spec, np.random.default_rng(42))
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(_ray_height(model, data, 9.5, 5.), 1., atol=1e-6)
        assert np.isnan(_ray_height(model, data, 10.5, 5.))

    def test_target_sampling_uses_native_surface_and_refuses_empty_gap(self):
        cfg = make_ame2_terrain_generator().sub_terrains["gap"]
        cfg = replace(cfg, size=(12., 12.), floor_ratio=0., min_gap=1., max_gap=1.)
        spec = _spec()
        output = cfg.function(1., spec, np.random.default_rng(42))
        targets = output.flat_patches["target"]
        assert targets.shape == (30, 3)
        assert np.all(targets[:, 0] >= 7.8)
        assert np.all(targets[:, 0] <= 11.)
        model = spec.compile()
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        for target in targets:
            np.testing.assert_allclose(target[2], _ray_height(model, data, *target[:2]), atol=1e-6)
        empty = replace(cfg, flat_patch_sampling={"target": FlatPatchSamplingCfg(
            num_patches=1, patch_radius=0.1, x_range=(1.4, 1.6), y_range=(-1., 1.),
        )})
        with self.assertRaisesRegex(RuntimeError, "native-surface.*targets"):
            empty.function(1., _spec(), np.random.default_rng(42))


if __name__ == "__main__":
    unittest.main()
