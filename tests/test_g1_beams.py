"""Physical bounds, support continuity and curriculum of the G1 radial beams."""

from dataclasses import replace
from itertools import product
import unittest

import mujoco
import numpy as np

from ame_g1.terrains.config import make_phase_terrain_generator


def _build(size, difficulty):
  cfg = replace(
    make_phase_terrain_generator("base").sub_terrains["beams"], size=size,
  )
  spec = mujoco.MjSpec()
  spec.worldbody.add_body(name="terrain")
  output = cfg.function(difficulty, spec, np.random.default_rng(42))
  model = spec.compile()
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  return cfg, output, model, data


class RadialBeamTests(unittest.TestCase):
  def test_rectangular_bounds_and_continuous_support_to_rim(self):
    corners = np.array(list(product((-1., 1.), repeat=3)))
    for size, difficulty in product(((8., 4.), (4., 8.), (8., 8.)), (0., .5, 1.)):
      with self.subTest(size=size, difficulty=difficulty):
        cfg, output, model, data = _build(size, difficulty)
        center = np.array(size) / 2
        np.testing.assert_allclose(output.origin, [*center, cfg.beam_height])
        for idx in range(model.ngeom):
          vertices = (corners * model.geom_size[idx]) @ data.geom_xmat[idx].reshape(3, 3).T
          vertices += data.geom_xpos[idx]
          self.assertTrue(np.all(vertices[:, :2] >= -1e-8))
          self.assertTrue(np.all(vertices[:, :2] <= np.array(size) + 1e-8))
        for angle in np.linspace(0., 2 * np.pi, cfg.num_beams, endpoint=False):
          direction = np.array([np.cos(angle), np.sin(angle)])
          exit_distance = min(
            half / abs(component) for half, component in zip(center, direction)
            if abs(component) > 1e-12
          )
          for distance in np.linspace(0., exit_distance - .001, 32):
            xy = center + distance * direction
            hit = mujoco.mj_ray(
              model, data, np.array([*xy, 2.]), np.array([0., 0., -1.]),
              None, True, -1, np.zeros(1, dtype=np.int32),
            )
            self.assertAlmostEqual(2. - hit, cfg.beam_height, places=6)

  def test_beams_narrow_with_difficulty_without_filling_the_pit(self):
    for difficulty, expected_width in ((0., .18), (.5, .15), (1., .12)):
      with self.subTest(difficulty=difficulty):
        cfg, _, model, data = _build((8., 4.), difficulty)
        np.testing.assert_allclose(model.geom_size[-cfg.num_beams:, 1] * 2, expected_width)
        angle = np.pi / cfg.num_beams
        xy = np.array(cfg.size) / 2 + np.array([np.cos(angle), np.sin(angle)])
        hit = mujoco.mj_ray(
          model, data, np.array([*xy, 2.]), np.array([0., 0., -1.]),
          None, True, -1, np.zeros(1, dtype=np.int32),
        )
        self.assertAlmostEqual(2. - hit, -cfg.floor_depth, places=6)


if __name__ == "__main__":
  unittest.main()
