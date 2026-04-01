from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mjlab.sensor import GridPatternCfg

from ame_g1.mdp.observations import terrain_points
from ame_g1.mdp.rewards import track_body_angular_velocity, track_body_linear_velocity


class ObservationContractTests(unittest.TestCase):
  def test_terrain_points_return_local_xyz_and_negative_miss_depth(self) -> None:
    sensor = SimpleNamespace(
      cfg=SimpleNamespace(
        pattern=GridPatternCfg(size=(0.1, 0.0), resolution=0.1),
        max_distance=5.0,
      ),
      data=SimpleNamespace(
        pos_w=torch.tensor([[1.0, 2.0, 0.5]], dtype=torch.float32),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        hit_pos_w=torch.tensor(
          [[[0.95, 2.0, 0.4], [1.05, 2.0, 0.0]]],
          dtype=torch.float32,
        ),
        distances=torch.tensor([[0.1, -1.0]], dtype=torch.float32),
      ),
    )
    env = SimpleNamespace(
      scene={"terrain_scan": sensor},
      num_envs=1,
      device="cpu",
      _ame_map_scan_drift_xy=torch.tensor([[0.02, -0.03]], dtype=torch.float32),
    )

    points = terrain_points(
      env,
      sensor_name="terrain_scan",
      apply_drift=True,
    )

    self.assertEqual(tuple(points.shape), (1, 2, 1, 3))
    torch.testing.assert_close(points[0, 0, 0], torch.tensor([-0.03, -0.03, -0.1]))
    torch.testing.assert_close(points[0, 1, 0], torch.tensor([0.07, -0.03, -5.0]))


class RewardContractTests(unittest.TestCase):
  def test_linear_tracking_ignores_vertical_velocity_error(self) -> None:
    env = SimpleNamespace(
      command_manager=SimpleNamespace(
        get_command=lambda _: torch.tensor([[0.4, -0.2, 0.0]], dtype=torch.float32)
      )
    )
    with patch(
      "ame_g1.mdp.rewards._body_velocities_b",
      return_value=(
        torch.tensor([[[0.1, -0.1, 0.8]]], dtype=torch.float32),
        torch.zeros(1, 1, 3, dtype=torch.float32),
      ),
    ):
      reward = track_body_linear_velocity(
        env,
        std=0.5,
        command_name="twist",
      )
    expected = torch.exp(-torch.tensor([(0.4 - 0.1) ** 2 + (-0.2 + 0.1) ** 2]) / 0.5**2)
    torch.testing.assert_close(reward, expected)

  def test_angular_tracking_ignores_roll_pitch_error(self) -> None:
    env = SimpleNamespace(
      command_manager=SimpleNamespace(
        get_command=lambda _: torch.tensor([[0.0, 0.0, 0.6]], dtype=torch.float32)
      )
    )
    with patch(
      "ame_g1.mdp.rewards._body_velocities_b",
      return_value=(
        torch.zeros(1, 1, 3, dtype=torch.float32),
        torch.tensor([[[0.9, -0.7, 0.4]]], dtype=torch.float32),
      ),
    ):
      reward = track_body_angular_velocity(
        env,
        std=0.5,
        command_name="twist",
      )
    expected = torch.exp(-torch.tensor([(0.6 - 0.4) ** 2]) / 0.5**2)
    torch.testing.assert_close(reward, expected)


if __name__ == "__main__":
  unittest.main()
