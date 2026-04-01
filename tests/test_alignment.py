from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from tensordict import TensorDict
import tyro

import mjlab
from mjlab.scripts.train import TrainConfig

from ame_g1 import mdp as ame_mdp
from ame_g1.config import register as register_ame_tasks
from ame_g1.config.variants import resolve_task_spec
from ame_g1.config.g1.env_cfgs import unitree_g1_ame_env_cfg
from ame_g1.config.g1.rl_cfg import unitree_g1_ame_ppo_runner_cfg
from ame_g1.learning.symmetry import augment_g1_symmetry


_JOINT_NAMES = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


class _FakeRobot:
  joint_names = _JOINT_NAMES


class _FakeScene(dict):
  def __init__(self) -> None:
    super().__init__({"robot": _FakeRobot()})


class _FakeEnv:
  scene = _FakeScene()


class AlignmentTests(unittest.TestCase):
  @classmethod
  def setUpClass(cls) -> None:
    register_ame_tasks()

  def test_runner_cfg_enables_symmetry_augmentation(self) -> None:
    cfg = unitree_g1_ame_ppo_runner_cfg(stage="stage1")
    symmetry_cfg = cfg.algorithm.symmetry_cfg
    self.assertIsNotNone(symmetry_cfg)
    assert symmetry_cfg is not None
    self.assertTrue(symmetry_cfg["use_data_augmentation"])
    self.assertFalse(symmetry_cfg["use_mirror_loss"])
    self.assertEqual(
      symmetry_cfg["data_augmentation_func"],
      "ame_g1.learning.symmetry:augment_g1_symmetry",
    )

  def test_default_terrains_match_paper_core_sets(self) -> None:
    stage1 = unitree_g1_ame_env_cfg(stage="stage1", play=False)
    stage2 = unitree_g1_ame_env_cfg(stage="stage2", play=False)
    stage1_keys = set(stage1.scene.terrain.terrain_generator.sub_terrains)
    stage2_keys = set(stage2.scene.terrain.terrain_generator.sub_terrains)
    self.assertEqual(stage1_keys, {"pallets", "gaps", "grid_stones", "beams"})
    self.assertEqual(
      stage2_keys,
      {
        "pallets",
        "gaps",
        "grid_stones",
        "beams",
        "pentagon_stones",
        "single_column_stones",
        "narrow_pallets",
        "consecutive_gaps",
      },
    )

  def test_reward_config_includes_torso_and_humanoid_alignment_terms(self) -> None:
    stage1 = unitree_g1_ame_env_cfg(stage="stage1", play=False)
    stage2 = unitree_g1_ame_env_cfg(stage="stage2", play=False)
    self.assertIs(stage1.rewards["track_linear_velocity"].func, ame_mdp.track_body_linear_velocity)
    self.assertIs(stage1.rewards["track_angular_velocity"].func, ame_mdp.track_body_angular_velocity)
    self.assertIs(stage1.rewards["body_ang_vel"].func, ame_mdp.body_angular_velocity_penalty_b)
    self.assertIn("body_linear_velocity", stage1.rewards)
    self.assertIn("foot_contact_forces", stage1.rewards)
    self.assertIn("ankle_orientation", stage1.rewards)
    self.assertIn("standing_joint_pos", stage2.rewards)
    self.assertIn("standing_joint_vel", stage2.rewards)

  def test_symmetry_augmentation_mirrors_obs_and_actions(self) -> None:
    proprio = torch.arange(99, dtype=torch.float32).reshape(1, 99)
    terrain = torch.arange(17 * 11 * 3, dtype=torch.float32).reshape(1, 17, 11, 3)
    actions = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    obs = TensorDict(
      {
        "actor_proprio": proprio.clone(),
        "actor_terrain": terrain.clone(),
        "critic_proprio": proprio.clone(),
        "critic_terrain": terrain.clone(),
      },
      batch_size=[1],
    )

    augmented_obs, augmented_actions = augment_g1_symmetry(
      env=_FakeEnv(),
      obs=obs,
      actions=actions,
    )

    self.assertEqual(tuple(augmented_obs.batch_size), (2,))
    self.assertEqual(tuple(augmented_actions.shape), (2, 29))
    torch.testing.assert_close(augmented_obs["actor_proprio"][0], proprio[0])
    torch.testing.assert_close(augmented_actions[0], actions[0])

    mirrored_terrain = augmented_obs["actor_terrain"][1]
    torch.testing.assert_close(mirrored_terrain[..., :2], terrain[0, ..., :2])
    torch.testing.assert_close(mirrored_terrain[..., 2], terrain[0, :, torch.arange(10, -1, -1), 2])

    mirrored = augmented_obs["actor_proprio"][1]
    self.assertEqual(mirrored[0].item(), proprio[0, 0].item())
    self.assertEqual(mirrored[1].item(), -proprio[0, 1].item())
    self.assertEqual(mirrored[5].item(), -proprio[0, 5].item())
    self.assertEqual(mirrored[96].item(), proprio[0, 96].item())
    self.assertEqual(mirrored[97].item(), -proprio[0, 97].item())
    self.assertEqual(mirrored[98].item(), -proprio[0, 98].item())
    self.assertEqual(mirrored[25].item(), -proprio[0, 32].item())
    self.assertEqual(mirrored[32].item(), -proprio[0, 25].item())
    self.assertEqual(augmented_actions[1, 1].item(), -actions[0, 7].item())
    self.assertEqual(augmented_actions[1, 7].item(), -actions[0, 1].item())

  def test_ankle_penalty_ignores_raycast_misses_for_height_span(self) -> None:
    fake_sensor = SimpleNamespace(
      data=SimpleNamespace(
        pos_w=torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32),
        hit_pos_w=torch.tensor(
          [[[0.0, 0.0, 0.90], [0.0, 0.0, 0.92], [0.0, 0.0, 0.00]]],
          dtype=torch.float32,
        ),
        distances=torch.tensor([[0.1, 0.1, -1.0]], dtype=torch.float32),
      ),
      cfg=SimpleNamespace(max_distance=5.0),
    )
    env = SimpleNamespace(scene={"terrain_scan": fake_sensor})
    with patch(
      "ame_g1.mdp.rewards.body_projected_gravity_l2",
      return_value=torch.tensor([2.0], dtype=torch.float32),
    ):
      penalty = ame_mdp.ankle_orientation_penalty(
        env,
        asset_cfg=None,  # Patched body_projected_gravity_l2 ignores this.
        sensor_name="terrain_scan",
        elevation_threshold=0.18,
        reduced_scale=0.1,
      )
    torch.testing.assert_close(penalty, torch.tensor([2.0], dtype=torch.float32))

  def test_train_config_tyro_help_and_gpu_ids_parse(self) -> None:
    default = TrainConfig.from_task(resolve_task_spec("g1", "stage1").task_id)
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
      with self.assertRaises(SystemExit) as exc:
        tyro.cli(TrainConfig, args=["--help"], default=default, config=mjlab.TYRO_FLAGS)
    self.assertEqual(exc.exception.code, 0)

    parsed = tyro.cli(
      TrainConfig,
      args=["--gpu-ids", "[0]"],
      default=default,
      config=mjlab.TYRO_FLAGS,
    )
    self.assertEqual(parsed.gpu_ids, [0])


if __name__ == "__main__":
  unittest.main()
