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
    cfg = unitree_g1_ame_ppo_runner_cfg(phase="base")
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
    base = unitree_g1_ame_env_cfg(phase="base", play=False)
    finetune = unitree_g1_ame_env_cfg(phase="finetune", play=False)
    base_keys = set(base.scene.terrain.terrain_generator.sub_terrains)
    finetune_keys = set(finetune.scene.terrain.terrain_generator.sub_terrains)
    self.assertEqual(base_keys, {"pallets", "gaps", "grid_stones", "beams"})
    self.assertEqual(
      finetune_keys,
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

  def test_reward_config_matches_official_term_set_and_phase_weights(self) -> None:
    base = unitree_g1_ame_env_cfg(phase="base", play=False)
    finetune = unitree_g1_ame_env_cfg(phase="finetune", play=False)
    expected_terms = {
      "termination_penalty",
      "track_lin_vel_xy_exp",
      "track_ang_vel_z_exp",
      "ang_vel_xy_l2",
      "undesired_contacts",
      "dof_torques_l2",
      "dof_acc_l2",
      "dof_vel_l2",
      "dof_pos_limits",
      "dof_torques_limits",
      "action_rate_l2",
      "flat_orientation_l2",
      "feet_air_time",
      "feet_air_time_variance",
      "feet_slide",
      "feet_stumble",
      "feet_too_near",
      "joint_coordination",
      "joint_deviation_hip",
      "joint_deviation_arms",
      "joint_deviation_waists",
    }
    self.assertEqual(set(base.rewards), expected_terms)
    self.assertEqual(set(finetune.rewards), expected_terms)
    self.assertIs(base.rewards["track_lin_vel_xy_exp"].func, ame_mdp.track_lin_vel_xy_yaw_frame_exp)
    self.assertIs(base.rewards["track_ang_vel_z_exp"].func, ame_mdp.track_ang_vel_z_world_exp)
    self.assertIs(base.rewards["joint_deviation_arms"].func, ame_mdp.joint_deviation_l1)
    self.assertAlmostEqual(base.rewards["feet_stumble"].weight, -2.0)
    self.assertAlmostEqual(finetune.rewards["feet_stumble"].weight, -5.0)
    self.assertAlmostEqual(base.rewards["action_rate_l2"].weight, -0.01)
    self.assertAlmostEqual(finetune.rewards["action_rate_l2"].weight, -0.05)
    sensor_names = {sensor.name for sensor in base.scene.sensors or ()}
    self.assertIn("nonfoot_ground_contact", sensor_names)

  def test_symmetry_augmentation_mirrors_obs_and_actions(self) -> None:
    actor_proprio = torch.arange(96, dtype=torch.float32).reshape(1, 96)
    critic_proprio = torch.arange(99, dtype=torch.float32).reshape(1, 99)
    terrain = torch.arange(17 * 11 * 3, dtype=torch.float32).reshape(1, 17, 11, 3)
    actions = torch.arange(29, dtype=torch.float32).reshape(1, 29)
    obs = TensorDict(
      {
        "actor_proprio": actor_proprio.clone(),
        "actor_terrain": terrain.clone(),
        "critic_proprio": critic_proprio.clone(),
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
    torch.testing.assert_close(augmented_obs["actor_proprio"][0], actor_proprio[0])
    torch.testing.assert_close(augmented_actions[0], actions[0])

    mirrored_terrain = augmented_obs["actor_terrain"][1]
    torch.testing.assert_close(mirrored_terrain[..., :2], terrain[0, ..., :2])
    torch.testing.assert_close(mirrored_terrain[..., 2], terrain[0, :, torch.arange(10, -1, -1), 2])

    mirrored_actor = augmented_obs["actor_proprio"][1]
    self.assertEqual(mirrored_actor[0].item(), -actor_proprio[0, 0].item())
    self.assertEqual(mirrored_actor[1].item(), actor_proprio[0, 1].item())
    self.assertEqual(mirrored_actor[5].item(), actor_proprio[0, 5].item())
    self.assertEqual(mirrored_actor[93].item(), actor_proprio[0, 93].item())
    self.assertEqual(mirrored_actor[94].item(), -actor_proprio[0, 94].item())
    self.assertEqual(mirrored_actor[95].item(), -actor_proprio[0, 95].item())
    self.assertEqual(mirrored_actor[22].item(), -actor_proprio[0, 29].item())
    self.assertEqual(mirrored_actor[29].item(), -actor_proprio[0, 22].item())
    mirrored_critic = augmented_obs["critic_proprio"][1]
    self.assertEqual(mirrored_critic[96].item(), critic_proprio[0, 96].item())
    self.assertEqual(mirrored_critic[97].item(), -critic_proprio[0, 97].item())
    self.assertEqual(mirrored_critic[98].item(), -critic_proprio[0, 98].item())
    self.assertEqual(mirrored_critic[25].item(), -critic_proprio[0, 32].item())
    self.assertEqual(mirrored_critic[32].item(), -critic_proprio[0, 25].item())
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
    default = TrainConfig.from_task(resolve_task_spec("g1", "base").task_id)
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
