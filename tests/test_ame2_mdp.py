"""Semantic migration checks for ray order, terrain goals and critic reflection."""

from types import SimpleNamespace
import unittest

import torch
from tensordict import TensorDict
from mjlab.sensor import GridPatternCfg, ObjRef
import warp as wp

from ame_g1.mdp.ame2_commands import SafeTerrainBasedPose2dCommandCfg
from ame_g1.mdp.ame2_observations import ame2_teacher_mapping, ame2_commands_obs, ame2_contact_states, contact_state_indices
from ame_g1.mdp.ame2_rewards import body_acceleration, ame2_is_terminated_term, ame2_undesired_torso_collision, ame2_undesired_stumbling
from mjlab.managers.termination_manager import TerminationManager, TerminationTermCfg
from ame_g1.mdp.ame2_symmetry import ame2_g1_symmetry_augmentation, mirror_critic
from ame_g1.mdp.ame2_scan import AME2GridPatternCfg, AME2RayCastSensorCfg


class Scene(dict):
  __getattr__ = dict.__getitem__


class MdpMigrationTests(unittest.TestCase):
  def test_stumbling_distinguishes_ground_support_from_lateral_impact(self):
    # Native MuJoCo netforce points down for a foot pressing on the ground.
    forces = torch.tensor([[[[5., 0., -200.]]], [[[250., 0., -100.]]], [[[0., 0., 0.]]]])
    env = SimpleNamespace(scene=Scene(contact=SimpleNamespace(data=SimpleNamespace(force_history=forces))))
    torch.testing.assert_close(ame2_undesired_stumbling(env, "contact"), torch.tensor([0., 1., 0.]))

  def test_height_map_preserves_xy_and_missing_ray_semantics(self):
    # A rectangular grid and a directional ramp expose a silent x/y transpose.
    pattern = GridPatternCfg(size=(.4, .2), resolution=.2)
    offsets, _ = pattern.generate_rays(None, "cpu")
    hits = offsets[None].clone()
    hits[..., 2] = 2 + 10 * hits[..., 0] + hits[..., 1]
    sensor = SimpleNamespace(cfg=SimpleNamespace(pattern=pattern), data=SimpleNamespace(
      hit_pos_w=hits, distances=torch.ones(1, 6), pos_w=torch.tensor([[0.,0.,2.]]),
    ))
    env = SimpleNamespace(num_envs=1, device="cpu", scene=Scene(
      scan=sensor, robot=SimpleNamespace(data=SimpleNamespace(root_link_pos_w=torch.tensor([[0., 0., 2.]]))),
    ))
    mapped = ame2_teacher_mapping(env, "scan", offset=0., clip_z=None, void_bias_range=(-1., -1.))
    self.assertEqual(mapped.shape, (1, 3, 2, 3))
    torch.testing.assert_close(mapped[..., 2], 10 * mapped[..., 0] + mapped[..., 1])
    sensor.data.distances[0, 0] = -1
    # A miss has finite hit_pos_w in mjlab; distances, rather than isfinite alone, decide validity.
    mapped = ame2_teacher_mapping(env, "scan", offset=0., clip_z=None, void_bias_range=(-1., -1.))
    self.assertAlmostEqual(mapped[0, 0, 0, 2].item(), -2.9, places=5)
    sensor.data.distances.fill_(-1)
    mapped = ame2_teacher_mapping(env, "scan", offset=0., clip_z=(-1.5, 1.5))
    torch.testing.assert_close(mapped[..., 2], torch.full((1, 3, 2), -1.5))

  def test_scan_drift_moves_actual_ray_inputs_and_only_reset_environments(self):
    wp.init()
    pattern = AME2GridPatternCfg(size=(.4, .2), resolution=.2)
    offsets, directions = pattern.generate_rays(None, "cpu")
    sensor = AME2RayCastSensorCfg(name="scan", frame=ObjRef(type="body", name="pelvis", entity="robot"), pattern=pattern).build()
    sensor._data = SimpleNamespace(xpos=torch.tensor([[[0.,0.,1.]],[[2.,3.,1.]]]), xmat=torch.eye(3).reshape(1,1,9).repeat(2,1,1))
    sensor._model = object()
    sensor._local_offsets, sensor._local_directions = offsets, directions
    sensor._frame_infos = [("body", 0, 0)]
    sensor._num_rays_per_frame = sensor._num_rays = len(offsets)
    ray_origins = torch.zeros(2,len(offsets),3)
    ray_directions = torch.zeros_like(ray_origins)
    sensor._ray_pnt = wp.from_torch(ray_origins, dtype=wp.vec3)
    sensor._ray_vec = wp.from_torch(ray_directions, dtype=wp.vec3)
    sensor.drift = torch.tensor([[.02,-.03,.01],[-.01,.02,-.02]])
    sensor.prepare_rays()
    expected = sensor._data.xpos + offsets[None] + sensor.drift[:,None]
    torch.testing.assert_close(ray_origins, expected)
    torch.testing.assert_close(sensor._cached_world_origins, expected)
    torch.testing.assert_close(sensor._cached_frame_pos, sensor._data.xpos + sensor.drift[:,None])
    # Neither ray offset nor sensor drift changes the simulated robot pose.
    torch.testing.assert_close(sensor._data.xpos[0,0], torch.tensor([0.,0.,1.]))
    untouched = sensor.drift[1].clone()
    sensor.reset(torch.tensor([0]))
    torch.testing.assert_close(sensor.drift[1], untouched)
    self.assertTrue((sensor.drift[0].abs() <= .04).all())

  def test_early_termination_count_uses_native_manager_and_excludes_timeouts(self):
    env = SimpleNamespace(num_envs=3, device="cpu")
    env.termination_manager = TerminationManager({
      "early_fall": TerminationTermCfg(func=lambda env: torch.tensor([True, True, False])),
      "early_impact": TerminationTermCfg(func=lambda env: torch.tensor([True, False, False])),
      "time_out": TerminationTermCfg(func=lambda env: torch.tensor([False, True, True]), time_out=True),
    }, env)
    env.termination_manager.compute()
    torch.testing.assert_close(ame2_is_terminated_term(env), torch.tensor([2., 0., 0.]))

  def test_torso_collision_sums_opposed_pair_forces_before_single_threshold(self):
    forces = torch.zeros(1,2,4,3)
    forces[0,0,0,1] = 2.
    forces[0,1,0,1] = -2.
    env = SimpleNamespace(scene=Scene(contact=SimpleNamespace(data=SimpleNamespace(force_history=forces))))
    torch.testing.assert_close(ame2_undesired_torso_collision(env,"contact"), torch.ones(1))

  def test_goal_sampling_respects_cone_and_partial_reset(self):
    patches = torch.tensor([[[[2., 0., .2], [-2., 0., .3], [1., 3., .4]]]])
    terrain = SimpleNamespace(flat_patches={"target": patches}, terrain_levels=torch.zeros(2,dtype=torch.long), terrain_types=torch.zeros(2,dtype=torch.long), env_origins=torch.zeros(2,3))
    root = SimpleNamespace(default_root_state=torch.tensor([[0.,0.,.7],[0.,0.,.7]]), root_link_pos_w=torch.zeros(2,3), root_link_quat_w=torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]), heading_w=torch.zeros(2))
    env = SimpleNamespace(num_envs=2, device="cpu", scene=Scene(terrain=terrain, robot=SimpleNamespace(data=root)))
    term = SafeTerrainBasedPose2dCommandCfg(resampling_time_range=(16.,16.)).build(env)
    term._resample_command(torch.tensor([0]))
    term._update_command(torch.tensor([0]))
    torch.testing.assert_close(term.command[0,:3], torch.tensor([2.,0.,.9]))
    torch.testing.assert_close(term.command[1], torch.zeros(4))
    terrain.flat_patches["target"][0,0,0,:2] = torch.tensor([-3., 0.])
    term._resample_command(torch.tensor([1]))  # No candidate in cone: official sampling of any patch.
    self.assertTrue(any(torch.equal(term.pos_command_w[1,:2], candidate[:2]) for candidate in patches[0,0]))

  def test_heavy_contact_channels_resolve_names_not_sensor_or_body_positions(self):
    forces = torch.zeros(1,4,2,3)
    forces[0,2,1,2] = 20.  # Left ankle, only in an older physics substep.
    sensor = SimpleNamespace(primary_names=["right_ankle", "pelvis", "left_ankle", "camera"], data=SimpleNamespace(force_history=forces))
    robot = SimpleNamespace(body_names=["pelvis","left_ankle","camera","right_ankle"], data=SimpleNamespace(indexing=SimpleNamespace(body_ids=torch.tensor([1,2,3,4]))))
    env = SimpleNamespace(scene=Scene(robot=robot,contacts=sensor), sim=SimpleNamespace(get_default_field=lambda _:torch.tensor([0.,10.,1.,.01,1.])))
    self.assertEqual(contact_state_indices(env,"contacts"), [1,2,0])
    torch.testing.assert_close(ame2_contact_states(env,"contacts"), torch.tensor([[0.,1.,0.]]))
    self.assertEqual(contact_state_indices(env,"contacts",ignore_bodies=("pelvis",)), [2,0])

  def test_distant_goal_clips_position_without_mutating_command(self):
    command = torch.tensor([[3.,4.,0.,.4],[.2,.3,0.,.4]])
    original = command.clone()
    env = SimpleNamespace(command_manager=SimpleNamespace(get_command=lambda _:command))
    observed = ame2_commands_obs(env, "base_pose", dist_thr=2.)
    torch.testing.assert_close(observed[0,:2], torch.tensor([1.2,1.6]))
    torch.testing.assert_close(observed[1,2:], torch.stack((command[1, 3].sin(), command[1, 3].cos())))
    torch.testing.assert_close(command, original)

  def test_acceleration_cache_does_not_advance_twice_or_spike_after_reset(self):
    data = SimpleNamespace(body_link_pos_w=torch.zeros(2,1,3))
    env = SimpleNamespace(scene=Scene(robot=SimpleNamespace(data=data)), common_step_counter=0, step_dt=.02, episode_length_buf=torch.zeros(2,dtype=torch.long))
    body_acceleration(env)
    for step in range(1,4):
      env.common_step_counter = step
      env.episode_length_buf.fill_(step)
      data.body_link_pos_w[:,:,0] = .02 * step
      measured = body_acceleration(env).clone()
      torch.testing.assert_close(body_acceleration(env), measured)
    torch.testing.assert_close(measured, torch.zeros_like(measured), atol=1e-4, rtol=0)
    data.body_link_pos_w[0,:,0] = 100.
    env.episode_length_buf[0] = 1
    env.common_step_counter += 1
    self.assertEqual(body_acceleration(env)[0].abs().sum().item(), 0.)

  def test_critic_reflection_is_an_involution_and_actor_groups_are_unchanged(self):
    legs = [f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    arms = [f"{side}_{joint}_joint" for side in ("left", "right")
            for joint in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
    joints = legs + [f"waist_{axis}_joint" for axis in ("yaw", "roll", "pitch")] + arms
    names = ["robot/left_foot", "robot/torso", "robot/right_foot"]
    params = {"contact_states": {"sensor_name": "contacts", "sensor_indices": [0,1,2]}, "height_scan": {"sensor_name": "scan"}}
    for selected in (joints, [name for name in reversed(joints) if "wrist" not in name]):
      with self.subTest(actions=len(selected)):
        env = SimpleNamespace(
          scene=Scene(robot=SimpleNamespace(joint_names=joints), contacts=SimpleNamespace(primary_names=names), scan=SimpleNamespace(cfg=SimpleNamespace(pattern=GridPatternCfg(size=(.4,.2),resolution=.2)))),
          action_manager=SimpleNamespace(get_term=lambda name:SimpleNamespace(target_names=selected)),
          observation_manager=SimpleNamespace(get_term_cfg=lambda group,name:SimpleNamespace(params=params[name])),
        )
        width = 9 + 3 * len(selected) + 4 + 1 + 3 + 6
        obs = torch.arange(2 * width).float().reshape(2,width)
        mirrored = mirror_critic(obs, env)
        torch.testing.assert_close(mirror_critic(mirrored, env), obs)
        for left, right, sign in (("left_hip_roll_joint", "right_hip_roll_joint", -1),
                                   ("left_knee_joint", "right_knee_joint", 1)):
          for offset in (9, 9 + len(selected), 9 + 2 * len(selected)):
            torch.testing.assert_close(mirrored[:,offset + selected.index(left)], sign * obs[:,offset + selected.index(right)])
        contacts = 9 + 3 * len(selected) + 5
        torch.testing.assert_close(mirrored[:,contacts:contacts+3], obs[:,[contacts+2,contacts+1,contacts]])
        td = TensorDict({"critic":obs,"teacher_prop":torch.randn(2,5),"teacher_mapping":torch.randn(2,3,2,3)},batch_size=[2])
        actions = torch.arange(2 * len(selected)).float().reshape(2,-1)
        augmented, augmented_actions = ame2_g1_symmetry_augmentation(td, actions=actions, env=env)
        torch.testing.assert_close(augmented["teacher_prop"][2:], td["teacher_prop"])
        torch.testing.assert_close(augmented["teacher_mapping"][2:], td["teacher_mapping"])
        torch.testing.assert_close(augmented_actions[2:,selected.index("left_hip_roll_joint")], -actions[:,selected.index("right_hip_roll_joint")])
        torch.testing.assert_close(td["critic"], obs)
        with self.assertRaisesRegex(ValueError, "critic layout expects"):
          mirror_critic(obs[:,:-1], env)


if __name__ == "__main__":
  unittest.main()
