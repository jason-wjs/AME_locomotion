# Third-party notices

These notices cover the migration files listed below. They do not assign a license to unrelated repository files or replace their existing notices.

## AME-1 community terrain migration

`src/ame_g1/terrains/community/{continuous,heightfields,surface}.py` adapts the
Isaac Lab v2.3.0 terrain generators and heightfield conversion at commit
`3c6e67bb5c7ada942a6d1884ab69338f57596f77` (BSD-3-Clause). The concentric-gap
recipe follows the community `AME_Locomotion` implementation at
`1d3519ee946c846c1c89dd660539a224f99972b5`, retaining its Isaac Lab source
notice. The source headers are preserved. The complete Isaac Lab license is
included in [isaaclab-BSD-3-Clause.txt](licenses/third_party/isaaclab-BSD-3-Clause.txt).

The port uses an explicit NumPy generator, native box geometry, narrow
heightfield strips and convex pieces for corrected sparse surfaces. Rough
heightfields retain the sampled heights but do not reproduce Isaac Lab's
steep-edge XY correction. These representation differences and their measured
limits are recorded in [behavior-repair.md](docs/behavior-repair.md).

## AME-2 reference

Reference: [leggedrobotics/ame2_minimal](https://github.com/leggedrobotics/ame2_minimal/tree/8beb9a688d67882828019583f7d5a78e3517f233), commit `8beb9a688d67882828019583f7d5a78e3517f233`.

The upstream root is GPL-3.0. Its [README license section](https://github.com/leggedrobotics/ame2_minimal/blob/8beb9a688d67882828019583f7d5a78e3517f233/README.md#license) explicitly preserves BSD-3-Clause for the modified `rsl_rl/` package and for `ame2/` files marked “The Isaac Lab Project Developers.” The portions identified here come from those explicit BSD exceptions; the root GPL declaration is not treated as a BSD grant for other upstream files.

### RSL-RL portions — BSD-3-Clause

Copyright (c) 2021–2026, ETH Zurich and NVIDIA CORPORATION, as retained in source headers. The package license also identifies Copyright (c) 2026, ETH Zurich and Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.

| Upstream source, relative to the pinned repository | Local adaptation, relative to `src/ame_g1/` |
| --- | --- |
| `rsl_rl/rsl_rl/models/AME2_models.py`; `rsl_rl/rsl_rl/modules/ame2_modules.py` | `learning/ame2_models.py` |
| `rsl_rl/rsl_rl/algorithms/ppo.py` | `learning/ame2_ppo.py` |
| `rsl_rl/rsl_rl/storage/rollout_storage.py` | `learning/ame2_storage.py` |
| `rsl_rl/rsl_rl/utils/muon.py` | `learning/ame2_muon.py` |

Changes adapt the teacher models, reward-component storage and PPO to RSL-RL 5.4.2, including its distribution/checkpoint API and deterministic export interfaces. Muon's algorithm references are retained in its source file.

Exact license text: [rsl_rl-BSD-3-Clause.txt](licenses/third_party/rsl_rl-BSD-3-Clause.txt), copied unchanged from upstream `rsl_rl/LICENSE`.

### Isaac Lab-derived portions — BSD-3-Clause

Copyright (c) 2022–2025 and 2022–2026, The Isaac Lab Project Developers, as identified by the respective source headers.

| Upstream source, relative to the pinned repository | Local adaptation, relative to `src/ame_g1/` |
| --- | --- |
| `ame2/ame2/tasks/ame2_env_cfg.py`; `ame2/ame2/tasks/agents/rsl_rl_ppo_cfg.py` | `config/g1/ame2_env_cfg.py`, `config/g1/ame2_rl_cfg.py`, `mdp/ame2_scan.py` |
| `ame2/ame2/assets/unitree.py` | `config/g1/ame2_robot.py` |
| `ame2/ame2/tasks/mdp/{actions,commands,curriculums,events,observations,rewards,symmetry,terminations}.py` | Corresponding `mdp/ame2_*.py` modules (`curriculums` becomes `ame2_curriculum`) |
| `ame2/ame2/tasks/{custom_terrains,terrains}.py` | `terrains/ame2_geometry.py`, `terrains/ame2_config.py` |

Changes replace Isaac Lab runtime interfaces with mjlab, retain the mjlab G1 asset, and adapt contact, scan, action-delay and command conventions. Terrain generation uses native MuJoCo surfaces, separate convex components, tile-boundary clipping and native-ray target sampling. Geometry differences are recorded in [stage2-preparation.md](docs/stage2-preparation.md).

Heightfield endpoint/border handling and rectangular-border construction also follow the BSD-3-Clause sources [`height_field/utils.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/terrains/height_field/utils.py) and [`trimesh/utils.py`](https://github.com/isaac-sim/IsaacLab/blob/v2.3.2/source/isaaclab/isaaclab/terrains/trimesh/utils.py) in Isaac Lab v2.3.2.

Exact license text: [isaaclab-BSD-3-Clause.txt](licenses/third_party/isaaclab-BSD-3-Clause.txt), copied unchanged from upstream `licenses/dependencies/isaaclab-license.txt`. Isaac Lab itself, the upstream robot meshes and the Livox scan-pattern data are not vendored by this migration.

Installed dependencies retain their own licenses. These notices and license texts must accompany redistribution of the corresponding adapted portions.
