# AME-2 teacher 搭建记录

2026-10-21 方向更新：停止 Dense 调优，转向官方 Gaze teacher 迁移；当前实现与验收见 [Gaze 迁移](gaze-migration.md)。下文“gaze 不在本里程碑内”描述的是早期 Dense 里程碑。

> 当前状态（2026-10-21）：阶段二尚未通过可靠稀疏地形验收，不能据以下早期搭建记录直接启动正式训练。最新原协议对照和容量结果见 [行为修复记录](behavior-repair.md)。目前采用每卡单个任务，在已授权的空闲卡上做有界 debug；GPU 编号及是否空闲必须现场核实。下文涉及旧 GPU0/GPU1 分工和共卡安排的描述为历史记录，已被当前单卡单任务安排取代。

更新：2026-10-18。对应[路线图](roadmap.md)的第一个里程碑。

## 状态

独立分支 `research/ame2-foundation` 已接入 `AME2-G1-Teacher`：目标命令、12 类
地形、teacher 网络、23 个奖励价值头、Muon/bf16 PPO 和 critic 对称增强。
默认任务仍是社区 `g1`。CPU 回归、GPU1 小规模更新与恢复、完整地形 CLI 短训练
已通过；新进程恢复、公开回放入口、原默认任务实际更新回归及 100 次更新稳定性
检查也已通过。实现固定在 `a02f18b6c15083ff3c826acef0f0e983dc609220`。
尚未完成空闲卡容量测试或正式训练。

第一阶段的源码、HEAD、环境及控制器保持原样。其训练提交为
`a95d88d2a05d4eaca713945da4802a8b8d40d3d6`，第二阶段从此提交开始，使用独立
checkout 和复制安装的 `.venv`。两者锁定 mjlab 1.6.0、RSL-RL 5.4.2。

官方参照固定为
[`leggedrobotics/ame2_minimal@8beb9a6`](https://github.com/leggedrobotics/ame2_minimal/tree/8beb9a688d67882828019583f7d5a78e3517f233)。
仅迁移已发布 teacher；neural mapping student、gaze、AMP 和真机部署不在本里程碑内。
来源和许可见 [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md)。

## 验证结果

| 检查 | 实际结果 | 证据目录 |
| --- | --- | --- |
| 最终 CPU 回归 | 46 项通过，含原有 19 项契约及新增迁移检查 | `artifacts/ame2_cpu/final_cpu_regression.log` |
| 官方网络同权重对照 | actor、MoE critic 的前向与所有参数梯度，CPU FP32/BF16 最大误差均为 0 | `artifacts/ame2_cpu/official_model_parity.json` |
| 地形原始生成器 | 12 类 × 2 个难度端点对照通过，倒楼梯底面适配除外 | `artifacts/ame2_cpu/terrain_source_parity.json` |
| 原生完整地形 | 120 块、5,372 geoms；3,600 个目标点高度与原生射线一致，最大误差 `2.84e-14 m` | `artifacts/ame2_cpu/terrain_world_validation.json` |
| GPU1 集成 | 4 环境真实 step、timeout、局部 reset、PPO 更新、保存加载、继续更新、JIT 推理通过，16.73 秒 | `artifacts/ame2_gpu/integration.log` |
| 完整地形 CLI | 40 环境、10 次更新、200 个优化器步，参数及全部记录指标有限 | `artifacts/ame2_gpu/cli_full_validation.json` |
| 新进程 CLI 恢复 | 再执行 2 次更新；优化器步数 200→240，课程步数 240→288，熵系数正确接续 | `artifacts/ame2_gpu/cli_resume_validation.json` |
| 回放 CLI | 2 环境执行 100 步，正常退出 | `artifacts/ame2_gpu/cli_play.log` |
| 原默认任务 GPU 回归 | `g1` base 的真实对称增强/PPO 更新通过 | `artifacts/ame2_gpu/ame1_regression.log` |
| 有限稳定性检查 | 完整地形、40 环境、100 次更新；模型、优化器及日志全部有限，2,000 优化器步与 2,400 课程步数正确 | `artifacts/ame2_gpu/stability_100/validation.json` |
| ONNX | 结构检查通过，随机输入下与 PyTorch 最大绝对误差约 `3.7e-8` | 同上 |
| 包装与许可 | wheel 和 sdist 均包含三份第三方许可/说明文件 | `artifacts/ame2_cpu/package-build.log` |

实际观测为 proprioception 100 维、地图 `26×15×3`、critic 516 维；动作 29 维，
价值输出 23 维。100 次更新共约 19 分钟（含场景初始化），每 5 秒采样到的进程
显存峰值为 2,350 MiB。共享 GPU 上稳定吞吐约 127 环境步/秒，平均采样 5.61 秒、
学习 1.46 秒/更新。这不是独占容量或吞吐结论。

短训练权重位于：

```
artifacts/ame2_gpu/cli_full/ame2_g1_teacher/2026-10-17_15-42-17_full_terrain_smoke/model_9.pt
```

该权重用于接口与恢复验证，未经过策略效果验收。几何检查、模块一致性及短更新
通过都不能证明完整训练已收敛。

## 迁移选择

- **学习链路**：扩展锁定 RSL 的 reward 分量、GAE、timeout bootstrap、critic-only
  镜像、Muon、bf16 和熵衰减，没有安装官方 RSL fork。checkpoint 保存奖励头顺序
  和熵系数。原生 RSL 将 Gaussian 标准差限制在 `[1e-6,1e6]`，官方 actor 无此限幅。
- **机器人**：使用 mjlab G1-29 资产。23 个非腕关节对齐官方电机参数，六个腕关节
  保留原生控制器；原生资产没有 head 参考体的地方使用 torso。质量、惯量和碰撞
  几何不宣称与官方 USD 相同，也不宣称直接兼容官方 checkpoint。
- **随机化与动作**：质量随机化按官方默认同比缩放惯量；动作延迟按物理子步
  0–1 步实现，局部 reset 清空历史。actor scan 带 drift，critic scan 无 drift。
- **状态与接触**：机体加速度使用控制周期位置差分，reset 后排除无效差分。
  原生 contact timer 基于几何 `found`，不等同于大于 1 N 的接触力计时；滑移项
  涉及接触时长时存在此差异。torso 自碰撞逐手臂汇总力范数后计算单一指示。
- **地形**：保留 12 类正权重地形的参数与采样比例。mjlab 每类一列，10 行共
  120 块，官方为 600 块。离散石块保持独立凸形状，粗糙 mesh 逐连通块凸化。
  裁掉越界部分，避免覆盖相邻地形；倒楼梯零厚度底台补 2 cm 底面，顶部不变。
  原生 heightfield 不含 Isaac 的坡面顶点水平移动。
- **目标**：以原生几何射线检查目标足迹，拒绝采样有明确上限；失败时不生成
  替代目标。几何可落脚不代表策略可达。

## 运行与接续

训练和回放复用已有入口：`--variant ame2 --phase base`。这里 `base` 表示完整
teacher，暂不提供 AME-2 finetune。`--global-context` 属于社区 AME-1 的结构开关；
AME-2 保留固定 teacher 全局编码。AME-1 attention dump 暂不支持 AME-2。

CPU 回归：

```bash
CUDA_VISIBLE_DEVICES="" MUJOCO_GL=disable OMP_NUM_THREADS=2 PYTHONPATH=tests \
  .venv/bin/python -m unittest test_alignment test_cli test_variant_configs \
  test_observation_contract test_runner_save_policy test_ame2_models test_ame2_ppo \
  test_ame2_actions test_ame2_mdp test_ame2_terrains test_ame2_config
```

小规模真实环境检查：

```bash
CUDA_VISIBLE_DEVICES=<allowed-GPU-UUID> AME_TEST_DEVICE=cuda:0 \
  MUJOCO_GL=disable OMP_NUM_THREADS=2 .venv/bin/python -m unittest \
  discover -s tests -p test_ame2_integration.py
```

已验证的完整地形短训练入口：

```bash
CUDA_VISIBLE_DEVICES=<allowed-GPU-UUID> OMP_NUM_THREADS=2 \
  .venv/bin/python -m ame_g1.cli.train --variant ame2 \
  --env.scene.num-envs 40 --agent.max-iterations 10 \
  --agent.run-name full_terrain_smoke --log-root artifacts/ame2_gpu/cli_full
```

剩余顺序：第一阶段 GPU0 训练结束及必要权重运行检查 → 空闲卡容量与吞吐
测试 → 正式训练。正式配方为 5,000 次更新、每次 24 步；环境数须通过容量
测试确定，不能直接把配置默认的 4,096 当成容量结论。容量测试后记录实际启动
命令、代码提交、锁文件、种子、GPU UUID、日志和预算，再启动长任务。

RSL 保存最后一次更新的零基编号，恢复后的首条日志可能重复该编号。用优化器
状态和实际执行更新预算核对接续，不只看最后日志编号。

GPU1 可在显存余量充足时共卡做小测试；整卡容量测试与正式训练等待实际空闲卡。
GPU0 的第一阶段 base、finetune 和必要结果检查优先，不以阶段切换时的进程间隙
视为空闲。第二阶段只启动一份正式实验，不停止或调整其他项目任务。

网络曾中断；本地与远端实现已按 60 个源码/测试/许可文件的 SHA-256 核对一致。
原搭建心跳已停用，由持续 goal 接续。CPU 和安全共卡工作完成、只剩正式训练而
两卡持续忙时，保存状态并停止主动探索，不无限轮询或增加额外研究内容。

容量测量复用第一阶段脚本，适配副本保存在实验目录
`artifacts/benchmarks/ame2/benchmark_ame.py`。它要求显式 GPU，默认 AME-2 与四个
mini-batch，检查跟踪文件干净，并在每次试验前拒绝已有计算进程的 GPU。
已验证当前占用中的 GPU1 会被拒绝，未启动任何容量任务。

第一阶段控制器的终态为 `training_completed_pending_evaluation`。GPU0 释放后，
先核对最终权重并完成有界的实际加载/rollout 检查；不能把这个状态称为完整效果
验收或发布完成。第一轮源码、环境和控制器保持原样，检查记录写入第二阶段实验
目录。容量测试从单次 512 环境、25 更新、5 次预热开始，再按实测显存余量和吞吐
决定下一档，不直接提交无人审查的倍增列表。
