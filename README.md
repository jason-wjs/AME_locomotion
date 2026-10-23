# AME Reproduction for Unitree G1 on mjlab

当前升级分支适配 mjlab 1.6.0，默认算法仍为社区衍生的 `g1` 变体。
升级前基线为 `99b8aa4`（`stable/ame1-community`）；本轮不发布新版本。
维护方式与 AME-2 差异盘点见 [迁移说明](docs/migration-mjlab-1.6.md)，
开发检查见 [CONTRIBUTING](CONTRIBUTING.md)。
下一阶段的目标、实施顺序与范围边界见 [研究路线图](docs/roadmap.md)。

## 项目简介

本仓库是对论文 [*Attention-Based Map Encoding for Learning Generalized Legged Locomotion*](https://www.science.org/doi/10.1126/scirobotics.adv3604) 中基于注意力的地形编码器方法在Unitree G1 (29DoF)上的复现，使用 `mjlab`作为训练引擎。

代码提供三个显式变体：

- `paper`：作为论文对照基线，尽量贴近 AME 原始设定。
  - `17 x 11` terrain grid
  - `z-only` CNN 输入
  - CNN 后拼接坐标
  - 无 CNN 下采样
  - 无全局上下文
- `g1`：默认工程变体，参考[AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion) 在 G1 上的三点改进。
  - CNN 输入改为 `xyz`
  - 使用 CNN 步长下采样缩短注意力序列
  - 可选全局上下文分支（`--global-context on|off`）
- `ame2-gaze`：当前重点迁移的 AME-2 主动注视 teacher，23 动作；实现与验证状态见 [Gaze 迁移](docs/gaze-migration.md)。尚无通过完整行为验收的自训练权重。
- `ame2`：保留作为历史对照的实验性 AME-2 dense teacher，使用目标位置/朝向命令及独立的地形、网络和 PPO 配置。
  - 仅提供 `--phase base`，自身训练策略尚未通过稀疏地形行为验收。
  - 接口范围与已完成验证见 [AME-2 搭建记录](docs/stage2-preparation.md)，后续修复结果见 [修复记录](docs/behavior-repair.md)。

AME-1（`paper` / `g1`）训练流程采用 `phase` 语义：

- `base`：基础训练
- `finetune`：在 `base` checkpoint 上微调

修复分支的 `g1 base` 使用社区的八类、200 块地形，训练和 play 共用同一生成器。
默认训练规模为 1792 环境、8 个 minibatch，并关闭 cuDNN autotuning；
这是单张 H100 80GB 上的实测容量配置，不是地形学习已通过验收的声明。
当前仍在排查高速稳定性和稀疏地形学习，未发布正式训练权重。
物理表示差异、对照结果及训练预算见 [修复记录](docs/behavior-repair.md)。

## 依赖与安装

- Python: `>=3.10,<3.14`
- mjlab: `==1.6.0`（其余精确依赖由 `uv.lock` 固定）

```bash
uv sync --locked
```

## 命令入口

本仓库统一使用 `uv run`，不使用 `python -m` 作为推荐入口。

```bash
uv run ame-train --help
uv run ame-play --help
uv run ame-eval --help
uv run ame-plot-attention --help
```

核心参数：

- `--variant`: `paper | g1 | ame2 | ame2-gaze`
- `--phase`: `base | finetune`
- `--global-context`: `on | off`（`paper` 仅支持 `off`）
- `ame-train --preflight`：只检查配置和实际碰撞容量，不启动训练。
  Gaze 长跑中断原因、容量修复与故障状态保存见 [EPA 容量修复](docs/gaze-collision-capacity.md)。

## 快速开始

基础训练（默认推荐）：

```bash
uv run ame-train \
  --variant g1 \
  --phase base \
  --global-context off
```

微调训练（从 base checkpoint 继续）：

```bash
uv run ame-train \
  --variant g1 \
  --phase finetune \
  --global-context off \
  --checkpoint-file /abs/path/to/base_model.pt
```

论文对照基线：

```bash
uv run ame-train \
  --variant paper \
  --phase base \
  --global-context off
```

可视化运行：

```bash
uv run ame-play \
  --variant g1 \
  --phase finetune \
  --global-context off \
  --checkpoint-file /abs/path/to/model.pt \
  --viewer
```

评估：

```bash
uv run ame-eval \
  --variant g1 \
  --phase finetune \
  --global-context off \
  --checkpoint-file /abs/path/to/model.pt
```

attention 导出与绘图：

```bash
uv run ame-play \
  --variant g1 \
  --phase finetune \
  --global-context off \
  --checkpoint-file /abs/path/to/model.pt \
  --dump-file artifacts/attention_dump.pt

uv run ame-plot-attention \
  --dump-file artifacts/attention_dump.pt \
  --output-dir artifacts/attention
```

## 便捷脚本

训练脚本（封装 `uv run ame-train`）：

- [scripts/run_train.sh](scripts/run_train.sh)

```bash
bash scripts/run_train.sh base
bash scripts/run_train.sh finetune
bash scripts/run_train.sh full
```

评估/可视化脚本（封装 `ame-play` / `ame-eval`）：

- [scripts/run_play.sh](scripts/run_play.sh)

```bash
bash scripts/run_play.sh play
bash scripts/run_play.sh eval
bash scripts/run_play.sh dump
```

脚本默认 `VARIANT=g1`、`PHASE=finetune`（play/eval），可通过环境变量覆盖，例如：

```bash
VARIANT=paper PHASE=base GLOBAL_CONTEXT=off bash scripts/run_play.sh headless
```

## 关键实现位置

- AME 模型: [src/ame_g1/learning/actor_critic.py](src/ame_g1/learning/actor_critic.py)
- 观测/奖励/事件: [src/ame_g1/mdp/](src/ame_g1/mdp/)
- 地形生成: [src/ame_g1/terrains/](src/ame_g1/terrains/)
- 任务与 phase 注册: [src/ame_g1/config/variants.py](src/ame_g1/config/variants.py)
- Env 配置: [src/ame_g1/config/g1/env_cfgs.py](src/ame_g1/config/g1/env_cfgs.py)
- RL 配置: [src/ame_g1/config/g1/rl_cfg.py](src/ame_g1/config/g1/rl_cfg.py)

## 兼容性说明

- 支持同一 `variant`、同一 `global-context` 模式下的 `base -> finetune` checkpoint 加载
- 不保证跨 `variant` checkpoint 兼容
- `global-context` 开关必须与 checkpoint 架构一致
