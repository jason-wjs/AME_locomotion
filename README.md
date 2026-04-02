# AME Reproduction for Unitree G1 on mjlab

## 项目简介

本仓库是对论文 [*Attention-Based Map Encoding for Learning Generalized Legged Locomotion*](https://www.science.org/doi/10.1126/scirobotics.adv3604) 中基于注意力的地形编码器方法在Unitree G1 (29DoF)上的复现，使用 `mjlab`作为训练引擎。

代码同时提供两个显式变体：

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

训练流程采用 `phase` 语义：

- `base`：基础训练
- `finetune`：在 `base` checkpoint 上微调

## 依赖与安装

- Python: `>=3.10,<3.14`
- mjlab: `>=1.2.0,<2`（会使用已发布最新兼容版本；当前最新为 1.2.0）

```bash
uv sync
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

- `--variant`: `paper | g1`
- `--phase`: `base | finetune`
- `--global-context`: `on | off`（`paper` 仅支持 `off`）

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
