# AME Reproduction for Unitree G1 on mjlab

## 项目简介

本项目是对论文 [*Attention-Based Map Encoding for Learning Generalized Legged Locomotion*](https://www.science.org/doi/10.1126/scirobotics.adv3604) 在 Unitree G1 29DoF 机器人平台上的复现。

为兼顾论文对照与在 G1 上的实用效果，当前代码提供两个显式变体：

- **`paper`**：面向原论文的复现路径，在观测、CNN 与注意力结构上尽量对齐论文与相关 guide，用作对照基线。
  - `17 x 11` terrain grid
  - `z-only` CNN 输入
  - CNN 后拼接网格坐标
  - 不使用 CNN 下采样
  - 不使用全局上下文

- **`g1`**：默认变体，针对 Unitree G1 的工程实践做了调整；**不以逐条复现原论文为约束**。实现上借鉴了 [AME_Locomotion](https://github.com/SII-FUSC/AME_Locomotion)。

  1. **CNN 输入使用 xyz 坐标，而非仅 z 高度**
     - CNN 直接处理 xyz 三维输入。
     - CNN 输出后不再额外拼接坐标。
     - 我们观察到该设置通常有更好的训练表现。
     - 可能原因是：直接输入 xyz 能让网络更早学习到位置相关信息。

  2. **用 CNN 步长下采样降低注意力计算开销**
     - 提高高程图分辨率或扩大范围时，MHA 序列长度会显著增加。
     - 我们在 CNN 阶段通过步长下采样缩短后续注意力序列。
     - 测试中未观察到明显的最终策略性能下降，同时训练开销更低。

  3. **引入 AME2 中的全局上下文**
     - 我们加入了 AME2 提出的通过 MLP + max-pool 获取全局上下文信息的设计。
     - 测试发现该设计能够提升策略表现。
     - 代价是注意力权重的可解释性变差，并且训练开销明显增加。

## 方法实现位置

AME 的主要实现位于：

- [src/ame_g1/learning/actor_critic.py](src/ame_g1/learning/actor_critic.py)

环境观测、奖励和地形相关逻辑位于：

- [src/ame_g1/mdp/observations.py](src/ame_g1/mdp/observations.py)
- [src/ame_g1/mdp/rewards.py](src/ame_g1/mdp/rewards.py)
- [src/ame_g1/terrains/](src/ame_g1/terrains/)

任务变体与训练配置入口位于：

- [src/ame_g1/config/variants.py](src/ame_g1/config/variants.py)
- [src/ame_g1/config/g1/env_cfgs.py](src/ame_g1/config/g1/env_cfgs.py)
- [src/ame_g1/config/g1/rl_cfg.py](src/ame_g1/config/g1/rl_cfg.py)

## 依赖与安装

- Python: `>=3.10,<3.14`
- `mjlab` 依赖策略: `>=1.2.0,<2`

```bash
uv sync
```


## 快速开始

训练 stage1：

```bash
uv run ame-train --variant g1 --stage stage1
```

训练 stage2：

```bash
uv run ame-train --variant g1 --stage stage2 --checkpoint-file /abs/path/to/model.pt
```

切换到论文基线：

```bash
uv run ame-train --variant paper --stage stage1
```

可视化运行：

```bash
uv run ame-play --variant g1 --stage stage2 --checkpoint-file /abs/path/to/model.pt --viewer
```

评估：

```bash
uv run ame-eval --variant g1 --stage stage2 --checkpoint-file /abs/path/to/model.pt
```

attention 导出与可视化：

```bash
uv run ame-play \
  --variant g1 \
  --stage stage2 \
  --checkpoint-file /abs/path/to/model.pt \
  --dump-file artifacts/attention_dump.pt

uv run ame-plot-attention \
  --dump-file artifacts/attention_dump.pt \
  --output-dir artifacts/attention
```

## 训练脚本

仓库内提供了一个便捷脚本：

- [scripts/train_ame1_g1.sh](scripts/train_ame1_g1.sh)

默认使用 `g1` 变体：

```bash
bash scripts/train_ame1_g1.sh stage1
bash scripts/train_ame1_g1.sh stage2
```

切换到 `paper`：

```bash
VARIANT=paper bash scripts/train_ame1_g1.sh stage1
```

该脚本内部同样统一通过 `uv run` 调用训练入口。

## 项目结构

```text
src/ame_g1/
├── cli/         # 用户命令入口
├── config/      # 变体 / stage 注册与 env/rl 配置
├── learning/    # AME 模型、对称增强、runner hook
├── mdp/         # observations / rewards / events
└── terrains/    # 地形生成与配置
```

## 说明

- 默认研究路径是 `g1`
- `paper` 保留为显式对照基线
- 支持同一变体内 `stage1 -> stage2` 的 checkpoint 加载
- 不保证跨变体 checkpoint 兼容
