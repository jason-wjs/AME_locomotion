# mjlab 1.6 升级与 AME-2 迁移盘点

本轮升级底座，保留 `g1` 默认变体、`paper` 对照、`base/finetune` 流程和
`global-context` 开关。AME-2 不在本轮引入，也不切换默认算法或发布新版本。

## 基线与维护方式

2026-10-16 检查结果：远端只有未保护的 `main` 分支，没有 tag、GitHub Release
或仓库内的 GitHub Actions 工作流。已有七个 unittest 测试文件；文档为 README
和作者的 `AME_guide.md`。原 `.gitignore` 忽略其他 docs 文件，本轮解除这一限制。

- 稳定源码：`99b8aa45435832af90f2f280a8f9ce757bee2409`，保留为
  `stable/ame1-community` 分支，不宣称它已通过本轮完整效果验证。
- 升级分支：`upgrade/mjlab-1.6`。完成审查与验收后才考虑合入 `main`。
- 官方参照：`leggedrobotics/ame2_minimal` 的 `8beb9a6`。
- 发布版本锁定依赖；升级通过独立分支与 PR 进行。初期只维护稳定线和升级线，
  不增加多条长期支持分支。

| 环境 | mjlab | MuJoCo / MuJoCo Warp | RSL-RL |
| --- | --- | --- | --- |
| 稳定源码的 uv.lock | 1.2.0 | 3.6.0 / 3.6.0 | 5.0.1 |
| 本轮目标 | 1.6.0 | 3.11.x / 3.11.x，精确版本见 uv.lock | 5.4.2 |

`uv sync --locked` 安装已提交的依赖集合。README 原先将宽版本范围描述为自动
使用最新兼容版本；已有 lock 时这不成立。稳定基线的 lock 也应保留使用。

## 实现差异与处理决定

这里比较的是实际代码，不把社区改进的收益直接推广到 AME-2。用户报告部分
设计优于论文 AME-1；本轮没有可用于统一比较的多种子完整实验结果。

| 模块 | 当前社区衍生实现 | 官方 AME-2 已发布实现 | 决定 |
| --- | --- | --- | --- |
| 任务 | `twist` 速度跟踪 | 安全地形目标位姿采样、到达目标与朝向奖励 | 本轮保留；AME-2 作为独立任务迁移 |
| 地图 | `paper` 为 17×11，`g1` 为 33×21；传感器相对高度、裁剪到 [-1.2, 0]；未命中先填负最大距离 | 显式网格坐标与传感器偏移、高度 offset=0.74；空洞按有效最低高度加随机偏置填充，裁剪与噪声逻辑不同 | 本轮保留；先验证坐标和排列，再接 AME-2 |
| 局部特征 | `paper` 用 z 输入后拼 xyz；`g1` 用 xyz 输入、stride=2 下采样 | 非坐标通道 CNN 与逐点全通道 FC 拼接后投影到 96 维 | 保留现有设计；下一轮逐项消融 |
| 注意力 | 64 维、16 heads；可选全局分支 | 96 维、32 heads；本体编码与全局 max-pool 特征共同生成 query | 新增官方参照，不能把现有 global-context 开关视为官方 AME-2 |
| Critic | 独立 query、共享局部编码与注意力的 AME critic；单个 value | 16-expert MoE critic，支持按奖励分量的 value heads | 后续连同 storage、returns 和 value loss 一起迁移 |
| PPO | Adam、标量奖励、actor/critic 对称数据增强 | 默认 Muon、bf16、熵衰减、多头价值训练、critic-only 对称增强 | 本轮不改学习配方 |
| 动作 | 继承 mjlab G1 的关节位置动作和逐关节 scale | 延迟关节位置动作，scale=0.25、delay_range=(0,1) | 后续对齐关节顺序、PD、缩放和延迟后再比较策略 |
| 地形 | base 四类，finetune 八类；8×4 m；训练总预算 40 tiles | 多地形目标到达，12×12 m，带目标 flat-patch 采样 | 保留旧地形；AME-2 单独迁移几何与采样，不只复用名称 |
| 课程与随机化 | base→finetune；后者加入本体噪声、地图噪声/漂移与动力学扰动 | 目标任务课程、地图噪声课程、不同终止及随机化配方 | 后续逐项记录单位、时序和 reset 行为 |
| gaze / student | Gaze 已通过 CPU 数值/闭环、GPU 容量和官方权重参考评估；自身学习与完整 GPU 集成验收进行中。student 未实现，见 [Gaze 迁移](gaze-migration.md) | 可选 gaze；LSIO/student 代码存在，但 neural mapping 模型尚未提供 | 后续独立能力；不宣称已完成感知闭环 |

对应源码：

- 当前：[模型](../src/ame_g1/learning/actor_critic.py)、
  [环境](../src/ame_g1/config/g1/env_cfgs.py)、
  [RL 配置](../src/ame_g1/config/g1/rl_cfg.py)、
  [地图](../src/ame_g1/mdp/observations.py)、[地形](../src/ame_g1/terrains/config.py)。
- 官方：[环境与任务](https://github.com/leggedrobotics/ame2_minimal/blob/8beb9a6/ame2/ame2/tasks/ame2_env_cfg.py)、
  [模型](https://github.com/leggedrobotics/ame2_minimal/blob/8beb9a6/rsl_rl/rsl_rl/models/AME2_models.py)、
  [训练配置](https://github.com/leggedrobotics/ame2_minimal/blob/8beb9a6/ame2/ame2/tasks/agents/rsl_rl_ppo_cfg.py)、
  [发布范围](https://github.com/leggedrobotics/ame2_minimal/blob/8beb9a6/README.md)。

## 兼容边界

命令入口、任务名、默认变体和模型参数布局保持不变。同一变体与 global-context
配置下，分别验证权重加载、推理、恢复训练；跨变体或跨 global-context 的
checkpoint 不承诺兼容。底层物理引擎升级后，即使同一输入的策略输出一致，
也不能据此保证轨迹、收敛速度或最终通过率不变。

本轮只支持锁定的 mjlab 1.6.0，不添加多版本运行时探测或静默降级逻辑。
`CollisionCfg`、command 的局部 reset 接口与 viewer 变动按实际使用点检查，
不会仅因上游列出 breaking change 就增加没有调用者的兼容层。

具体适配：

- 显式保持 `rel_forward_envs=0` 和射线碰撞组 `(0, 1, 2)`，防止继承的新默认值
  改变旧命令与地图分布。移除本任务没有使用的上游新增 `foot_height_scan`。
- 保留上游提高后的接触容量 `nconmax=70` 及局部 reset/课程初始化修复。
  G1 的关节顺序、动作 scale 和 PD 常量在所检查的两个上游版本之间没有变化。
- global-context 检查读取真实的 `actor_state_dict`。基线使用的 RSL-RL 5.0.1
  和新版本 5.4.2 都采用这一格式；原检查错误地读取 `model_state_dict`。
- 修复 global-context JIT 导出中可选张量的类型标注，不改变网络参数布局。
- 将实体名称配置按新接口改为 tuple，消除 CLI 类型不匹配警告。
- runner 直接复用 `MjlabOnPolicyRunner.save`，保留本项目的保存周期和最终导出策略；
  删除没有 AME 调用者的 motion-tracking 下载分支。支持上游 `--log-root`，
  在算法构造修改配置之前保存配置快照。

## 验证与后续工作

验证命令见 [贡献说明](../CONTRIBUTING.md)。CI 运行现有的快速契约与配置检查；
环境 step/reset、短训练、checkpoint 恢复和导出在真实依赖上单独执行。
短训练成功仅证明链路工作，不代表 AME-2 复现或迁移后性能等价。

2026-10-16 已验证：Python 3.11、PyTorch 2.11.0、Warp 1.18.0、H100。
`uv sync --locked --offline` 成功；现有完整测试集 26 项通过，包括四组变体/阶段的
观测和局部 reset、真实 PPO 更新、base→finetune 严格加载及 global-context JIT。
已有测试只补充了具体升级风险的覆盖，没有增加平行的测试框架。

另外使用稳定源码和 RSL-RL 5.0.1 生成 `paper`、`g1`、`g1 + global-context`
三种 checkpoint（含非空 Adam 状态），在 5.4.2 上严格加载。同一输入的推理输出
以及继续一次相同梯度更新后的输出一致。该探测使用相同 PyTorch 环境和合成观测，
不涉及旧物理引擎轨迹，也不是已有收敛策略的性能评估。
三种配置的 JIT 和 ONNX（固定 batch=1）导出均通过输出数值对照；ONNX 使用
reference evaluator 验证，不据此宣称已验证真机推理后端。

实际 CLI 短训练：`g1 + global-context on`，8 个环境、每次 6 个 rollout steps、
每次 1 个 PPO epoch，base 完成 2 次更新，再通过 `--checkpoint-file` 在 finetune
完成 1 次更新。两次运行都成功保存 checkpoint 和最终 ONNX，`--log-root` 指向的
目录内保留了环境/算法配置。日志与一次性跨版本探测位于本地忽略的
`artifacts/upgrade/`，不会随代码提交或上传。

下一轮先接官方 AME-2 teacher，再对社区改进做单项与组合实验。gaze、student、
neural mapping 按可用依赖分别验收。完整效果评估需要固定地形/命令分布、训练预算
和多个随机种子；本轮不启动长训练。

评估入口还需要独立改进：当前 `ame-eval` 复用 play 的近乎无限 episode 时长，
存活策略可能长时间无法完成指定 episode 数；`avg_tracking_error` 目前是每条
episode 的误差累计再平均，受 episode 长度影响。正式效果对照前应定义有限时长、
统一归一化的评估协议。本轮保留现有入口，没有将这些旧行为计为效果验证通过。
