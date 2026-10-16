# 开发与维护

`main` 保持可用；`stable/ame1-community` 保留升级前基线。
当前升级在 `upgrade/mjlab-1.6` 上进行，经审查与验证后才合入主线。
本轮没有新版本发布，也没有默认算法切换。

## 环境与检查

```bash
uv sync --locked
PYTHONPATH=tests uv run --locked python -m unittest \
  test_alignment test_cli test_variant_configs \
  test_observation_contract test_runner_save_policy
```

仿真与命令入口检查：

```bash
PYTHONPATH=tests uv run --locked python -m unittest test_smoke test_cli_modules
```

仿真检查会创建环境并编译 Warp 内核，首次运行较慢。GPU 短训练使用项目现有
`ame-train` 入口，显式选择可用 GPU、小批量环境、少量迭代和 tensorboard 日志；
不要在共享机器上改动别人的环境或运行中的任务。

完整测试也可显式使用一张 GPU（下面的索引应替换为实际可用卡）：

```bash
CUDA_VISIBLE_DEVICES=0 AME_TEST_DEVICE=cuda:0 OMP_NUM_THREADS=2 \
  uv run --locked python -m unittest discover -s tests -v
```

## 改动约定

- 保持清晰的模块职责，优先复用上游接口。只为实际差异引入配置或适配代码。
- 修改训练语义时列明观测、动作、奖励、课程或优化器的变化。
- 保留命令和任务入口；必要的破坏性变化需要迁移说明。
- checkpoint 兼容性分别报告权重加载、推理与恢复训练，不以加载成功替代效果评估。
- 复用已有测试，只增加能捕获具体风险的验证。不要为每个配置字段重复编写测试。
- PR 说明实际执行的检查与未验证范围；长训练效果需有配置、种子和结果来源。
- 依赖变更提交 `pyproject.toml` 和 `uv.lock`；不提交模型、训练日志或凭据。

发布前更新 [CHANGELOG](CHANGELOG.md) 和相应迁移指南。预览能力与已完成完整训练
评估的能力明确区分。默认配置的调整通过独立变更说明与实验结果评审。
