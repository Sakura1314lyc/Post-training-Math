# GRPO v3 探索性消融日志

## 低温 rollout pilot

本实验在 Confirmatory v2 完成后开展，属于 post-hoc exploratory pilot，不修改已有确认性
结论。它只改变 rollout temperature：从 0.9 降到 0.3；初始策略、训练数据子集、三个
随机种子、学习率、KL、奖励、completion 长度和 50-step 预算保持不变。

预注册假设：较低温度会减少 256-token 截断，并可能缓解原 seed42 在前 50 steps 已出现的
dev-select 退化。主要指标是 dev-select 数值准确率；截断、格式与生成长度是诊断指标。

本 pilot 只使用已经消费的 dev-select。禁止查看或使用 dev-audit、GSM8K test、SVAMP；
单 seed 结果不能作为稳定改进结论。

冻结配置见 `configs/experimental/grpo_v3_temperature03_pilot.json`。结果待运行后补充。
