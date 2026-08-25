# Exploratory v4 results

本目录保存 Confirmatory v2 完成后的低学习率 GRPO 探索性结果，不属于确认性主实验。

- `grpo_lr1e6_data42_train42_gen42_step50_dev_select_v3.json`：低学习率 treatment 的完整
  final step-50 dev-select 评测。
- `sft_vs_grpo_lr1e6_step50_transition_analysis.json`：treatment 与 SFT step-0 的预注册
  逐题配对比较。
- `grpo_lr5e6_vs_lr1e6_step50_transition_analysis.json`：treatment 与原 learning rate
  `5e-6` step-50 的预注册逐题配对比较。
- `grpo_temp03_vs_lr1e6_step50_transition_analysis.json`：用于核查 v3/v4 汇总数字巧合的
  额外诊断比较，不属于预注册主比较。

冻结协议、哈希和机器可读摘要见
`configs/experimental/grpo_v4_lr1e6_pilot.json`，中文解释见
`reports/GRPO_V4_EXPLORATORY_LOG_zh.md`。

本轮只使用已消费的 dev-select；没有查看 dev-audit、GSM8K test 或 SVAMP。
