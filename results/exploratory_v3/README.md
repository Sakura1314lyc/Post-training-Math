# Exploratory v3 results

本目录仅包含 Confirmatory v2 完成后的 post-hoc 探索性结果，不属于确认性主实验。

- `grpo_temp03_data42_train42_gen42_step50_dev_select_v3.json`：
  GRPO temperature=0.3、step-50 在固定 dev-select 上的完整评测结果。
- `sft_vs_grpo_temp03_step50_transition_analysis.json`：
  SFT seed42 step-0 与低温 GRPO 的逐题配对比较。
- `grpo_temp09_vs_temp03_step50_transition_analysis.json`：
  temperature=0.9 与 0.3 两个 step-50 策略的逐题配对比较。

冻结配置和机器可读摘要见
`configs/experimental/grpo_v3_temperature03_pilot.json`，中文解释见
`reports/GRPO_V3_EXPLORATORY_LOG_zh.md`。

本轮只使用已消费的 dev-select；没有查看 dev-audit、GSM8K test 或 SVAMP。
