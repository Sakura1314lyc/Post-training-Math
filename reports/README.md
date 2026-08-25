# Reports / 实验报告

- [`SFT_EXPERIMENT_REPORT_zh.md`](SFT_EXPERIMENT_REPORT_zh.md)：
  Qwen2.5-Math-1.5B 在 GSM8K 上的完整 LoRA-SFT 实验报告，包含迭代过程、
  validation/test 配对结果、效率分析、负结果和下一步建议。
- [`OPD_EXPERIMENT_REPORT_zh.md`](OPD_EXPERIMENT_REPORT_zh.md)：
  中文 OPD/GKD 报告，包含教师资格检查、8 GiB 双模型方案、checkpoint 选择、
  validation/test 配对结果与统计解释。
- [`OPD_EXPERIMENT_REPORT.md`](OPD_EXPERIMENT_REPORT.md)：English OPD/GKD report.
- [`SVAMP_EXPERIMENT_REPORT_zh.md`](SVAMP_EXPERIMENT_REPORT_zh.md)：
  独立 SVAMP 泛化评测报告，包含固定协议、Base/SFT/OPD/GRPO 三随机种子结果、配对
  检验与泛化结论。
- [`GRPO_EXPERIMENT_REPORT_zh.md`](GRPO_EXPERIMENT_REPORT_zh.md)：
  完整 GRPO 报告，包含原生 TRL 实现、8 GiB smoke、三随机种子训练、正式 GSM8K
  validation/test、SVAMP 泛化与 SFT/OPD 配对分析。
- [`CONFIRMATORY_V2_PROTOCOL_zh.md`](CONFIRMATORY_V2_PROTOCOL_zh.md)：
  针对单 SFT seed、开发集选择偏差、小训练预算、seed 混杂和 GRPO KL reference
  问题的新一轮确认性实验协议。
- [`CONFIRMATORY_V2_GRPO_TRAJECTORY_zh.md`](CONFIRMATORY_V2_GRPO_TRAJECTORY_zh.md)：
  确认性实验结束后的 GRPO seed42 checkpoint 轨迹诊断；明确标记为 post-hoc，不能用于
  事后替换正式 step-200 结果。
- [`GRPO_V3_EXPLORATORY_LOG_zh.md`](GRPO_V3_EXPLORATORY_LOG_zh.md)：
  确认性协议结束后的单因素探索性消融日志；与主结论隔离，不使用 audit/test/SVAMP 调参。
