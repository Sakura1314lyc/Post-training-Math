# SVAMP 独立泛化评测

## 固定协议

- 数据：[`MU-NLPC/Calc-svamp`](https://huggingface.co/datasets/MU-NLPC/Calc-svamp/blob/main/README.md)，`default/test`，共 1,000 题。
- 说明：原始 [`SVAMP`](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json)
  没有官方 train/test 划分，按 Calc-SVAMP 数据卡将完整集合视为 test。
- 评分器：`svamp_numeric_v1`。
- 解码：greedy、原生 EOS、`max_new_tokens=1024`，不启用答案行提前停止。
- 模型：Raw Base、SFT v7，以及 OPD seed 42/43/44 的固定 checkpoint-30。
- 约束：只允许 10 题 smoke 检查运行链路；完整结果不得用于选择 checkpoint、prompt
  或超参数。

## 先运行 smoke

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split test \
  --num-samples 10 \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/smoke/svamp_base_15b_smoke10_v1.json
```

只确认能加载 10 个样本、能生成 JSON、终止统计合理；不要根据这 10 题准确率改变正式
协议。

## 完整评测命令

Raw Base：

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/final/svamp_base_15b_v1.json
```

SFT v7：

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/final/svamp_sft_v7_15b_ckpt888_v1.json
```

OPD seed 42：

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_pilot50/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/final/svamp_opd_seed42_ckpt30_v1.json
```

OPD seed 43：

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_seed43/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/final/svamp_opd_seed43_ckpt30_v1.json
```

OPD seed 44：

```bash
python3 scripts/eval_sft_adapter.py \
  --benchmark svamp \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_seed44/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/svamp/final/svamp_opd_seed44_ckpt30_v1.json
```

## 最终结果（2026-08-21）

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 | OPD seed 44 | OPD 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 82.30% | 81.70% | 81.80% | 81.93% ± 0.32 |
| 严格 `####` 准确率 | 0.00% | 81.40% | 82.10% | 81.60% | 81.70% | 81.80% ± 0.26 |
| 格式遵循率 | 0.00% | **98.80%** | 98.70% | 98.40% | 98.30% | 98.47% ± 0.21 |
| 达到 token 上限 | 7.00% | **0.90%** | 1.00% | 1.30% | 1.40% | 1.23% ± 0.21 |
| 平均生成 token | 290.81 | **55.07** | 59.04 | 61.67 | 63.23 | 61.32 ± 2.12 |
| 全量评测时间 | 109m 21s | **27m 32s** | 29m 23s | 34m 38s | 31m 28s | 31m 50s ± 2m 39s |

同题配对结果：

- Base→SFT：Base-only 110、SFT-only 73，差值 −3.70 pp，精确双侧 McNemar
  `p=0.00761528`。
- SFT→OPD seed 42：SFT-only 21、OPD-only 29，差值 +0.80 pp，`p=0.322236`。
- SFT→OPD seed 43：SFT-only 26、OPD-only 28，差值 +0.20 pp，`p=0.891923`。
- SFT→OPD seed 44：SFT-only 28、OPD-only 31，差值 +0.30 pp，`p=0.794844`。

最终结论：SFT 在独立 SVAMP 上显著降低数值准确率，但大幅改善格式、终止和效率；三个
OPD 运行相对 SFT 均为正向点估计，平均恢复 0.43 pp，但每次配对差异都不显著，而且 OPD
均值仍比 Raw Base 低 3.27 pp。详细分析见
[`../../reports/SVAMP_EXPERIMENT_REPORT_zh.md`](../../reports/SVAMP_EXPERIMENT_REPORT_zh.md)。

配对分析文件：

- `final/svamp_base_sft_v7_transition_analysis_v1.json`
- `final/svamp_sft_v7_opd_seed42_transition_analysis_v1.json`
- `final/svamp_sft_v7_opd_seed43_transition_analysis_v1.json`
- `final/svamp_sft_v7_opd_seed44_transition_analysis_v1.json`
- `final/svamp_opd_multiseed_summary_v1.json`
