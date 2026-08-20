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

## 阶段性结果（2026-08-20）

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 | OPD seed 44 |
|---|---:|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 82.30% | 81.70% | 待运行 |
| 严格 `####` 准确率 | 0.00% | 81.40% | 82.10% | 81.60% | 待运行 |
| 格式遵循率 | 0.00% | **98.80%** | 98.70% | 98.40% | 待运行 |
| 达到 token 上限 | 7.00% | **0.90%** | 1.00% | 1.30% | 待运行 |
| 平均生成 token | 290.81 | **55.07** | 59.04 | 61.67 | 待运行 |
| 全量评测时间 | 109m 21s | **27m 32s** | 29m 23s | 34m 38s | 待运行 |

同题配对结果：

- Base→SFT：Base-only 110、SFT-only 73，差值 −3.70 pp，精确双侧 McNemar
  `p=0.00761528`。
- SFT→OPD seed 42：SFT-only 21、OPD-only 29，差值 +0.80 pp，`p=0.322236`。
- SFT→OPD seed 43：SFT-only 26、OPD-only 28，差值 +0.20 pp，`p=0.891923`。

当前结论只能视为阶段性结果：SFT 在独立 SVAMP 上显著降低数值准确率，但大幅改善格式、
终止和效率；前两个 OPD 运行相对 SFT 均为正向点估计，但幅度小且没有统计显著性，仍需
完成预注册的 seed 44 后才能汇总三次均值与标准差。详细分析见
[`../../reports/SVAMP_EXPERIMENT_REPORT_zh.md`](../../reports/SVAMP_EXPERIMENT_REPORT_zh.md)。

配对分析文件：

- `final/svamp_base_sft_v7_transition_analysis_v1.json`
- `final/svamp_sft_v7_opd_seed42_transition_analysis_v1.json`
- `final/svamp_sft_v7_opd_seed43_transition_analysis_v1.json`
