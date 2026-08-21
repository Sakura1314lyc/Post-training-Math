# GRPO results / GRPO 结果

本目录记录从最终 SFT v7 adapter 继续进行的 GRPO 实验。训练脚本为
[`scripts/train_grpo.py`](../../scripts/train_grpo.py)，checkpoint 保存在被 Git 忽略的
`outputs/grpo/`，这里只提交评测结果和机器可读汇总。

## 当前状态（2026-08-21）

- 1-step smoke 已通过：奖励、梯度、参数更新、保存和 8 GiB 显存链路均正常；
- seed 42/43/44 的 30-step 训练均完成，每次 256 个候选 prompt、4 generations、
  30 optimizer steps / 120 rollouts；
- 三次固定 374 题 pilot validation 已完成；
- seed42 pilot test 已完成；seed43/44 test 留待下一实验日；
- 今天的评测使用 `max_new_tokens=512` 且启用答案行提前停止，与既有正式 SFT/OPD 的
  `1024 + 原生 EOS` 协议不同，因此全部移入 `pilot/`，不能直接作为最终同协议比较。

机器可读状态见 [`grpo_stage_summary.json`](grpo_stage_summary.json)，阶段分析见
[`GRPO_EXPERIMENT_REPORT_zh.md`](../../reports/GRPO_EXPERIMENT_REPORT_zh.md)。

## Pilot 结果

| 固定 validation（374 题） | seed42 | seed43 | seed44 | 三次均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | 86.10% | 85.56% | 85.03% | 85.56% ± 0.53 |
| 严格准确率 | 85.83% | 85.56% | 85.03% | 85.47% ± 0.41 |
| 格式遵循率 | 98.93% | 99.20% | 98.93% | 99.02% ± 0.15 |
| 达到 512-token 上限 | 0.53% | 0.00% | 0.53% | 0.36% ± 0.31 |

seed42 pilot test 为 `953/1319 = 72.25%`，严格准确率同为 72.25%，格式率 98.18%，
达到 512-token 上限 18 题。该数值只用于检查方向，不能直接与正式 SFT/OPD 报告做
最终配对结论。

## 下一实验日：正式同协议评测

validation 与 test 都必须重新使用正式协议：贪心解码、1,024 token 上限、禁用答案行
提前停止。评测 seed 固定为 42，以保持验证划分和生成随机状态一致；模型训练 seed 由路径
区分。

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft

for train_seed in 42 43 44; do
  time python3 scripts/eval_sft_adapter.py \
    --base-model Qwen/Qwen2.5-Math-1.5B \
    --adapter outputs/grpo/qwen25_math_15b_grpo_seed${train_seed}_pilot30/checkpoint-30 \
    --output results/grpo/dev/gsm8k_grpo_seed${train_seed}_ckpt30_validation_native_v3.json \
    --benchmark gsm8k \
    --eval-split train_validation \
    --validation-size 0.05 \
    --seed 42 \
    --max-new-tokens 1024 \
    --no-stop-after-answer-line \
    --dtype bfloat16
done
```

完成 validation 后，不再据结果修改参数；对三个已固定运行执行官方 test：

```bash
for train_seed in 42 43 44; do
  time python3 scripts/eval_sft_adapter.py \
    --base-model Qwen/Qwen2.5-Math-1.5B \
    --adapter outputs/grpo/qwen25_math_15b_grpo_seed${train_seed}_pilot30/checkpoint-30 \
    --output results/grpo/final/test_gsm8k_grpo_seed${train_seed}_ckpt30_native_v3.json \
    --benchmark gsm8k \
    --eval-split test \
    --seed 42 \
    --max-new-tokens 1024 \
    --no-stop-after-answer-line \
    --dtype bfloat16
done
```

## 目录

```text
pilot/dev/   # 今天的 512-token slice 与 validation，仅作 pilot
pilot/test/  # 今天的 seed42 512-token test，仅作 pilot
dev/         # 明日正式 1024-token validation（运行后生成）
final/       # 明日正式 1024-token test（运行后生成）
```

## 比较纪律

- checkpoint 固定为 step 30；不能根据官方 test 再选择 checkpoint 或参数；
- 当前训练 `--seed` 同时控制 256 条训练样本抽取与训练随机性，三次 SD 是端到端运行波动；
- 配对比较必须具有相同 evaluator、数据集、prompt 和 generation metadata；
- `compare_base_sft.py` 默认拒绝不同评测协议，防止再次混用 512/1024 token 结果。
