# GSM8K 大模型后训练实验

[English](README.md) | [简体中文](README_zh.md)

本项目完成了课程代码实战中的 SFT 与 On-Policy Distillation（OPD/GKD）：

> `Qwen/Qwen2.5-Math-1.5B`（Raw Math Base）→ GSM8K LoRA-SFT →
> Instruct 教师引导的 on-policy distillation → validation 选型 → 官方 test 配对评测

完整分析见 [SFT 中文报告](reports/SFT_EXPERIMENT_REPORT_zh.md)和
[OPD 中文报告](reports/OPD_EXPERIMENT_REPORT_zh.md)。

## 最终结果

| GSM8K 官方 test（1,319 题） | Raw Base | SFT v7 | OPD step 30 |
|---|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | **72.33%** |
| 严格 `####` 准确率 | 0.00% | 71.65% | **72.25%** |
| 格式遵循率 | 0.00% | **98.26%** | 97.73% |
| 达到 1,024 token 上限 | 3.11% | **1.36%** | 1.90% |
| 平均生成 token | 371.74 | **102.30** | 111.24 |
| 全量评测时间 | 179m 0s | 64m 4s | 75m 0s |

逐题配对中，Base 独自答对 187 题，SFT 独自答对 185 题，精确双侧 McNemar
`p=0.958659`。OPD 相比 SFT 为 SFT-only 49、OPD-only 58，净提升 9 题、
`+0.68 pp`，但 McNemar `p=0.439440`。因此本实验不能声称 SFT 或 OPD 显著提高
了数值推理准确率；可靠结论是 SFT 明显改善格式与效率，而 OPD 获得了当前最高的
test 点估计，但伴随轻微格式、截断和长度退化。

## 最终方案

- 初始模型：`Qwen/Qwen2.5-Math-1.5B`
- 框架：LLaMA-Factory + PEFT LoRA
- 配置：[`configs/main/qwen25_math_15b_base_lora_sft_v7.yaml`](configs/main/qwen25_math_15b_base_lora_sft_v7.yaml)
- 训练数据：`data/gsm8k_sft_clean.json`，共 7,473 条
- LoRA：`q_proj,v_proj,lm_head`，rank 8，alpha 16
- 训练：1 epoch，学习率 `2e-5`，seed 42
- 最终 adapter：`outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888`
- 评分器：`gsm8k_numeric_v3`
- OPD 教师：`Qwen/Qwen2.5-Math-1.5B-Instruct`，NF4 4-bit 冻结
- OPD：TRL GKD，`lmbda=1.0`、`beta=0.5`，50 optimizer steps / 200 rollouts
- 最终 OPD adapter：`outputs/opd/qwen25_math_15b_gkd_pilot50/checkpoint-30`

v7 数据删除了 GSM8K 答案中的 23,716 个 `<<expression=result>>` 计算器标注，
同时逐条验证 `####` 最终答案不变。相比使用原始标注的 v3，v7 在 validation 上
由 80.75% 恢复到 85.29%。

## 项目结构

```text
configs/main/               # 主实验 v1、v2、v3、最终 v7
configs/archive/math_15b/   # v4-v6 失败消融配置
configs/archive/            # 更早的 Instruct/0.5B 实验
data/                       # 训练数据与 LLaMA-Factory 数据注册
scripts/                    # 数据准备、评测、重评分、配对分析
results/dev/                # validation、smoke 与消融结果
results/final/              # 最终官方 test 结果
results/opd/                # 教师、OPD validation/test 与配对分析
results/archive/            # 历史 continued-SFT 结果
reports/                    # 完整实验报告
tests/                      # 单元测试
outputs/                    # 本地 checkpoint，不提交 Git
```

## 复现实验

进入项目并激活环境：

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft
```

### 1. 重新生成清洗数据

仓库已经包含生成后的数据；需要验证数据处理时运行：

```bash
python3 scripts/prepare_clean_sft_data.py \
  --input data/gsm8k_sft_formal.json \
  --output data/gsm8k_sft_clean.json \
  --overwrite
```

预期输出为 7,473 条样本、删除 23,716 个计算器标注。

### 2. 训练最终 SFT

```bash
llamafactory-cli train configs/main/qwen25_math_15b_base_lora_sft_v7.yaml
```

训练日志中的 `eval_loss` 是 token-level teacher-forcing loss，不能单独代表自由生成时
的数学准确率。checkpoint 选择必须使用完整 validation 生成评测。

### 3. Validation 评测

Raw Base：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split train_validation \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/dev/base/dev_math_base_15b_v3.json
```

SFT v7：

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888 \
  --eval-split train_validation \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/dev/sft_v7/dev_math_base_sft_v7_15b_ckpt888.json
```

这里显式使用 `--no-stop-after-answer-line`，目的是观察模型是否会通过自己的 EOS 正常
停止，而不是依赖评测器看到 `####` 后强制截断。

### 4. 官方 Test

最终结果已经生成并保存在 `results/final/`。如果从头复现，先运行 Base，再使用完全
相同的生成参数运行 v7；不要用 test 继续选择参数。脚本默认拒绝覆盖已有结果，重复
运行时请更换 `--output`，或在确认后显式加入 `--overwrite`。

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/final/test_gsm8k_base_15b_v3.json

python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/qwen25_math_15b_base_lora_sft_v7/checkpoint-888 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json
```

### 5. OPD/GKD Pilot

OPD 脚本自动复现 SFT 的固定 374 题 validation 并将其排除，避免训练泄漏。学生保持
BF16 LoRA，教师使用 NF4 4-bit；`outputs/` 中的 checkpoint 不提交 Git。

```bash
python scripts/train_opd_gkd.py \
  --output-dir outputs/opd/qwen25_math_15b_gkd_pilot50 \
  --num-samples 256 \
  --max-steps 50 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --max-new-tokens 256 \
  --lmbda 1.0 \
  --beta 0.5 \
  --save-steps 10 \
  --save-total-limit 5 \
  --save-final-adapter
```

最终 OPD test 评测：

```bash
python scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_pilot50/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/opd/final/test_gsm8k_gkd_pilot50_ckpt30_v3.json
```

### 6. 配对分析

```bash
python3 scripts/compare_base_sft.py \
  --base results/final/test_gsm8k_base_15b_v3.json \
  --sft results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json \
  --output results/final/test_base_sft_v7_ckpt888_transition_analysis.json
```

## 评测指标

- 数值准确率：判断最终数值是否正确，不要求固定输出格式。
- 严格准确率：答案正确并以 `#### <answer>` 结尾。
- 格式遵循率：是否以可解析的 `#### <answer>` 结尾。
- 终止统计：区分原生 EOS、答案行提前停止、token 上限和其他情况。
- McNemar 检验：利用同题配对结果判断正确率变化是否具有方向性证据。

分别报告这些指标，可以避免把“推理正确”“格式对齐”和“生成能正常结束”混为一谈。

## 测试

```bash
python3 -m unittest discover -s tests -v
```

## 历史结果

早期 `Qwen2.5-1.5B-Instruct` continued-SFT、v1-v6 失败消融和调试结果均保留，
用于展示负结果与实验迭代，不应替代最终 Math Base → SFT v7 结论。
