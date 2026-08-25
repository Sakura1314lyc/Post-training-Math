# GSM8K 大模型后训练实验

[简体中文](README.md) | [English](README_en.md)

本项目完成了课程代码实战中的 SFT、On-Policy Distillation（OPD/GKD）与 GRPO
三随机种子实验：

> `Qwen/Qwen2.5-Math-1.5B`（Raw Math Base）→ GSM8K LoRA-SFT →
> Instruct 教师引导的 on-policy distillation / GRPO → validation 选型 →
> 官方 test 配对评测 → SVAMP 独立泛化评测

完整分析见 [SFT 中文报告](reports/SFT_EXPERIMENT_REPORT_zh.md)和
[OPD 中文报告](reports/OPD_EXPERIMENT_REPORT_zh.md)。独立泛化评测见
[SVAMP 泛化报告](reports/SVAMP_EXPERIMENT_REPORT_zh.md)，GRPO 完整分析见
[GRPO 中文报告](reports/GRPO_EXPERIMENT_REPORT_zh.md)。

针对旧实验的单 SFT seed、validation 反复查看、OPD/GRPO 小预算、seed 混杂和
GRPO KL reference 限制，仓库已新增一套与旧结论分开的
[confirmatory v2 修正协议](reports/CONFIRMATORY_V2_PROTOCOL_zh.md)；具体配置和命令见
[`configs/confirmatory/`](configs/confirmatory/README.md)。下表仍代表早期探索性实验，
不能与确认性结果混写。

Confirmatory v2 的 dev-select 阶段已完成：SFT 三 seed 为 `83.78% ± 1.47 pp`；
OPD 三 seed 为 `82.80% ± 1.32 pp`，但格式合规率仅 `2.58%`；使用正确 SFT KL
reference 的 GRPO 三 seed 为 `78.61% ± 0.53 pp`，三次均显著低于规范 SFT seed42。
因此 OPD/GRPO 均在 audit 前被拒绝。预先指定的 SFT seed42 随后仅进行一次
`dev_audit`，得到 `81.02%`；相比 dev-select 低 2.67 pp，但独立两比例检验不显著
（`p=0.3375`）。当前协议已经结束，audit 不再用于调参。机器可读汇总见
[`results/confirmatory_v2/confirmatory_v2_progress.json`](results/confirmatory_v2/confirmatory_v2_progress.json)。

## 最终结果

| GSM8K 官方 test（1,319 题） | Raw Base | SFT v7 | OPD 三次均值 ± SD | GRPO 三次均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | **72.91% ± 0.54** | 72.18% ± 0.35 |
| 严格 `####` 准确率 | 0.00% | 71.65% | **72.83% ± 0.54** | 72.10% ± 0.40 |
| 格式遵循率 | 0.00% | **98.26%** | 97.68% ± 0.16 | 98.13% ± 0.09 |
| 达到 1,024 token 上限 | 3.11% | **1.36%** | 1.95% ± 0.16 | 1.42% ± 0.04 |
| 平均生成 token | 371.74 | **102.30** | 111.60 ± 1.34 | 102.99 ± 1.06 |

逐题配对中，Base 独自答对 187 题，SFT 独自答对 185 题，精确双侧 McNemar
`p=0.958659`。OPD seed 42/43/44 的 test 准确率分别为 72.33%、73.39%、
73.01%，均高于 SFT，三次均值为 `72.91% ± 0.54 pp`。对应 SFT→OPD 的
McNemar `p` 分别为 0.439440、0.050487 和 0.117213，均未低于 0.05；三次
validation 也都低于 SFT。因此可靠结论是：SFT 明显改善格式与效率，OPD 的 test
收益具有跨运行的方向一致性，但仍没有充分统计证据证明稳定提升，并伴随格式、截断和
长度退化。GRPO seed 42/43/44 分别达到 72.25%、72.48% 和 71.80%，均值为
`72.18% ± 0.35 pp`，相对 SFT 平均提高 0.53 pp；三次配对检验均不显著。其 SVAMP
均值为 `81.20% ± 0.36 pp`，反而比 SFT 低 0.30 pp，因此同样没有跨数据集提升证据。

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
- OPD 报告 checkpoint：三个运行均固定使用 step 30，位于
  `outputs/opd/qwen25_math_15b_gkd_{pilot50,seed43,seed44}/checkpoint-30`
- GRPO：TRL GRPO，数值/格式奖励权重 1.0/0.1，`beta=0`，30 steps /
  120 rollouts；seed 42/43/44 均固定 checkpoint-30

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
results/svamp/              # 独立 SVAMP 泛化评测协议与结果
results/grpo/               # GRPO validation/test、配对分析与多随机种子汇总
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

seed 42 的 OPD test 评测：

```bash
python scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_pilot50/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/opd/final/test_gsm8k_gkd_pilot50_ckpt30_v3.json
```

seed 43/44 使用完全相同的超参数和固定 checkpoint-30。历史实验使用的旧 `--seed`
同时控制 256 条训练样本抽取与训练随机性，因此这里只能报告端到端运行波动，而不是
固定数据下的纯训练 seed 波动。新脚本已经提供 `--data-seed`、`--training-seed` 和
`--generation-seed` 三个独立参数供 confirmatory v2 使用。三次历史结果汇总见
[`opd_ckpt30_multiseed_summary.json`](results/opd/final/opd_ckpt30_multiseed_summary.json)。

### 6. 独立 SVAMP 泛化评测

为避免继续围绕 GSM8K test 调整实验，额外使用
[`MU-NLPC/Calc-svamp`](https://huggingface.co/datasets/MU-NLPC/Calc-svamp/blob/main/README.md)
的 `default/test` 全部 1,000 题做一次固定协议评测。原始
[`SVAMP`](https://github.com/arkilpatel/SVAMP/blob/main/SVAMP.json) 没有官方
train/test 划分，因此这里遵循 Calc-SVAMP 数据卡，将完整集合视为 test；该版本还修正了
原数据中一条方程与答案不一致的样本。

先只运行 10 题 smoke，检查数据加载、输出格式和 EOS，不根据结果修改 checkpoint、prompt
或生成参数：

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

smoke 正常后，用同一协议依次评测 Raw Base、SFT v7，以及 OPD/GRPO seed 42/43/44；
正式命令与文件命名见 [`results/svamp/README.md`](results/svamp/README.md)。所有八个运行固定使用
贪心解码、原生 EOS 和 1,024 token 上限，SVAMP 结果只用于最终泛化报告，不再用于选
checkpoint 或调参。SVAMP 结果的评分器版本为 `svamp_numeric_v1`。

截至 2026-08-22，固定协议中的八组评测已经全部完成：

| SVAMP test（1,000 题） | Raw Base | SFT v7 | OPD 三次均值 ± SD | GRPO 三次均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | **85.20%** | 81.50% | 81.93% ± 0.32 | 81.20% ± 0.36 |
| 严格准确率 | 0.00% | 81.40% | 81.80% ± 0.26 | 81.07% ± 0.38 |
| 格式遵循率 | 0.00% | **98.80%** | 98.47% ± 0.21 | 98.63% ± 0.25 |
| 达到 token 上限 | 7.00% | **0.90%** | 1.23% ± 0.21 | 1.10% ± 0.17 |
| 平均生成 token | 290.81 | **55.07** | 61.32 ± 2.12 | 57.25 ± 2.02 |

Base→SFT 的差值为 −3.70 pp，配对 McNemar `p=0.00761528`；SFT→OPD seed
42/43/44 分别为 +0.80/+0.20/+0.30 pp，`p=0.322236/0.891923/0.794844`。
因此 SFT 的跨数据集数值准确率显著下降，同时格式、终止和速度明显改善；OPD 三次方向
一致但只平均恢复 0.43 pp，GRPO 相对 SFT 平均下降 0.30 pp。两者均无单次显著提升，
且都未恢复 Raw Base 水平。完整分析见
[`SVAMP_EXPERIMENT_REPORT_zh.md`](reports/SVAMP_EXPERIMENT_REPORT_zh.md)。

### 7. 配对分析

```bash
python3 scripts/compare_base_sft.py \
  --base results/final/test_gsm8k_base_15b_v3.json \
  --sft results/final/test_gsm8k_sft_v7_15b_ckpt888_native_v3.json \
  --output results/final/test_base_sft_v7_ckpt888_transition_analysis.json
```

### 8. GRPO

当前 LLaMA-Factory checkout 没有 GRPO 训练入口，因此使用原生 TRL 0.24 脚本
[`train_grpo.py`](scripts/train_grpo.py)。脚本从 SFT v7 adapter 继续训练，复现并排除固定
374 题 validation，以数值正确性为主奖励、严格 `####` 格式为辅助奖励。它使用
Transformers 原生生成而不是 vLLM，并包含 TRL 0.24 与本地 Transformers 5.x 的可选依赖
探测兼容层。

8 GiB GPU 首先只运行 1 optimizer step：

```bash
python3 scripts/train_grpo.py \
  --output-dir outputs/grpo/qwen25_math_15b_grpo_smoke \
  --num-samples 8 \
  --max-steps 1 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 128
```

当前实现固定 `beta=0`。原因是直接继续训练现有 PEFT adapter 时，TRL 禁用 adapter 得到
的是 Raw Base，而不是初始 SFT policy；使用非零 KL 会引用错误的策略。

smoke 已成功完成：7.49 秒训练、峰值分配显存 3.33 GiB、奖励和指标有限、LoRA 参数确实
更新。随后 seed 42/43/44 均完成 30 steps / 120 rollouts，单次约 137–141 秒。固定
checkpoint-30 随后使用与 SFT/OPD 完全一致的 1,024-token、原生 EOS 正式协议评测：

| GRPO 正式 validation（374 题） | seed42 | seed43 | seed44 | 三次均值 ± SD |
|---|---:|---:|---:|---:|
| 数值准确率 | 85.83% | 85.83% | 85.29% | 85.65% ± 0.31 |
| 严格准确率 | 85.83% | 85.83% | 85.29% | 85.65% ± 0.31 |
| 格式遵循率 | 99.20% | 99.73% | 99.47% | 99.47% ± 0.27 |

正式 GSM8K test 的 seed 42/43/44 数值准确率为 72.25%、72.48% 和 71.80%，
三次均值 `72.18% ± 0.35 pp`；相对 SFT 的差值分别为 +0.61、+0.83 和 +0.15 pp，
McNemar `p` 分别为 0.291215、0.168978 和 0.885433。方向一致但单次均不显著。
SVAMP 三次均值为 `81.20% ± 0.36 pp`，比 SFT 低 0.30 pp，也未显示跨数据集收益。
完整协议、结果和逐题配对分析见
[`GRPO_EXPERIMENT_REPORT_zh.md`](reports/GRPO_EXPERIMENT_REPORT_zh.md)与
[`results/grpo/README.md`](results/grpo/README.md)。

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
