# Qwen2.5-Math-1.5B 在 GSM8K 上的 OPD/GKD 实验报告

## 1. 实验摘要

本实验从最终 SFT v7 adapter 出发，使用 `Qwen/Qwen2.5-Math-1.5B-Instruct`
作为冻结教师，通过 TRL `GKDTrainer` 实现 On-Policy Distillation（OPD）。学生在
训练时现场采样回答，教师在这些学生轨迹上提供 token-level 软分布，学生最小化广义
Jensen-Shannon divergence。设置 `lmbda=1.0`，保证每个训练 batch 都来自学生的
on-policy rollout。

最终固定评测 checkpoint-30，并用 seed 42/43/44 做三次端到端重复。在官方 GSM8K
test 上，三次 OPD 数值准确率分别为 72.33%、73.39% 和 73.01%，均高于 SFT v7
的 71.65%；均值与样本标准差为 **72.91% ± 0.54 个百分点**。对应的精确双侧
McNemar `p` 分别为 0.439440、0.050487 和 0.117213，均未低于 0.05。

三次固定 validation 准确率分别为 84.76%、84.22% 和 83.69%，均低于 SFT 的
85.29%。因此结果应表述为：**OPD 的 test 收益在三次运行中方向一致，但单次配对
检验均不显著，且 validation 方向相反；当前证据仍不足以证明稳定、显著的能力提升。**

OPD 还带来稳定的行为代价：三次平均格式遵循率为 `97.68% ± 0.16`，低于 SFT 的
98.26%；平均截断率为 `1.95% ± 0.16`，高于 SFT 的 1.36%；平均生成长度由 SFT
的 102.30 增至 `111.60 ± 1.34` token。

## 2. SFT 与 OPD 的区别

SFT 在固定的 `(prompt, reference response)` 上进行 teacher-forcing，要求学生模仿
单条目标轨迹。当前 OPD 的训练分布则由学生自己决定：

1. 学生根据 prompt 采样一条新回答；
2. 教师读取相同的 prompt 和学生回答；
3. 在学生实际访问到的 token 状态上计算教师与学生的概率分布差异；
4. 只更新学生现有的 LoRA 参数。

所以 OPD 的核心价值不是再模仿一遍 GSM8K 标准答案，而是让教师纠正学生当前真正会
生成的轨迹。它仍属于知识蒸馏目标，不应与带显式标量奖励、优势估计和策略梯度的
GRPO 混为一谈。

## 3. 实验设置

### 3.1 模型与软件

| 项目 | 设置 |
|---|---|
| 学生 Base | `Qwen/Qwen2.5-Math-1.5B` |
| 学生初始 adapter | SFT v7 `checkpoint-888` |
| 教师 | `Qwen/Qwen2.5-Math-1.5B-Instruct` |
| OPD 实现 | TRL `GKDTrainer` / `GKDConfig` |
| 学生精度 | BF16 |
| 教师精度 | bitsandbytes NF4 4-bit，BF16 compute |
| 可训练参数 | 2,317,312（0.1499%） |
| 设备 | NVIDIA GeForce RTX 5060 Laptop GPU，8 GiB |
| 主要版本 | PyTorch 2.12.1+cu130、Transformers 5.8.0、TRL 0.24.0、PEFT 0.18.1、bitsandbytes 0.50.1 |

学生和教师 tokenizer 词表完全一致。生成同时接受 Base EOS `151643` 和 Qwen ChatML
`<|im_end|>` `151645`，避免学生已经生成对话结束符却继续输出。

### 3.2 教师资格检查

教师先在与 SFT 相同的固定 374 题 validation 上评测：

| 模型 | 数值准确率 | 达到长度上限 |
|---|---:|---:|
| SFT v7 | 319/374（85.29%） | 2/374 |
| Instruct 教师 | 363/374（97.06%） | 1/374 |

逐题配对中，SFT-only 为 5，Teacher-only 为 49，精确双侧 McNemar
`p=3.89139e-10`。教师显著强于学生，满足蒸馏前提。教师使用 `\boxed{}` 而不是
`####`，所以其格式指标为 0；OPD 使用教师软分布，不要求直接复制教师最终字符串格式。

### 3.3 数据纪律

- 原始训练文件：`data/gsm8k_sft_clean.json`，7,473 条；
- 使用与 SFT 完全相同的 `test_size=0.05, seed=42`；
- 固定 validation：374 条，全部从 OPD 训练候选中排除；
- 可用 OPD 训练集：7,099 条；
- 每次运行从其中随机抽取 256 条；
- seed 42 在 validation 选定 checkpoint-30 后完成初次 test；seed 43/44 随后以
  相同超参数和固定 checkpoint-30 做鲁棒性复现，两个运行均完整评测，没有按
  validation 或 test 表现筛选。

脚本已验证 OPD 排除的 374 个 source index 与既有 Base/SFT validation 顺序和集合
完全一致，训练/validation 重叠为 0。

### 3.4 Pilot 超参数

| 参数 | 值 |
|---|---:|
| `max_steps` | 50 |
| micro batch | 1 |
| gradient accumulation | 4 |
| 实际 rollout 数 | 200 |
| 学习率 | `5e-6`，constant |
| `max_length` | 512 |
| `max_new_tokens` | 256 |
| sampling temperature | 0.9 |
| `lmbda` | 1.0（完全 on-policy） |
| `beta` | 0.5（广义 JSD） |
| `seq_kd` | false |
| checkpoint 间隔 | 10 steps |

训练耗时 `998.613` 秒（约 16 分 39 秒），峰值显存 `6.776/7.960 GiB`。平均
train loss 为 `0.31605`；前 10 步均值 `0.33109`，后 10 步均值 `0.28776`。
on-policy batch 随学生采样而变化，因此 loss 有明显噪声，不能单独用于选择 checkpoint。

### 3.5 多运行设计

三个运行使用 seed 42/43/44。当前脚本的 `--seed` 同时控制 256 条训练 prompt 的抽取、
dataloader 和训练/生成随机性，因此这里测量的是端到端 pipeline 波动，不是固定相同
数据下只改变优化随机性的纯 seed 实验。三个训练子集两两重合 7、19、9 条；该差异应
在解释标准差时保留。

## 4. Checkpoint 选择

先在固定 validation 位置 20–69 的 50 道题上做统一 greedy 评测：

| 模型 | 数值正确 | 严格正确 | 格式合规 | 达到上限 |
|---|---:|---:|---:|---:|
| SFT v7 | 44/50 | 44/50 | 49/50 | 0 |
| OPD step 10 | 44/50 | 44/50 | 49/50 | 0 |
| OPD step 20 | 45/50 | 45/50 | 50/50 | 0 |
| OPD step 30 | 45/50 | 45/50 | 50/50 | 0 |
| OPD step 40 | 43/50 | 43/50 | 49/50 | 1 |
| OPD step 50 | 45/50 | 44/50 | 49/50 | 1 |

step 20 和 step 30 进入完整 validation。step 40 已出现准确率下降，step 50 有一条
回答重复到 1,024 token，说明不能默认使用最后 checkpoint。

完整 validation 结果：

| 模型 | 数值/严格准确率 | 格式合规 | 达到上限 |
|---|---:|---:|---:|
| SFT v7 | 319/374（85.29%） | 372/374 | 2/374 |
| OPD seed 42 step 20 | 314/374（83.96%） | 372/374 | 1/374 |
| OPD seed 42 step 30 | 317/374（84.76%） | 372/374 | 1/374 |
| OPD seed 43 step 30 | 315/374（84.22%） | 372/374 | 1/374 |
| OPD seed 44 step 30 | 313/374（83.69%） | 371/374 | 2/374 |
| OPD 三次均值 ± SD | 84.22% ± 0.53 | 371.67/374（99.38% ± 0.15） | 1.33/374（0.36% ± 0.15） |

seed 42 的 SFT→step-30 有 14 道由对变错、12 道由错变对，`p=0.845019`。虽然
OPD 没有超过 SFT validation，但 step-30 是两个完整候选中更好的 checkpoint，因此
在不查看官方 test 的前提下被选为最终模型。固定 step-30 后，seed 43 的 SFT-only /
OPD-only 为 14/10（`p=0.541256`），seed 44 为 19/13（`p=0.377086`）。三个运行
都低于 SFT validation，平均差为 `-1.07 pp`。

## 5. 官方 Test 结果

### 5.1 总体指标

| 指标 | Raw Base | SFT v7 | OPD seed 42 | OPD seed 43 | OPD seed 44 | OPD 均值 ± SD |
|---|---:|---:|---:|---:|---:|---:|
| 数值准确率 | 71.80% | 71.65% | 72.33% | **73.39%** | 73.01% | **72.91% ± 0.54** |
| 严格准确率 | 0.00% | 71.65% | 72.25% | **73.31%** | 72.93% | **72.83% ± 0.54** |
| 格式遵循率 | 0.00% | **98.26%** | 97.73% | 97.80% | 97.50% | 97.68% ± 0.16 |
| 达到 1,024 token 上限 | 3.11% | **1.36%** | 1.90% | 1.82% | 2.12% | 1.95% ± 0.16 |
| 平均生成 token | 371.74 | **102.30** | 111.24 | 110.48 | 113.09 | 111.60 ± 1.34 |
| 中位生成 token | 339 | **83** | 87 | 86 | 87 | 86.67 ± 0.58 |
| 墙钟时间 | 179m 0s | 64m 4s | 75m 0s | 72m 8s | 77m 43s | 74m 57s ± 2m 48s |

三次 OPD 的数值正确数均比严格正确数多 1：各有一条回复的数值可解析为正确，但最终
没有形成合规的 `#### <answer>` 结尾。格式与截断退化也在三个运行中方向一致。

### 5.2 与 SFT v7 的逐题配对

| OPD 运行 | SFT-only | OPD-only | 净变化 | 准确率差 | McNemar p |
|---|---:|---:|---:|---:|---:|
| seed 42 | 49 | 58 | +9 | +0.68 pp | 0.439440 |
| seed 43 | 52 | 75 | +23 | +1.74 pp | 0.050487 |
| seed 44 | 50 | 68 | +18 | +1.36 pp | 0.117213 |

seed 43 最接近传统的 0.05 阈值，但 `p=0.050487` 仍不小于 0.05；三个单次配对检验
均不显著。三次运行共享同一个 test 集，不能把它们当作三个独立数据集来简单合并成
新的显著性检验。

### 5.3 与 Raw Base 的逐题配对

| OPD 运行 | Base-only | OPD-only | 净变化 | 准确率差 | McNemar p |
|---|---:|---:|---:|---:|---:|
| seed 42 | 181 | 188 | +7 | +0.53 pp | 0.754825 |
| seed 43 | 175 | 196 | +21 | +1.59 pp | 0.299100 |
| seed 44 | 182 | 198 | +16 | +1.21 pp | 0.441647 |

三次 test 点估计相对 SFT 和 Base 都为正，说明结果不再只是单一 seed 的偶然正号；
但 validation 平均为 `84.22% ± 0.53`，低于 SFT 的 85.29%，而且所有逐题检验均
未达到显著。因此证据增强为“test 上方向可复现”，仍不能升级为“能力提升已证实”。

## 6. 结论与经验

本次实验完成了从 SFT 学生、教师资格检查、on-policy rollout、软分布蒸馏、
checkpoint 选择到 test 与多运行鲁棒性评测的完整 OPD 闭环。

可以可靠陈述：

- 8 GiB GPU 可使用“BF16 LoRA 学生 + NF4 冻结教师”完成同规模 1.5B OPD；
- 纯 on-policy GKD 确实更新了学生 LoRA，50-step pilot 可稳定训练；
- 三次 OPD test 均高于 SFT，均值为 `72.91% ± 0.54 pp`，平均点估计比 SFT 高
  `1.26 pp`；
- 三次 validation 均低于 SFT，且所有单次 McNemar 检验均未达到 0.05；
- OPD 的格式、截断与生成长度退化在三个运行中稳定出现。

不能陈述：

- OPD 已被证明显著优于 SFT 或 Base；
- 更低的 GKD train loss 必然意味着更高的 GSM8K accuracy；
- checkpoint-50 因训练更久就一定优于 checkpoint-20/30。

官方 test 已经被初次报告和两次鲁棒性复现使用，不应再据此调整本实验超参数。若进行
新研究，应使用新 benchmark 或预先冻结新的 validation 方案，并将“数据子集 seed”与
“训练 seed”拆开；随后才能公平测试更多 rollout、不同 `beta` 及格式/长度正则项。

## 7. 复现入口

- 训练脚本：[`scripts/train_opd_gkd.py`](../scripts/train_opd_gkd.py)
- 教师结果：[`results/opd/teacher/`](../results/opd/teacher/)
- checkpoint 选择与 validation：[`results/opd/dev/`](../results/opd/dev/)
- 最终 test 与配对分析：[`results/opd/final/`](../results/opd/final/)

Pilot 训练：

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

seed 43/44 鲁棒性运行只改变 seed 和输出目录：

```bash
for seed in 43 44; do
  python scripts/train_opd_gkd.py \
    --output-dir "outputs/opd/qwen25_math_15b_gkd_seed${seed}" \
    --num-samples 256 \
    --max-steps 50 \
    --gradient-accumulation-steps 4 \
    --learning-rate 5e-6 \
    --max-new-tokens 256 \
    --lmbda 1.0 \
    --beta 0.5 \
    --save-steps 10 \
    --save-total-limit 5 \
    --seed "$seed" \
    --save-final-adapter
done
```

最终评测：

```bash
python scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-Math-1.5B \
  --adapter outputs/opd/qwen25_math_15b_gkd_pilot50/checkpoint-30 \
  --eval-split test \
  --max-new-tokens 1024 \
  --no-stop-after-answer-line \
  --output results/opd/final/test_gsm8k_gkd_pilot50_ckpt30_v3.json
```

三次结果与配对检验的机器可读汇总见
[`opd_ckpt30_multiseed_summary.json`](../results/opd/final/opd_ckpt30_multiseed_summary.json)。
