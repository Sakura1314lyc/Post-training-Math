# Qwen2.5-Math-1.5B 在 GSM8K 上的 OPD/GKD 实验报告

## 1. 实验摘要

本实验从最终 SFT v7 adapter 出发，使用 `Qwen/Qwen2.5-Math-1.5B-Instruct`
作为冻结教师，通过 TRL `GKDTrainer` 实现 On-Policy Distillation（OPD）。学生在
训练时现场采样回答，教师在这些学生轨迹上提供 token-level 软分布，学生最小化广义
Jensen-Shannon divergence。设置 `lmbda=1.0`，保证每个训练 batch 都来自学生的
on-policy rollout。

最终选择 checkpoint-30。在官方 GSM8K test 上：

- Raw Base：`947/1319 = 71.80%`；
- SFT v7：`945/1319 = 71.65%`；
- OPD checkpoint-30：`954/1319 = 72.33%`。

OPD 相比 SFT 净提升 9 题（`+0.68` 个百分点），但精确双侧 McNemar
`p=0.439440`；相比 Base 净提升 7 题（`+0.53` 个百分点），`p=0.754825`。
因此结果应表述为：**OPD 获得了当前最佳的 test 点估计，但没有统计证据证明它带来
稳定、显著的准确率提升。**

OPD 还带来轻微行为代价：相比 SFT，格式遵循率由 `98.26%` 降至 `97.73%`，达到
1,024-token 上限的样本由 18 增至 25，平均生成长度由 `102.30` 增至 `111.24`
token。

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
- pilot 从其中固定随机抽取 256 条；
- 官方 test 只在 validation 选择完 checkpoint 后运行一次。

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
| OPD step 20 | 314/374（83.96%） | 372/374 | 1/374 |
| OPD step 30 | 317/374（84.76%） | 372/374 | 1/374 |

SFT→step-30 有 14 道由对变错、12 道由错变对，`p=0.845019`。虽然 OPD 没有
超过 SFT validation，但 step-30 是两个完整候选中更好的 checkpoint，因此在不查看
官方 test 的前提下被选为最终模型。

## 5. 官方 Test 结果

### 5.1 总体指标

| 指标 | Raw Base | SFT v7 | OPD step 30 |
|---|---:|---:|---:|
| 数值准确率 | 947/1319（71.80%） | 945/1319（71.65%） | **954/1319（72.33%）** |
| 严格准确率 | 0/1319（0.00%） | 945/1319（71.65%） | **953/1319（72.25%）** |
| 格式遵循率 | 0/1319（0.00%） | **1296/1319（98.26%）** | 1289/1319（97.73%） |
| 达到 1,024 token 上限 | 41/1319（3.11%） | **18/1319（1.36%）** | 25/1319（1.90%） |
| 平均生成 token | 371.74 | **102.30** | 111.24 |
| 中位生成 token | 339 | **83** | 87 |
| 墙钟时间 | 179m 0s | 64m 4s | 75m 0s |

OPD 的 25 条截断回复全部答错。另有 1 条回复数值正确但只输出了独立 `####`，随后
在下一行写答案，因此数值正确数为 954，严格正确数为 953。

### 5.2 逐题配对

SFT v7 与 OPD：

| 转移 | 数量 |
|---|---:|
| 两者都对 | 896 |
| SFT 对、OPD 错 | 49 |
| SFT 错、OPD 对 | 58 |
| 两者都错 | 316 |

净改善 9 题，准确率差 `+0.68 pp`，McNemar `p=0.439440`。

Raw Base 与 OPD：共同答对 766、Base-only 181、OPD-only 188、共同答错 184；
净改善 7 题，差 `+0.53 pp`，McNemar `p=0.754825`。

两项检验均不显著。当前证据只支持“OPD 点估计最高”，不支持“OPD 稳定提高了
数学推理能力”。validation 的 `-0.53 pp` 与 test 的 `+0.68 pp` 方向相反，也说明
真实效应很小，容易被样本差异和训练随机性淹没。

## 6. 结论与经验

本次实验完成了从 SFT 学生、教师资格检查、on-policy rollout、软分布蒸馏、
checkpoint 选择到独立 test 的完整 OPD 闭环。

可以可靠陈述：

- 8 GiB GPU 可使用“BF16 LoRA 学生 + NF4 冻结教师”完成同规模 1.5B OPD；
- 纯 on-policy GKD 确实更新了学生 LoRA，50-step pilot 可稳定训练；
- OPD 在 test 上获得当前最高点估计 72.33%，相比 SFT 净改善 9 题；
- 提升没有统计显著性，并伴随轻微格式、终止和长度退化。

不能陈述：

- OPD 已被证明显著优于 SFT 或 Base；
- 更低的 GKD train loss 必然意味着更高的 GSM8K accuracy；
- checkpoint-50 因训练更久就一定优于 checkpoint-20/30。

官方 test 已经用于最终报告，不应再据此调整本实验超参数。若进行新研究，应预先注册
新的 validation 方案或使用新 benchmark，并测试多 seed、更多 rollout、不同 `beta`
以及格式/长度正则项。

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
