# Confirmatory v2 实验修正方案

## 目的

本方案不改写已经完成的 v7、OPD pilot 和 GRPO pilot 结论，而是建立一组新的
confirmatory v2 实验，用于修正原实验在随机种子、验证集使用、训练预算和 GRPO
参考策略方面的限制。旧结果仍应标记为探索性结果。

## 已落实的修正

### 1. SFT 多随机种子与验证选择偏差

- 新增 SFT seed 42、43、44 三份独立配置。
- 从清洗后的 7473 条训练数据中冻结 `train`、`dev_select`、`dev_audit` 三个互斥分区。
- `dev_select` 复现旧 v7 的 374 条 validation，便于与旧结果衔接。
- `dev_audit` 是另外冻结的 374 条样本，新实验训练时强制排除。
- SFT 训练固定为一个 epoch，只在 epoch 末评测一次 `dev_select`，不再根据反复查看
  validation 曲线修改同一实验。
- 所有超参数和选择规则冻结后，`dev_audit` 只运行一次。

这能降低新实验的开发集选择偏差，但不能追溯消除旧 v7 已经产生的选择偏差。

### 2. 扩大 OPD/GRPO 预算

- 新协议将 OPD 从 50 steps / 约 200 rollouts 提高到至少 200 steps / 约 800
  on-policy rollouts。
- 新协议将 GRPO 从 30 steps / 约 120 rollouts 提高到至少 200 steps / 约 800
  rollouts。
- 正式结果固定使用最终 step-200 adapter；中间 checkpoint 只用于断点恢复，不再根据
  `dev_select` 选择表现最好的训练步数。
- 若 8 GiB 显存或运行时间迫使预算降低，结果必须继续标记为 pilot，不能写成充分训练
  后的最终结论。

### 3. 拆分随机性来源

OPD 和 GRPO 现在分别记录并控制：

- `data_seed`：训练分区内抽取哪一组样本；
- `training_seed`：LoRA 初始化、训练采样顺序和优化相关随机性；
- `generation_seed`：on-policy rollout 生成随机性。

主方差实验固定 `data_seed=42`，只改变 training/generation seed 42、43、44。另做数据子集
方差时，固定 training/generation seed，再单独改变 `data_seed`。这样报告的标准差不会再把
“换了训练题目”和“同一批题目的训练随机性”混为一个量。

### 4. GRPO 的 KL reference

旧实现直接继续训练 SFT LoRA。TRL 对 PEFT 模型计算 reference 时会关闭当前 adapter，因而
参考策略实际是 Raw Base，而不是 SFT；因此旧实现只能安全使用 `beta=0`。

新实现先将 SFT LoRA 合并进基础模型，再添加一层全新的 GRPO LoRA。TRL 关闭新 LoRA 后，
得到的正是冻结的 SFT 初始策略，因此可以合法使用非零 `beta`。运行清单会显式保存
`kl_reference=merged_sft_policy`，避免误报。

### 5. 推理过程奖励

在最终答案数值奖励和 `####` 格式奖励之外，GRPO 可选加入“显式数值等式算术一致性”奖励。
它会安全解析回答中的纯数值加减乘除等式，仅当检测到的等式全部成立时给奖励。

这一项只能减少明显的算术错误，不能证明文字推理忠实，也不能排除 reward hacking。若要把
“推理忠实性”本身作为强结论，仍需人工标注、程序化证明检查器或独立过程监督数据；当前
报告只能称其为 arithmetic-consistency proxy。

## 报告要求

- SFT、OPD、GRPO 均列出每个 seed 的原始结果，不只报告最好结果。
- 同时报告均值、样本标准差、格式合规率、截断率和运行预算。
- 主方差与数据子集方差分开报告。
- `dev_select` 用于选择，`dev_audit` 用于一次性确认；两者不得混写。
- 已经使用过的 GSM8K test 和 SVAMP 结果只用于旧方案的最终比较，不参与 v2 调参。

完整命令见 `configs/confirmatory/README.md`。

## 最终进度（2026-08-25）

- 已冻结 6725/374/374 的 train/dev-select/dev-audit 切分，三者互斥且覆盖全部
  7473 条数据；audit 在所有选择规则提交冻结后仅评测一次。
- SFT seed 42/43/44 均完成 1 epoch、841 steps。dev-select 数值准确率分别为
  83.69%、82.35%、85.29%，均值为 **83.78% ± 1.47 pp**（样本 SD）。
- 后续固定使用规范 seed 42 作为共同 SFT 起点，不选择 dev-select 最好的 seed 44。
- OPD 固定 data seed 42、分别使用 train/generation seed 42/43/44，三组均已完成
  200 steps / 800 on-policy microbatches。训练 loss 均值为 0.2847，样本 SD 为
  0.0023；单次耗时约 98.6--119.1 分钟。
- 三组 OPD 的 dev-select 数值准确率分别为 81.28%、83.69%、83.42%，均值为
  **82.80% ± 1.32 pp**。相对固定 SFT seed 42（83.69%），没有形成稳定提升；三组
  McNemar 双侧精确检验 p 值分别为 0.281、1.000、1.000。
- 更重要的是，这套 200-step OPD 配置出现明显输出行为退化：严格 `####` 准确率均值仅
  **1.25%**，格式合规率均值仅 **2.58%**，平均 **30.93%** 的回答达到 1024-token
  上限，平均生成长度由 SFT 的 96 tokens 左右升至 428 tokens。因而该 OPD 配置不能
  作为可部署改进，也没有理由进入封存的 audit；这一负结果将如实保留。
- 下一阶段按冻结协议运行带有效 SFT KL reference 的 GRPO。`dev_audit` 继续保持未评测。
- 规范 SFT seed 42 已通过 `merge_and_unload(safe_merge=True)` 合并为独立 BF16 policy；
  合并目录不含残留 adapter，后续在其上新建 GRPO LoRA。这样关闭新 LoRA 时得到的是固定
  SFT policy，可作为 `beta>0` 的有效 KL reference。
- GRPO 二次 smoke 已确认：两步奖励标准差均非零，第二步 KL 为 0.000162，LoRA 参数发生
  更新，指标有限，峰值显存 3.59 GiB。随机 rollout 的 256-token 截断比例仍偏高，正式
  训练继续启用 truncated-completion masking，并将截断比例作为必须报告的诊断指标。
- 正式 GRPO data42/train42/gen42 已完成 200 steps / 800 rollouts，耗时 1898 秒，峰值
  显存 3.60 GiB，参数变化 0.000607。平均 reward 为 0.495、平均 KL 为 0.000274，全部
  step 的奖励方差均非零；平均截断率为 45.63%，已由 masking 排除截断 completion 的
  token loss。四个 50-step 阶段未呈现持续发散，但最终判断仍以固定 dev-select 评测为准。
- 该 seed42 最终 adapter 在 dev-select 上取得 296/374（79.14%），低于共同 SFT 起点的
  313/374（83.69%）4.55 pp。逐题转移为 28 个 SFT-correct→GRPO-wrong、11 个
  SFT-wrong→GRPO-correct，McNemar 双侧精确检验 `p=0.00948`，构成显著负向结果。
  格式合规率仍有 95.72%，但评测截断率升至 4.55%。协议要求继续完成 seed43/44，不依据
  单个 seed 修改超参数或选择结果；audit 仍保持封存。
- 正式 GRPO data42/train43/gen43 也已完成 200 steps / 800 rollouts：训练耗时
  1864 秒，train loss 为 -0.0825，平均 reward 0.540、平均 KL 0.000288，参数变化
  0.000472，峰值显存 3.60 GiB；平均 rollout 截断率为 42.13%。其固定 dev-select
  评测为 292/374（78.07%），相对共同 SFT 起点下降 5.61 pp。逐题转移为 32 个
  SFT-correct→GRPO-wrong、11 个反向改善，McNemar `p=0.00191`；评测格式合规率
  95.19%，截断率 5.35%。前两个训练 seed 均为显著负向结果，但仍按冻结协议完成 seed44。
- 正式 GRPO data42/train44/gen44 已完成 200 steps / 800 rollouts：训练耗时
  1871 秒，train loss -0.0845，平均 reward 0.505、平均 KL 0.000346，参数变化
  0.000547，峰值显存 3.60 GiB，平均 rollout 截断率 45.13%。至此三个正式训练 seed
  全部完成；训练耗时均值为 1878 ± 18 秒，平均 KL 为 0.000303，rollout 截断率均值为
  44.29%。seed44 的 dev-select 为 294/374（78.61%），相对共同 SFT 起点下降
  5.08 pp；配对转移为 30 个退化、11 个改善，McNemar `p=0.00432`。

### Dev-select 阶段结论

| 方法 | 数值准确率 | 严格准确率 | 格式合规率 | 达到 1024-token 上限 | 平均生成 tokens |
|---|---:|---:|---:|---:|---:|
| 规范 SFT seed42 | 83.69% | 83.69% | 99.73% | 0.80% | 96.83 |
| OPD 三 seed 均值 ± SD | 82.80% ± 1.32 | 1.25% ± 0.15 | 2.58% ± 0.15 | 30.93% ± 2.48 | 427.97 ± 22.38 |
| GRPO 三 seed 均值 ± SD | 78.61% ± 0.53 | 77.90% ± 0.67 | 95.72% ± 0.53 | 4.63% ± 0.67 | 138.49 ± 5.97 |

确认性结果不支持将 OPD 或 GRPO 作为 SFT 的改进：OPD 的宽松数值准确率接近 SFT，
但输出格式和终止行为崩坏；GRPO 保留了大部分格式能力，却在三个 seed 上分别下降
4.55、5.61、5.08 pp，且三次 McNemar 检验均显著。两种方法均在进入 audit 前被拒绝，
不能选择其中“最好的一次”规避负结果。

### 一次性 Dev-audit

预先指定的规范 SFT seed42 在 `dev_audit` 上得到 303/374（**81.02%**），严格准确率
同为 81.02%，格式合规率 99.47%，截断率 0.53%，平均生成 98.89 tokens。相比它在
`dev_select` 上的 83.69%，audit 低 2.67 pp。由于两个分区互斥，不能使用 McNemar；
独立两比例双侧检验得到 `z=-0.959`、`p=0.3375`，差异不显著。audit 准确率的 95%
Wilson 区间为 76.73%--84.67%。

因此最终报告以 **81.02%** 作为未参与调参的规范 SFT 点估计，同时如实说明其低于
dev-select，但现有样本不足以确认显著选择偏差。audit 至此已消费完毕，不再用于调参；
此前被拒绝的 OPD/GRPO 没有查看 audit。Confirmatory v2 协议正式结束。

协议结束后的 checkpoint 轨迹分析单独记录于
[`CONFIRMATORY_V2_GRPO_TRAJECTORY_zh.md`](CONFIRMATORY_V2_GRPO_TRAJECTORY_zh.md)，其结论
仅用于诊断，不属于确认性选型结果。

精确指标、路径及 SHA-256 见
`results/confirmatory_v2/confirmatory_v2_progress.json`。
