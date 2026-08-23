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

## 当前进度（2026-08-23）

- 已冻结 6725/374/374 的 train/dev-select/dev-audit 切分，三者互斥且覆盖全部
  7473 条数据；audit 仍未评测。
- SFT seed 42/43/44 均完成 1 epoch、841 steps。dev-select 数值准确率分别为
  83.69%、82.35%、85.29%，均值为 **83.78% ± 1.47 pp**（样本 SD）。
- 后续固定使用规范 seed 42 作为共同 SFT 起点，不选择 dev-select 最好的 seed 44。
- OPD data42/train42/gen42 已完成 200 steps / 800 on-policy microbatches；训练耗时
  5925 秒，峰值显存 6.86 GiB，最终参数已更新且指标有限。其 dev-select 评测和另外
  两个训练 seed 留待后续完成。

精确指标、路径及 SHA-256 见
`results/confirmatory_v2/confirmatory_v2_progress.json`。
