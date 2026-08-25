# GRPO v4 探索性消融日志

## 低学习率策略漂移 pilot

本实验在 Confirmatory v2 和 GRPO v3 低温负结果之后开展，属于 post-hoc exploratory
pilot，不修改已有确认性结论。v3 表明降低 rollout temperature 虽能缩短训练采样，却不能
恢复最终 dev-select 表现。因此本轮不继续扫描 temperature，而是直接检验更新幅度。

唯一变化是把 GRPO learning rate 从 `5e-6` 降到 `1e-6`。初始 SFT 策略、标准
temperature=0.9、训练样本及顺序、三个随机种子、KL reference 与 beta、奖励函数、
completion 上限和 50-step / 200-rollout 预算全部保持不变。

预注册假设：较低学习率会限制前 50 steps 的策略漂移，从而保留更多 SFT 数值能力和格式
稳定性。训练完成后只评 final step-50；禁止通过中间 checkpoint 选择结果。

## 预先冻结的判断标准

只有同时满足以下条件，才把结果标为“值得独立复现”：

1. dev-select 数值正确数至少 306/374，即 81.82%；
2. 格式合规率至少 97%；
3. 1024-token 截断率不超过 3%。

这些门槛仅用于探索性筛选，即使全部满足也不能建立确认性改进结论。计划进行两项逐题
配对比较：低学习率 treatment 对 SFT step-0，以及 treatment 对原 `5e-6` step-50。

本轮仍只使用已经消费的 dev-select。禁止查看 dev-audit、GSM8K test 或 SVAMP，也禁止
根据结果继续在同一 dev-select 上扫描更多学习率。

冻结配置见 `configs/experimental/grpo_v4_lr1e6_pilot.json`。结果待本地运行后补充。

## 冻结训练命令

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft

time python scripts/train_grpo.py \
  --policy-initialization merged_sft \
  --merged-sft-model outputs/confirmatory_v2/merged_sft_seed42 \
  --dataset data/gsm8k_sft_clean.json \
  --split-manifest data/confirmatory_v2/split_manifest.json \
  --output-dir outputs/exploratory_v4/grpo_lr1e6_data42_train42_gen42_steps50 \
  --num-samples 1024 \
  --max-steps 50 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --learning-rate 1e-6 \
  --temperature 0.9 \
  --top-p 1.0 \
  --beta 0.04 \
  --accuracy-reward-weight 1.0 \
  --format-reward-weight 0.1 \
  --arithmetic-consistency-reward-weight 0.05 \
  --data-seed 42 \
  --training-seed 42 \
  --generation-seed 42 \
  --save-steps 50 \
  --save-total-limit 1 \
  --save-final-adapter
```

本轮不使用 `--resume-from-checkpoint`，且若目标输出目录已存在则先停止并核对，不覆盖旧结果。

## 训练完成，等待预注册评测

训练于 2026-08-25 正常完成 50 steps / 200 rollouts，耗时 499.69 秒，峰值显存
3.59 GiB，train loss 为 -0.05619。跟踪 LoRA 参数最大绝对变化为 `1.6529e-5`，参数确实
更新且全部训练指标有限。

50-step 均值为：rollout 截断率 46.00%，平均长度 182.06 tokens，KL `2.426e-4`，
entropy 0.3835，零组内奖励方差比例为 0%。降低学习率明显缩小了参数更新幅度，但没有改善
训练 rollout 截断；最终是否保留 SFT 能力仍必须由预注册的 step-50 dev-select 评测决定。

run manifest、summary 与 adapter 权重的 SHA-256 已写入冻结配置。此时尚未运行或查看
dev-select 结果。

## 固定 dev-select 评测结果

唯一一次 final step-50 评测耗时 2127.72 秒。协议与 Confirmatory v2 保持一致：固定
374 题、贪心解码、1024-token 上限，且不在完整 `####` 行后人为截停。

| 模型 | 数值准确率 | 严格准确率 | 格式合规 | 截断率 | 平均 tokens |
|---|---:|---:|---:|---:|---:|
| SFT seed42 step-0 | 83.69% | 83.69% | 99.73% | 0.80% | 96.83 |
| GRPO lr=5e-6 step-50 | 78.34% | 77.01% | 94.12% | 6.15% | 153.31 |
| GRPO lr=1e-6 step-50 | 78.61% | 76.47% | 92.78% | 7.75% | 166.98 |

低学习率 treatment 只得到 294/374，低于预注册的 306/374；格式 92.78% 低于 97%；
截断 7.75% 高于 3%。三个条件全部失败，因此结果不值得进入独立复现阶段。

与 SFT step-0 配对比较时，32 题由对变错、13 题由错变对，准确率差为 -5.08 pp，
exact McNemar `p=0.00661`。低学习率 GRPO 相对初始 SFT 的退化仍有统计证据。

与原 learning rate `5e-6` step-50 配对比较时，3 题由对变错、4 题由错变对，准确率仅
增加 0.27 pp，exact McNemar `p=1.0`。因此没有证据表明降低学习率改善了任务表现。

## 与 v3 汇总数字相同的核查

v4 与 v3 低温 pilot 恰好同为 294/374、严格 286/374、格式 347/374、截断 29/374。
这不是重复使用 adapter 或结果文件：两份 adapter 和评测 JSON 的 SHA-256 均不同。

逐题核查显示，374 题中有 12 题正确性不同、20 个规范化预测不同、109 条完整响应不同；
两者分别有 6 题对→错和 6 题错→对，McNemar `p=1.0`。因此相同汇总数字只是净变化抵消。

## 结论

本轮得到负结果。把学习率降低五倍确实显著缩小了跟踪 LoRA 参数的更新幅度，但没有保留
SFT 数值能力，也没有改善格式或长输出问题。这削弱了“早期退化主要由更新步幅过大造成”
这一解释。

结合 v3，本项目已经排除了两个简单修补方向：单独降低 rollout temperature，以及单独
降低 learning rate。继续在同一 dev-select 上扫描超参数的科学价值很低；下一步应停止
GRPO 点状调参，转向 reward 可辨识度/截断掩码机制的代码级诊断，或在新的预先冻结数据上
开展结构性实验。本轮没有查看 dev-audit、GSM8K test 或 SVAMP。
