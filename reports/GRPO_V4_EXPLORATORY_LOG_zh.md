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
