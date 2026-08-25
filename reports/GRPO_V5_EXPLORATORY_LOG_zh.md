# GRPO v5 探索性结构消融日志

## 移除 tied `lm_head` target

本实验属于 Confirmatory v2 完成后的 post-hoc exploratory pilot。此前降低 rollout
temperature 和降低 learning rate 均未修复 GRPO 的早期能力退化。结构检查进一步发现，
现有 `q_proj,v_proj,lm_head` adapter 会因 tied embeddings 保存完整的 2.33 亿参数
`lm_head.base_layer.weight`，最终文件约 454 MiB，并持续触发 PEFT tied-target 警告。

本轮唯一结构变化是将 fresh GRPO LoRA targets 改为 `q_proj,v_proj`。标准
temperature=0.9、learning rate `5e-6`、初始 SFT、训练样本与顺序、三个随机种子、KL、
奖励和 50-step 预算保持不变。

完整结构证据见 `GRPO_ADAPTER_STRUCTURE_DIAGNOSTIC_zh.md`，冻结机器协议见
`configs/experimental/grpo_v5_qv_only_pilot.json`。

## Smoke gate

正式训练前必须先运行 1-step / 4-rollout smoke，并同时满足：

1. 全量漂移统计包含 112 个 LoRA tensor、1,089,536 个参数；
2. target 分组严格为 `q_proj,v_proj`，至少一个 tensor 发生更新且所有指标有限；
3. 保存的 safetensors 不含 `lm_head` 或 `base_layer`；
4. adapter 权重文件不超过 10 MiB；
5. 不再出现 tied `lm_head` 的 PEFT warning。

任一条件失败都停止，不进入正式训练。

### 冻结 smoke 命令

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft

time python scripts/train_grpo.py \
  --policy-initialization merged_sft \
  --merged-sft-model outputs/confirmatory_v2/merged_sft_seed42 \
  --lora-target-modules q_proj,v_proj \
  --dataset data/gsm8k_sft_clean.json \
  --split-manifest data/confirmatory_v2/split_manifest.json \
  --output-dir outputs/exploratory_v5/grpo_qv_only_smoke \
  --num-samples 8 \
  --max-steps 1 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --learning-rate 5e-6 \
  --temperature 0.9 \
  --top-p 1.0 \
  --beta 0.04 \
  --accuracy-reward-weight 1.0 \
  --format-reward-weight 0.1 \
  --arithmetic-consistency-reward-weight 0.05 \
  --data-seed 42 \
  --training-seed 42 \
  --generation-seed 42 \
  --save-final-adapter
```

如果 smoke 输出目录已经存在，停止并核对，不覆盖或续跑。正式 50-step 命令只有在 smoke
gate 经脚本与权重检查全部通过后才提供。

## 正式结果边界

正式训练完成后只评 final step-50 一次。值得在新数据上复现的门槛仍为：数值正确数至少
306/374、格式合规率至少 97%、截断率不超过 3%，且三个条件必须全部满足。

这是最后一次允许复用 dev-select 的 GRPO pilot。禁止查看中间 checkpoint，禁止使用
dev-audit、GSM8K test 或 SVAMP，也禁止根据结果继续扫描同一 dev-select 上的 target、
temperature 或 learning rate。

## Smoke 结果

Smoke 于 2026-08-25 完成，训练耗时 12.52 秒，峰值显存 3.56 GiB，指标有限。
全量漂移统计与预期严格一致：112 个 LoRA tensor、1,089,536 个参数，其中 56 个
`lora_B` tensor 全部更新，56 个初始非零的 `lora_A` 在第一步尚未更新。全局最大绝对变化
为 `5.00e-6`，L2 delta 为 `0.003147`。

旧的单参数指标因恰好跟踪第一个 `lora_A` 而显示“未更新”，但全量统计证明训练确实发生。
这验证了新增诊断的必要性。

保存后的 adapter 权重文件为 4,372,840 bytes（约 4.17 MiB），恰好包含 112 个 tensor、
1,089,536 个参数；没有任何键包含 `lm_head` 或 `base_layer`。adapter config 的 target
严格为 `q_proj,v_proj`，因此 tied `lm_head` warning 的触发路径已被移除。

所有 smoke gate 条件通过，可以进入冻结的正式 50-step 训练。此时尚未运行正式训练或查看
任何新的 dev-select 结果。

## 冻结正式训练命令

```bash
cd /home/sakura/projects/llm/post-training-math
conda activate sft

time python scripts/train_grpo.py \
  --policy-initialization merged_sft \
  --merged-sft-model outputs/confirmatory_v2/merged_sft_seed42 \
  --lora-target-modules q_proj,v_proj \
  --dataset data/gsm8k_sft_clean.json \
  --split-manifest data/confirmatory_v2/split_manifest.json \
  --output-dir outputs/exploratory_v5/grpo_qv_only_data42_train42_gen42_steps50 \
  --num-samples 1024 \
  --max-steps 50 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --learning-rate 5e-6 \
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

如果正式输出目录已经存在，停止并核对，不覆盖或续跑。训练结束后先检查全量漂移和 adapter
结构，再决定是否执行协议中唯一一次 final step-50 dev-select 评测。

## 正式训练结果

正式训练于 2026-08-25 正常完成 50 steps / 200 rollouts，耗时 643.48 秒，峰值显存
3.57 GiB，train loss 为 -0.05353，所有指标有限。

全量漂移显示 112/112 个 LoRA tensor 全部更新：全局最大绝对变化 `1.8036e-4`，L2 delta
0.04891，相对训练前 LoRA 范数约 0.400%。`lora_A` 和 `lora_B` 的 L2 delta 分别为
0.04000 和 0.02816。相比旧的单 A 张量，这些数值完整描述了 adapter 更新。

50-step rollout 均值为：截断率 43.50%，平均长度 177.15 tokens，KL `2.569e-4`，
entropy 0.3742，零组内奖励方差比例为 0%。这些训练期诊断不能代替固定评测。

最终 adapter 仍为 4,372,840 bytes，包含 112 个 tensor 和 1,089,536 个参数，不含任何
`lm_head` 或 `base_layer`。run manifest、summary 和 adapter 权重哈希已写入冻结配置。
训练完成后只按冻结协议评测 final step-50，未查看任何中间 checkpoint。

## Final step-50 评测

唯一一次正式 `dev_select` 评测得到：

| 指标 | SFT step-0 | 旧 GRPO（q/v/lm_head） | v5（q/v only） |
|---|---:|---:|---:|
| 数值正确率 | 313/374（83.69%） | 293/374（78.34%） | 294/374（78.61%） |
| 严格正确率 | 313/374（83.69%） | 288/374（77.01%） | 287/374（76.74%） |
| 格式合规率 | 373/374（99.73%） | 352/374（94.12%） | 352/374（94.12%） |
| 达到长度上限 | 3/374（0.80%） | 23/374（6.15%） | 24/374（6.42%） |

v5 平均生成 155.37 tokens，中位数 90.5；终止原因是 EOS 350 题、达到
`max_new_tokens` 24 题。结果文件 SHA-256 为
`baff87476b9d16e49e839bfa82f8eb9a3a72c7a4b55aa525b8faac59c11c10d6`。

预声明的三个 promising 条件全部失败：正确数 294 小于 306，格式合规率 94.12% 小于
97%，长度上限命中率 6.42% 高于 3%。因此本实验不进入新数据复现阶段。

## 配对比较

相对 SFT step-0，v5 有 31 题由对变错、12 题由错变对，净下降 19 题，准确率差
-5.08 个百分点；精确双侧 McNemar `p=0.00540157`。这说明退化不仅是样本波动，在该冻结
题集上具有统计证据。

相对旧的 `q_proj,v_proj,lm_head` GRPO，v5 有 8 题由对变错、9 题由错变对，净增加
1 题，准确率差 +0.27 个百分点，McNemar `p=1`。没有证据表明移除 `lm_head` 改善了
数值能力。

## 结论与停止规则

本轮得到一个清晰的“工程成功、能力失败”结论：移除 tied `lm_head` 将 adapter 从约
454 MiB 降至 4.17 MiB，消除了完整 `lm_head.base_layer.weight` 保存路径和对应 warning，
结构修复是有效的；但准确率仍与旧 GRPO 基本相同，并显著低于 SFT，所以该保存结构不是
GRPO 早期能力退化的主要原因。

按照预先冻结的 decision boundary，到此停止在已复用 `dev_select` 上继续扫描 target、
temperature、learning rate 或中间 checkpoint。`dev_audit`、GSM8K test 和 SVAMP 继续保持
禁止用于这条探索线的调参。后续应转向总结现有负结果，或先冻结全新的开发数据与独立协议，
而不是依据本结果再启动同一开发集上的 GRPO pilot。
