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
