# Confirmatory v2 protocol

This directory defines a new experiment family. Existing GSM8K and SVAMP
results remain exploratory evidence and must not be used to tune this protocol.

Protocol rules:

1. Run `scripts/prepare_confirmatory_splits.py` once. Commit its manifest, not
   the generated JSON copies.
2. `dev_select` may be used for checkpoint/model selection. `dev_audit` must
   remain unopened until the SFT/OPD/GRPO settings and selection rule are frozen.
3. Train SFT with seeds 42, 43, and 44. Report mean, standard deviation, and all
   individual runs.
4. For controlled OPD/GRPO variance, hold `data_seed=42` fixed and vary only
   `training_seed` and `generation_seed` across 42/43/44. A separate data-subset
   ablation may vary `data_seed` while holding the other two fixed.
5. Use larger post-training budgets than the exploratory runs: OPD at least 200
   optimizer steps / 800 on-policy rollouts; GRPO at least 200 optimizer steps /
   800 rollouts. If hardware forces a smaller budget, label it a pilot.
   The confirmatory result is the final step-200 adapter; intermediate step-50,
   step-100, and step-150 checkpoints are recovery artifacts and must not be
   selected by `dev_select` performance.
6. GRPO with `beta>0` must use `policy-initialization=merged_sft`. The old
   continued-adapter path is retained only for reproduction with `beta=0`.
7. After settings are frozen, evaluate `dev_audit` once and archive the command,
   split manifest, run manifests, and environment versions.

The audit split reduces validation-selection bias for the new v2 experiments;
it does not retroactively remove bias from the already completed v7 study.

## Commands

Freeze the split first (CPU-only):

```bash
python scripts/prepare_confirmatory_splits.py
```

Train the three SFT seeds independently. Each config performs one fixed epoch
and evaluates `dev_select` only once at the end:

```bash
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed42.yaml
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed43.yaml
llamafactory-cli train configs/confirmatory/qwen25_math_15b_base_lora_sft_v8_seed44.yaml
```

Example controlled OPD run after choosing the SFT checkpoint by a predeclared
rule (replace `<SFT_ADAPTER>` and seed 42 with 43/44 for repetitions):

```bash
python scripts/train_opd_gkd.py \
  --adapter <SFT_ADAPTER> \
  --dataset data/gsm8k_sft_clean.json \
  --split-manifest data/confirmatory_v2/split_manifest.json \
  --output-dir outputs/confirmatory_v2/opd_train42_gen42 \
  --num-samples 1024 \
  --max-steps 200 \
  --gradient-accumulation-steps 4 \
  --learning-rate 5e-6 \
  --max-new-tokens 256 \
  --lmbda 1.0 \
  --beta 0.5 \
  --data-seed 42 \
  --training-seed 42 \
  --generation-seed 42 \
  --save-steps 50 \
  --save-final-adapter
```

For GRPO with a valid SFT KL reference, merge the selected SFT policy once:

```bash
python scripts/merge_sft_adapter.py \
  --adapter <SFT_ADAPTER> \
  --output-dir outputs/confirmatory_v2/merged_sft_seed42
```

Then run fresh-LoRA GRPO (repeat training/generation seeds 42/43/44 while
keeping `data-seed=42`):

```bash
python scripts/train_grpo.py \
  --policy-initialization merged_sft \
  --merged-sft-model outputs/confirmatory_v2/merged_sft_seed42 \
  --dataset data/gsm8k_sft_clean.json \
  --split-manifest data/confirmatory_v2/split_manifest.json \
  --output-dir outputs/confirmatory_v2/grpo_train42_gen42 \
  --num-samples 1024 \
  --max-steps 200 \
  --num-generations 4 \
  --gradient-accumulation-steps 4 \
  --max-prompt-length 512 \
  --max-completion-length 256 \
  --learning-rate 5e-6 \
  --beta 0.04 \
  --accuracy-reward-weight 1.0 \
  --format-reward-weight 0.1 \
  --arithmetic-consistency-reward-weight 0.05 \
  --data-seed 42 \
  --training-seed 42 \
  --generation-seed 42 \
  --save-steps 50 \
  --save-final-adapter
```

Do not run the following audit command until every setting and checkpoint rule
is frozen. For a fresh GRPO adapter, use its merged SFT directory as
`--base-model`; for SFT/OPD use the original base model and the corresponding
adapter.

```bash
python scripts/eval_sft_adapter.py \
  --base-model <REFERENCE_BASE_OR_MERGED_SFT> \
  --adapter <FINAL_ADAPTER> \
  --benchmark gsm8k \
  --local-dataset-file data/confirmatory_v2/dev_audit.json \
  --local-split-role dev_audit \
  --output results/confirmatory_v2/dev_audit_final.json \
  --max-new-tokens 1024
```
