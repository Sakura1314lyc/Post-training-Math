# GSM8K Post-training Experiment

This project compares Qwen2.5-Instruct before and after LoRA-SFT on GSM8K. The
current models are **Instruct checkpoints**, so `base` in result names means the
pre-SFT experimental baseline, not a raw pretrained base model.

## Canonical workflow

Run commands from this directory:

```bash
cd /home/sakura/projects/llm/post-training-math
```

### 1. Select a checkpoint on the held-out validation split

The training configs use `val_size: 0.05` and `seed: 42`. `train_validation`
reproduces the same deterministic GSM8K train split used by LLaMA-Factory.

Evaluate the pre-SFT baseline:

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --eval-split train_validation \
  --output results/dev_base_15b.json
```

Evaluate a checkpoint (repeat for checkpoint-100 through checkpoint-888):

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/qwen25_15b_lora_sft_v1/checkpoint-100 \
  --eval-split train_validation \
  --output results/dev_sft_15b_ckpt100.json
```

Use validation exact-match accuracy—not token-level eval loss—to select the
checkpoint. Do not use GSM8K test results for checkpoint selection.

### 2. Run the final full-test comparison once

Omitting `--num-samples` evaluates all 1,319 GSM8K test examples.

```bash
python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --eval-split test \
  --output results/final_base_15b_test.json

python3 scripts/eval_sft_adapter.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --adapter outputs/qwen25_15b_lora_sft_v1/checkpoint-BEST \
  --eval-split test \
  --output results/final_sft_15b_test.json
```

For a quick smoke evaluation, explicitly pass `--num-samples 20`. The script
uses greedy decoding and refuses to replace existing output unless
`--overwrite` is provided.

### 3. Produce the paired report

```bash
python3 scripts/compare_base_sft.py \
  --base results/final_base_15b_test.json \
  --sft results/final_sft_15b_test.json \
  --output results/final_base_sft_analysis.json
```

The report includes relaxed numeric-answer accuracy, strict `####` accuracy,
paired transitions, output-format compliance, response-length statistics, and
the exact two-sided McNemar p-value. The relaxed metric measures whether the
conclusion is numerically correct even when the model fails the requested
format; the strict metric measures both correctness and format following.

## Scripts

- `evaluation_utils.py`: the single source of truth for GSM8K extraction,
  numeric normalization, and scoring.
- `eval_sft_adapter.py`: evaluates either the baseline or a PEFT adapter.
- `rescore.py`: migrates legacy JSON results to the canonical evaluator without
  overwriting the originals.
- `compare_base_sft.py`: performs a strict paired comparison. Both files must
  contain exactly the same questions.
- `prepare_formal_sft_data.py`: creates the formal Alpaca-format SFT dataset.
- `prepare_sft_data.py`: creates the historical smoke/full datasets used for
  early pipeline checks.

Run the evaluator unit tests with:

```bash
python3 -m unittest discover -s tests -v
```

## Current experimental observation

On the repeatedly inspected first 100 test examples, the 1.5B baseline scored
73% relaxed accuracy, while the final 1.5B SFT adapter scored 49%. In contrast,
strict `####` accuracy improved from 21% to 49%, and parseable
`#### <number>` compliance improved from 25% to 100%. This separates the main
finding: SFT improved format following but degraded relaxed numerical reasoning
on this exploratory subset. Treat these 100 examples as exploratory evidence,
not the final held-out result.
