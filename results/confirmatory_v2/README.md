# Confirmatory v2 artifacts

This directory contains small, tracked evaluation artifacts for the new
confirmatory protocol. Model checkpoints and generated split copies remain under
ignored `outputs/` and `data/confirmatory_v2/*.json` paths.

Final status on 2026-08-25:

- frozen split: 6,725 train / 374 dev-select / 374 dev-audit;
- SFT seeds 42/43/44: one-epoch training and dev-select evaluation complete;
- canonical downstream SFT policy: seed 42, fixed before downstream evaluation;
- OPD data42 with train/generation seeds 42/43/44: 200-step training and dev-select
  evaluation complete; rejected for severe output-format and termination regression;
- GRPO data42 with train/generation seeds 42/43/44: 200-step training and dev-select
  evaluation complete; all three significantly worse than canonical SFT;
- dev-audit: evaluated exactly once on canonical SFT seed42, 303/374 (81.02%);
- protocol status: complete; dev-audit is consumed and must not be reused for tuning.

OPD/GRPO were rejected before audit and were never evaluated on it. Intermediate
OPD/GRPO checkpoints remain diagnostic or recovery artifacts; they cannot replace
the predeclared final step-200 results.

See [`confirmatory_v2_progress.json`](confirmatory_v2_progress.json) for exact
metrics and hashes, and
[`../../configs/confirmatory/README.md`](../../configs/confirmatory/README.md)
for the frozen protocol and commands.
