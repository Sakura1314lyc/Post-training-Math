# Confirmatory v2 artifacts

This directory contains small, tracked evaluation artifacts for the new
confirmatory protocol. Model checkpoints and generated split copies remain under
ignored `outputs/` and `data/confirmatory_v2/*.json` paths.

Current status on 2026-08-23:

- frozen split: 6,725 train / 374 dev-select / 374 dev-audit;
- SFT seeds 42/43/44: training and dev-select generation complete;
- canonical downstream SFT policy: seed 42, fixed before downstream evaluation;
- OPD data42/train42/gen42: 200-step training complete, evaluation pending;
- OPD train43/train44 and all confirmatory GRPO runs: pending;
- dev-audit: sealed and unevaluated.

`dev-audit` must not be evaluated until all OPD/GRPO settings and the final
adapter rule are frozen. Intermediate OPD/GRPO checkpoints are recovery
artifacts; the confirmatory result is always the final step-200 adapter.

See [`confirmatory_v2_progress.json`](confirmatory_v2_progress.json) for exact
metrics and hashes, and
[`../../configs/confirmatory/README.md`](../../configs/confirmatory/README.md)
for the frozen protocol and commands.
