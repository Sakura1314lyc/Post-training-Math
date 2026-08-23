"""Merge an SFT LoRA into its base model for a correct GRPO KL reference.

This is a preparation command, not a training command. The merged directory is
used as the immutable initial policy; train_grpo.py adds a fresh GRPO adapter.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Math-1.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.adapter.is_dir():
        raise FileNotFoundError(f"adapter directory does not exist: {args.adapter}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir} (choose a new directory)"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    peft_config = PeftConfig.from_pretrained(args.adapter)
    configured_base = getattr(peft_config, "base_model_name_or_path", None)
    if configured_base and configured_base != args.base_model:
        raise ValueError(
            "adapter base model does not match --base-model: "
            f"{configured_base!r} != {args.base_model!r}"
        )

    print("Loading base model in BF16...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )
    print("Loading and merging SFT adapter...")
    sft_policy = PeftModel.from_pretrained(base_model, args.adapter)
    merged_model = sft_policy.merge_and_unload(safe_merge=True)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    print("Saving merged SFT policy...")
    merged_model.save_pretrained(
        args.output_dir,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(args.output_dir)
    provenance = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model": args.base_model,
        "sft_adapter": str(args.adapter),
        "merge_method": "PeftModel.merge_and_unload(safe_merge=True)",
        "dtype": "bfloat16",
        "purpose": "immutable_sft_reference_for_fresh_grpo_lora",
    }
    with (args.output_dir / "merge_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(provenance, file, ensure_ascii=False, indent=2)
    print("Saved merged SFT policy to:", args.output_dir)


if __name__ == "__main__":
    main()
