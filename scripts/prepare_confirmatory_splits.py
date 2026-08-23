"""Create frozen train/dev-select/dev-audit files for confirmatory experiments."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

from datasets import Dataset

from experiment_protocol import sha256_file


DEFAULT_INPUT = Path("data/gsm8k_sft_clean.json")
DEFAULT_OUTPUT_DIR = Path("data/confirmatory_v2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze reproducible train, selection-dev, and audit-dev partitions."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--selection-size", type=float, default=0.05)
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--audit-size", type=float, default=0.05)
    parser.add_argument("--audit-seed", type=int, default=20260823)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def holdout_count(num_records: int, size: float) -> int:
    if not 0.0 < size < 1.0:
        raise ValueError("partition sizes must be fractions in (0, 1)")
    return math.ceil(num_records * size)


def build_partitions(
    num_records: int,
    selection_size: float,
    selection_seed: int,
    audit_size: float,
    audit_seed: int,
) -> dict[str, list[int]]:
    """Preserve the legacy HF validation split, then sample a fresh audit set."""
    if num_records < 3:
        raise ValueError("at least three records are required")
    selection_count = holdout_count(num_records, selection_size)
    audit_count = holdout_count(num_records, audit_size)
    if selection_count + audit_count >= num_records:
        raise ValueError("selection and audit partitions leave no training records")

    source_dataset = Dataset.from_dict({"source_index": list(range(num_records))})
    legacy_split = source_dataset.train_test_split(
        test_size=selection_count,
        seed=selection_seed,
    )
    train_candidates = list(legacy_split["train"]["source_index"])
    dev_select = list(legacy_split["test"]["source_index"])
    dev_audit = random.Random(audit_seed).sample(train_candidates, audit_count)
    audit_set = set(dev_audit)
    train = [index for index in train_candidates if index not in audit_set]
    return {"train": train, "dev_select": dev_select, "dev_audit": dev_audit}


def write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    with args.input.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a non-empty JSON list")

    outputs = {
        "train": args.output_dir / "train.json",
        "dev_select": args.output_dir / "dev_select.json",
        "dev_audit": args.output_dir / "dev_audit.json",
        "manifest": args.output_dir / "split_manifest.json",
        "dataset_info": args.output_dir / "dataset_info.json",
    }
    existing = [path for path in outputs.values() if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"outputs already exist: {names} (pass --overwrite)")

    partitions = build_partitions(
        len(records),
        args.selection_size,
        args.selection_seed,
        args.audit_size,
        args.audit_seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in ("train", "dev_select", "dev_audit"):
        partition_records = []
        for source_index in partitions[name]:
            item = dict(records[source_index])
            item["source_index"] = source_index
            partition_records.append(item)
        write_json(outputs[name], partition_records)

    manifest = {
        "schema_version": 1,
        "protocol": "gsm8k_confirmatory_v2",
        "dataset": str(args.input),
        "dataset_num_records": len(records),
        "dataset_sha256": sha256_file(args.input),
        "selection_size": args.selection_size,
        "selection_seed": args.selection_seed,
        "audit_size": args.audit_size,
        "audit_seed": args.audit_seed,
        "partitions": partitions,
    }
    write_json(outputs["manifest"], manifest)
    dataset_template = {
        "formatting": "alpaca",
        "columns": {
            "prompt": "instruction",
            "query": "input",
            "response": "output",
            "system": "system",
        },
    }
    dataset_info = {
        "gsm8k_confirmatory_train": {
            "file_name": "train.json",
            **dataset_template,
        },
        "gsm8k_confirmatory_dev_select": {
            "file_name": "dev_select.json",
            **dataset_template,
        },
        "gsm8k_confirmatory_dev_audit": {
            "file_name": "dev_audit.json",
            **dataset_template,
        },
    }
    write_json(outputs["dataset_info"], dataset_info)
    print(
        "Frozen partitions: "
        + ", ".join(f"{name}={len(indices)}" for name, indices in partitions.items())
    )
    print("Saved to:", args.output_dir)


if __name__ == "__main__":
    main()
