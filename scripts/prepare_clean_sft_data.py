"""Remove GSM8K calculator annotations while preserving answers and metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from evaluation_utils import extract_ground_truth


DEFAULT_INPUT = Path("data/gsm8k_sft_formal.json")
DEFAULT_OUTPUT = Path("data/gsm8k_sft_clean.json")
CALCULATOR_ANNOTATION_PATTERN = re.compile(r"<<[^<>\n]*>>")


def clean_response(response: str) -> tuple[str, int]:
    """Remove ``<<expression=result>>`` spans and return the removal count."""
    cleaned, removed = CALCULATOR_ANNOTATION_PATTERN.subn("", response)
    return cleaned, removed


def build_clean_records(records: list[dict]) -> tuple[list[dict], int]:
    """Clean all responses and verify that labels and record order are unchanged."""
    cleaned_records = []
    removed_total = 0

    for index, record in enumerate(records):
        response = record.get("output")
        if not isinstance(response, str):
            raise ValueError(f"record {index} does not contain a string output")

        ground_truth_before = extract_ground_truth(response)
        cleaned_response, removed = clean_response(response)
        ground_truth_after = extract_ground_truth(cleaned_response)

        if ground_truth_before is None:
            raise ValueError(f"record {index} has no GSM8K #### answer")
        if ground_truth_before != ground_truth_after:
            raise ValueError(
                f"record {index} answer changed during cleaning: "
                f"{ground_truth_before!r} -> {ground_truth_after!r}"
            )
        if "<<" in cleaned_response or ">>" in cleaned_response:
            raise ValueError(f"record {index} contains an unhandled angle annotation")

        cleaned_record = dict(record)
        cleaned_record["output"] = cleaned_response
        cleaned_records.append(cleaned_record)
        removed_total += removed

    return cleaned_records, removed_total


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create GSM8K SFT data without <<expression=result>> annotations."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"input file does not exist: {args.input}")
    if args.output.exists() and not args.overwrite:
        parser.error(f"output already exists: {args.output} (pass --overwrite to replace it)")

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError("input must be a non-empty JSON list")

    cleaned_records, removed_total = build_clean_records(records)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(cleaned_records, f, ensure_ascii=False, indent=2)

    print("Samples:", len(cleaned_records))
    print("Removed calculator annotations:", removed_total)
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
