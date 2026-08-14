import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_PATH = Path("data/gsm8k_sft_formal.json")

SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the problem step by step, but keep the reasoning concise. "
    "At the end, output the final numerical answer in the exact format: "
    "#### <answer>"
)


def main():
    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train"
    )

    records = []

    for sample in dataset:
        records.append({
            "instruction": sample["question"],
            "input": "",
            "output": sample["answer"],
            "system": SYSTEM_PROMPT
        })

    with open(
        OUTPUT_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Samples:", len(records))
    print("Saved to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()