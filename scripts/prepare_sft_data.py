import json
from pathlib import Path

from datasets import load_dataset


OUTPUT_DIR = Path("data")

SMOKE_SIZE = 512
SEED = 42


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Loading GSM8K train split...")

    dataset = load_dataset(
        "openai/gsm8k",
        "main",
        split="train"
    )

    print("Train samples:", len(dataset))

    # 固定随机种子，保证 smoke 数据可复现
    dataset = dataset.shuffle(seed=SEED)

    records = []

    for sample in dataset:

        record = {
            "instruction": (
                "Solve the following math problem step by step. "
                "End your response with the final answer "
                "in the format: #### <answer>"
            ),

            "input": sample["question"],

            # GSM8K 本身已经包含 reasoning + #### answer
            "output": sample["answer"]
        }

        records.append(record)

    # --------------------------------------------------------
    # 小数据集：先验证整个 SFT pipeline
    # --------------------------------------------------------
    smoke_path = OUTPUT_DIR / "gsm8k_sft_smoke.json"

    with open(
        smoke_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records[:SMOKE_SIZE],
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------------
    # 完整训练数据
    # --------------------------------------------------------
    full_path = OUTPUT_DIR / "gsm8k_sft_train.json"

    with open(
        full_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            records,
            f,
            ensure_ascii=False,
            indent=2
        )

    print("Smoke samples:", SMOKE_SIZE)
    print("Full samples:", len(records))

    print("Saved:")
    print(smoke_path)
    print(full_path)


if __name__ == "__main__":
    main()