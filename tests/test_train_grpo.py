import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_grpo import (  # noqa: E402
    GRPOConfig,
    build_grpo_config_kwargs,
    build_grpo_records,
    completion_to_text,
    ensure_trl_model_compatibility,
    numeric_accuracy_reward,
    strict_format_reward,
    validate_args,
)


class TrainGrpoTest(unittest.TestCase):
    def sample_records(self):
        system = "Return #### <answer>."
        return [
            {
                "instruction": f"Question {index}",
                "input": "",
                "output": f"Reasoning\n#### {index}",
                "system": system,
            }
            for index in range(8)
        ]

    def make_args(self, adapter: Path, dataset: Path, **overrides):
        values = {
            "adapter": adapter,
            "dataset": dataset,
            "num_samples": 8,
            "all_training_samples": False,
            "validation_size": 0.05,
            "max_steps": 1,
            "max_prompt_length": 512,
            "max_completion_length": 128,
            "num_generations": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 5.0e-6,
            "temperature": 0.9,
            "top_p": 1.0,
            "beta": 0.0,
            "accuracy_reward_weight": 1.0,
            "format_reward_weight": 0.1,
            "mask_truncated_completions": True,
            "save_steps": 0,
            "save_total_limit": 5,
            "resume_from_checkpoint": None,
            "output_dir": Path("outputs/grpo/test"),
            "seed": 42,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_build_records_is_deterministic_and_keeps_numeric_label(self):
        records = self.sample_records()
        first = build_grpo_records(records, 3, 43, [0, 2, 4, 6])
        second = build_grpo_records(records, 3, 43, [0, 2, 4, 6])
        self.assertEqual(first, second)
        self.assertEqual(len(first), 3)
        self.assertTrue({item["source_index"] for item in first} <= {0, 2, 4, 6})
        for item in first:
            self.assertEqual(item["prompt"][-1]["role"], "user")
            self.assertEqual(item["ground_truth"], str(item["source_index"]))

    def test_build_records_rejects_missing_answer(self):
        records = self.sample_records()
        records[0]["output"] = "No final marker"
        with self.assertRaisesRegex(ValueError, "parseable"):
            build_grpo_records(records, 1, 1, [0])

    def test_completion_normalization_and_rewards(self):
        completions = [
            "work\n#### 5",
            [{"role": "assistant", "content": "work\n#### 6"}],
            {"role": "assistant", "content": "The answer is 7"},
        ]
        self.assertEqual(completion_to_text(completions[1]), "work\n#### 6")
        self.assertEqual(
            numeric_accuracy_reward(completions, ["5", "9", "7"]),
            [1.0, 0.0, 1.0],
        )
        self.assertEqual(strict_format_reward(completions), [1.0, 1.0, 0.0])

    def test_model_compatibility_restores_warnings_dictionary(self):
        class ModelWithoutWarnings:
            pass

        model = ModelWithoutWarnings()
        self.assertEqual(
            ensure_trl_model_compatibility(model),
            ["warnings_issued"],
        )
        self.assertEqual(model.warnings_issued, {})
        self.assertEqual(ensure_trl_model_compatibility(model), [])

    def test_validate_args_checks_group_batch_and_reference(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter"
            adapter.mkdir()
            dataset = root / "data.json"
            dataset.write_text("[]", encoding="utf-8")
            validate_args(self.make_args(adapter, dataset))
            with self.assertRaisesRegex(ValueError, "divisible"):
                validate_args(
                    self.make_args(
                        adapter,
                        dataset,
                        num_generations=4,
                        gradient_accumulation_steps=2,
                    )
                )
            with self.assertRaisesRegex(ValueError, "Raw Base"):
                validate_args(self.make_args(adapter, dataset, beta=0.01))

    def test_grpo_config_kwargs_match_installed_trl(self):
        args = self.make_args(Path("adapter"), Path("dataset"))
        kwargs = build_grpo_config_kwargs(args)
        supported = set(inspect.signature(GRPOConfig).parameters)
        self.assertEqual(set(kwargs) - supported, set())
        self.assertEqual(kwargs["reward_weights"], [1.0, 0.1])
        self.assertFalse(kwargs["use_vllm"])


if __name__ == "__main__":
    unittest.main()
