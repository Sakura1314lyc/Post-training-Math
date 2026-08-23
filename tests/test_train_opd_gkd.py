import argparse
import inspect
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_opd_gkd import (  # noqa: E402
    GKDConfig,
    build_gkd_config_kwargs,
    build_conversational_records,
    resolve_terminators,
    split_source_indices,
    validate_args,
    write_json_atomic,
)


class FakeTokenizer:
    eos_token_id = 10

    @staticmethod
    def convert_tokens_to_ids(token):
        return 11 if token == "<|im_end|>" else -1


class TrainOpdGkdTest(unittest.TestCase):
    def test_build_conversational_records_is_deterministic_and_preserves_roles(self):
        records = [
            {
                "instruction": f"Question {index}",
                "input": "Context" if index == 2 else "",
                "output": f"Reasoning\n#### {index}",
                "system": "Math system prompt",
            }
            for index in range(4)
        ]

        first = build_conversational_records(records, num_samples=4, seed=42)
        second = build_conversational_records(records, num_samples=4, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        for item in first:
            self.assertEqual(
                [message["role"] for message in item["messages"]],
                ["system", "user", "assistant"],
            )
        context_item = next(
            item for item in first if item["source_index"] == 2
        )
        self.assertEqual(context_item["messages"][1]["content"], "Question 2\n\nContext")

    def test_build_conversational_records_rejects_missing_output(self):
        records = [
            {
                "instruction": "Question",
                "input": "",
                "output": "",
                "system": "System prompt",
            }
        ]

        with self.assertRaisesRegex(ValueError, "no valid output"):
            build_conversational_records(records, num_samples=1, seed=42)

    def test_training_candidates_exclude_fixed_validation_split(self):
        records = [
            {
                "instruction": f"Question {index}",
                "input": "",
                "output": f"Reasoning\n#### {index}",
                "system": "Math system prompt",
            }
            for index in range(20)
        ]
        training_indices, validation_indices = split_source_indices(
            num_records=20, validation_size=0.2, seed=42
        )

        self.assertEqual(len(training_indices), 16)
        self.assertEqual(len(validation_indices), 4)
        self.assertTrue(set(training_indices).isdisjoint(validation_indices))
        selected = build_conversational_records(
            records,
            num_samples=10,
            seed=42,
            candidate_source_indices=training_indices,
        )
        self.assertTrue(
            {item["source_index"] for item in selected}.isdisjoint(validation_indices)
        )

    def test_resolve_terminators_includes_base_and_chatml_eos(self):
        self.assertEqual(resolve_terminators(FakeTokenizer()), [10, 11])

    def test_validate_args_checks_paths_and_ranges(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            adapter = root / "adapter"
            adapter.mkdir()
            dataset = root / "dataset.json"
            dataset.write_text("[]", encoding="utf-8")
            args = argparse.Namespace(
                num_samples=1,
                all_training_samples=False,
                validation_size=0.05,
                max_steps=1,
                max_length=512,
                max_new_tokens=256,
                learning_rate=1.0e-5,
                temperature=0.9,
                lmbda=1.0,
                beta=0.5,
                gradient_accumulation_steps=1,
                save_steps=0,
                save_total_limit=5,
                adapter=adapter,
                dataset=dataset,
                resume_from_checkpoint=None,
            )

            validate_args(args)
            args.lmbda = 1.1
            with self.assertRaisesRegex(ValueError, "lmbda"):
                validate_args(args)

    def test_write_json_atomic_replaces_existing_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "summary.json"
            output.write_text("incomplete", encoding="utf-8")

            write_json_atomic(output, {"source_indices": [1, 2, 3]})

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                '{\n  "source_indices": [\n    1,\n    2,\n    3\n  ]\n}',
            )
            self.assertFalse((Path(temp_dir) / "summary.json.tmp").exists())

    def test_build_gkd_config_accepts_current_trl_api(self):
        args = argparse.Namespace(
            output_dir=Path("outputs/test"),
            max_steps=50,
            gradient_accumulation_steps=4,
            learning_rate=5.0e-6,
            max_length=512,
            max_new_tokens=256,
            temperature=0.9,
            lmbda=1.0,
            beta=0.5,
            use_liger_kernel=False,
            save_steps=10,
            save_total_limit=5,
            seed=42,
        )

        config_kwargs = build_gkd_config_kwargs(args)
        supported_parameters = set(inspect.signature(GKDConfig).parameters)

        self.assertEqual(set(config_kwargs) - supported_parameters, set())
        self.assertEqual(config_kwargs["max_steps"], 50)
        self.assertEqual(config_kwargs["save_steps"], 10)
        self.assertEqual(config_kwargs["save_strategy"], "steps")
        self.assertEqual(config_kwargs["lmbda"], 1.0)
        self.assertTrue(config_kwargs["disable_dropout"])


if __name__ == "__main__":
    unittest.main()
