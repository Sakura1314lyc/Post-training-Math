import inspect
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_grpo import (  # noqa: E402
    GRPOConfig,
    arithmetic_consistency_reward,
    arithmetic_consistency_score,
    build_grpo_config_kwargs,
    build_grpo_records,
    build_reward_spec,
    completion_to_text,
    ensure_trl_model_compatibility,
    numeric_accuracy_reward,
    policy_reference_description,
    snapshot_trainable_parameters,
    strict_format_reward,
    summarize_trainable_parameter_drift,
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
            "base_model": "base-model",
            "policy_initialization": "continued_adapter",
            "merged_sft_model": None,
            "lora_rank": 8,
            "lora_alpha": 16,
            "lora_target_modules": "q_proj,v_proj,lm_head",
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
            "arithmetic_consistency_reward_weight": 0.0,
            "mask_truncated_completions": True,
            "save_steps": 0,
            "save_total_limit": 5,
            "resume_from_checkpoint": None,
            "output_dir": Path("outputs/grpo/test"),
            "seed": 42,
            "split_manifest": None,
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

    def test_arithmetic_consistency_is_a_conservative_bonus(self):
        self.assertEqual(arithmetic_consistency_score("2 + 3 = 5\n#### 5"), 1.0)
        self.assertEqual(arithmetic_consistency_score("2 + 3 = 6\n#### 6"), 0.0)
        self.assertEqual(arithmetic_consistency_score("Reasoning only\n#### 5"), 0.0)
        self.assertEqual(
            arithmetic_consistency_reward(
                ["8 / 4 = 2\n#### 2", "8 / 4 = 3\n#### 3"]
            ),
            [1.0, 0.0],
        )

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

    def test_trainable_parameter_drift_covers_all_lora_groups(self):
        q_a = torch.nn.Parameter(torch.tensor([[1.0, 2.0]]))
        q_b = torch.nn.Parameter(torch.tensor([[0.0], [0.0]]))
        lm_b = torch.nn.Parameter(torch.tensor([[0.0, 0.0]]))
        parameters = [
            (
                "base_model.model.layers.0.q_proj.lora_A.default.weight",
                q_a,
            ),
            (
                "base_model.model.layers.0.q_proj.lora_B.default.weight",
                q_b,
            ),
            ("base_model.model.lm_head.lora_B.default.weight", lm_b),
        ]
        snapshots = snapshot_trainable_parameters(parameters)
        with torch.no_grad():
            q_a.add_(torch.tensor([[0.5, 0.0]]))
            q_b.add_(torch.tensor([[0.25], [-0.25]]))

        summary = summarize_trainable_parameter_drift(parameters, snapshots)
        self.assertEqual(summary["all"]["tensor_count"], 3)
        self.assertEqual(summary["all"]["parameter_count"], 6)
        self.assertEqual(summary["all"]["updated_tensor_count"], 2)
        self.assertAlmostEqual(summary["all"]["max_abs_delta"], 0.5)
        self.assertEqual(set(summary["by_lora_matrix"]), {"lora_A", "lora_B"})
        self.assertEqual(set(summary["by_target_module"]), {"lm_head", "q_proj"})
        self.assertEqual(
            summary["by_target_and_matrix"]["lm_head.lora_B"][
                "updated_tensor_count"
            ],
            0,
        )
        self.assertIsNone(
            summary["by_target_and_matrix"]["q_proj.lora_B"][
                "relative_l2_delta"
            ]
        )

    def test_trainable_parameter_drift_rejects_name_changes(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        snapshots = snapshot_trainable_parameters([("before", parameter)])
        with self.assertRaisesRegex(ValueError, "names changed"):
            summarize_trainable_parameter_drift([("after", parameter)], snapshots)

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
            merged = root / "merged"
            merged.mkdir()
            merged_args = self.make_args(
                adapter,
                dataset,
                policy_initialization="merged_sft",
                merged_sft_model=merged,
                beta=0.01,
            )
            validate_args(merged_args)
            description = policy_reference_description(merged_args)
            self.assertEqual(description["kl_reference"], "merged_sft_policy")
            self.assertTrue(description["kl_reference_correct"])

    def test_grpo_config_kwargs_match_installed_trl(self):
        args = self.make_args(Path("adapter"), Path("dataset"))
        kwargs = build_grpo_config_kwargs(args)
        supported = set(inspect.signature(GRPOConfig).parameters)
        self.assertEqual(set(kwargs) - supported, set())
        self.assertEqual(kwargs["reward_weights"], [1.0, 0.1])
        self.assertFalse(kwargs["use_vllm"])
        self.assertTrue(kwargs["disable_dropout"])

        args.arithmetic_consistency_reward_weight = 0.05
        reward_functions, reward_weights = build_reward_spec(args)
        self.assertEqual(len(reward_functions), 3)
        self.assertEqual(reward_weights, [1.0, 0.1, 0.05])


if __name__ == "__main__":
    unittest.main()
