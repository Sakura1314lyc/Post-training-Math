import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from compare_base_sft import validate_comparison_protocol  # noqa: E402


class CompareBaseSftTest(unittest.TestCase):
    def payload(self):
        return {
            "evaluation_version": "gsm8k_numeric_v3",
            "benchmark": "gsm8k",
            "dataset": {
                "name": "openai/gsm8k",
                "config": "main",
                "evaluation_split": "test",
            },
            "prompt": {"system": "Solve it", "chat_template": "test"},
            "generation": {
                "do_sample": False,
                "max_new_tokens": 1024,
                "stop_after_completed_answer_line": False,
            },
        }

    def test_matching_protocol_is_accepted(self):
        base = self.payload()
        candidate = self.payload()
        validate_comparison_protocol(base, candidate)

    def test_different_generation_protocol_is_rejected(self):
        base = self.payload()
        candidate = self.payload()
        candidate["generation"]["max_new_tokens"] = 512
        with self.assertRaisesRegex(ValueError, "generation"):
            validate_comparison_protocol(base, candidate)

    def test_missing_legacy_metadata_is_accepted(self):
        base = self.payload()
        candidate = self.payload()
        del base["benchmark"]
        validate_comparison_protocol(base, candidate)

    def test_missing_generation_metadata_is_rejected(self):
        base = self.payload()
        candidate = self.payload()
        del base["generation"]
        with self.assertRaisesRegex(ValueError, "generation"):
            validate_comparison_protocol(base, candidate)

    def test_different_evaluator_is_rejected(self):
        base = self.payload()
        candidate = self.payload()
        candidate["evaluation_version"] = "svamp_numeric_v1"
        with self.assertRaisesRegex(ValueError, "different evaluators"):
            validate_comparison_protocol(base, candidate)


if __name__ == "__main__":
    unittest.main()
