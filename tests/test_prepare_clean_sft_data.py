import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from prepare_clean_sft_data import build_clean_records, clean_response  # noqa: E402


class PrepareCleanSftDataTest(unittest.TestCase):
    def test_clean_response_removes_only_calculator_annotations(self):
        response = "Compute 2+3 = <<2+3=5>>5.\n#### 5"
        cleaned, removed = clean_response(response)

        self.assertEqual(cleaned, "Compute 2+3 = 5.\n#### 5")
        self.assertEqual(removed, 1)

    def test_build_clean_records_preserves_fields_order_and_answers(self):
        records = [
            {
                "instruction": "Question A",
                "input": "",
                "output": "First <<1+1=2>>2.\n#### 2",
                "system": "System prompt",
            },
            {
                "instruction": "Question B",
                "input": "",
                "output": "No annotation.\n#### 3",
                "system": "System prompt",
            },
        ]

        cleaned, removed = build_clean_records(records)

        self.assertEqual(removed, 1)
        self.assertEqual([item["instruction"] for item in cleaned], ["Question A", "Question B"])
        self.assertEqual(cleaned[0]["output"], "First 2.\n#### 2")
        self.assertEqual(cleaned[1], records[1])
        self.assertEqual(cleaned[0]["system"], records[0]["system"])

    def test_build_clean_records_rejects_unhandled_angle_markers(self):
        records = [
            {
                "instruction": "Question",
                "input": "",
                "output": "Malformed <<annotation\n#### 4",
            }
        ]

        with self.assertRaisesRegex(ValueError, "unhandled angle annotation"):
            build_clean_records(records)


if __name__ == "__main__":
    unittest.main()
