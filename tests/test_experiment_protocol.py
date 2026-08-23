import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from experiment_protocol import (  # noqa: E402
    load_split_manifest,
    resolve_seed_args,
    sha256_file,
)


class ExperimentProtocolTest(unittest.TestCase):
    def test_explicit_seeds_are_independent(self):
        args = Namespace(
            seed=None,
            data_seed=11,
            training_seed=22,
            generation_seed=33,
        )
        resolve_seed_args(args)
        self.assertEqual(
            (args.data_seed, args.training_seed, args.generation_seed),
            (11, 22, 33),
        )

    def test_legacy_seed_sets_all_three(self):
        args = Namespace(
            seed=44,
            data_seed=None,
            training_seed=None,
            generation_seed=None,
        )
        resolve_seed_args(args)
        self.assertEqual(
            (args.data_seed, args.training_seed, args.generation_seed),
            (44, 44, 44),
        )

    def test_legacy_and_explicit_seeds_cannot_be_mixed(self):
        args = Namespace(
            seed=44,
            data_seed=42,
            training_seed=None,
            generation_seed=None,
        )
        with self.assertRaisesRegex(ValueError, "cannot be combined"):
            resolve_seed_args(args)

    def test_manifest_validates_hash_coverage_and_disjointness(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset = root / "data.json"
            dataset.write_text('[{"id": 0}, {"id": 1}, {"id": 2}]', encoding="utf-8")
            manifest = root / "manifest.json"
            payload = {
                "schema_version": 1,
                "dataset_num_records": 3,
                "dataset_sha256": sha256_file(dataset),
                "partitions": {
                    "train": [0],
                    "dev_select": [1],
                    "dev_audit": [2],
                },
            }
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(
                load_split_manifest(manifest, dataset, 3),
                payload["partitions"],
            )
            payload["partitions"]["dev_audit"] = [1]
            manifest.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "overlap"):
                load_split_manifest(manifest, dataset, 3)


if __name__ == "__main__":
    unittest.main()
