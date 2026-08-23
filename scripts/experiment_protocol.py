"""Shared helpers for reproducible confirmatory experiment protocols."""

from __future__ import annotations

import hashlib
import json
from argparse import Namespace
from pathlib import Path


DEFAULT_SEED = 42
PARTITION_NAMES = ("train", "dev_select", "dev_audit")


def resolved_seed(args: Namespace, name: str) -> int:
    """Return an explicit seed, falling back to the legacy seed and then 42."""
    value = getattr(args, name, None)
    if value is not None:
        return int(value)
    legacy = getattr(args, "seed", None)
    return DEFAULT_SEED if legacy is None else int(legacy)


def resolve_seed_args(args: Namespace) -> Namespace:
    """Resolve independent seeds while preserving the deprecated ``--seed`` API."""
    explicit_names = ("data_seed", "training_seed", "generation_seed")
    explicit_values = [getattr(args, name, None) for name in explicit_names]
    legacy = getattr(args, "seed", None)
    if legacy is not None and any(value is not None for value in explicit_values):
        raise ValueError(
            "deprecated --seed cannot be combined with --data-seed, "
            "--training-seed, or --generation-seed"
        )
    for name in explicit_names:
        setattr(args, name, resolved_seed(args, name))
    return args


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_manifest(
    manifest_path: Path,
    dataset_path: Path,
    num_records: int,
) -> dict[str, list[int]]:
    """Load and validate a frozen train/dev-select/dev-audit partition manifest."""
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)
    if manifest.get("schema_version") != 1:
        raise ValueError("split manifest schema_version must be 1")
    if manifest.get("dataset_num_records") != num_records:
        raise ValueError("split manifest dataset size does not match the input dataset")
    expected_hash = manifest.get("dataset_sha256")
    if not isinstance(expected_hash, str) or expected_hash != sha256_file(dataset_path):
        raise ValueError("split manifest dataset SHA-256 does not match the input dataset")

    raw_partitions = manifest.get("partitions")
    if not isinstance(raw_partitions, dict):
        raise ValueError("split manifest has no partitions object")
    partitions: dict[str, list[int]] = {}
    seen: set[int] = set()
    for name in PARTITION_NAMES:
        indices = raw_partitions.get(name)
        if not isinstance(indices, list) or not all(
            isinstance(index, int) for index in indices
        ):
            raise ValueError(f"split manifest partition {name!r} must be an integer list")
        if len(indices) != len(set(indices)):
            raise ValueError(f"split manifest partition {name!r} contains duplicates")
        if any(index < 0 or index >= num_records for index in indices):
            raise ValueError(f"split manifest partition {name!r} has an invalid index")
        overlap = seen.intersection(indices)
        if overlap:
            raise ValueError("split manifest partitions overlap")
        seen.update(indices)
        partitions[name] = indices
    if seen != set(range(num_records)):
        raise ValueError("split manifest partitions do not cover the full dataset")
    if not partitions["train"] or not partitions["dev_select"] or not partitions["dev_audit"]:
        raise ValueError("split manifest partitions must all be non-empty")
    return partitions
