"""Verify a confirmatory target set against every registered target artifact.

Only source identifiers are read from historical/post-hoc target files. No
features, labels, reconstructions, or outcomes are reported.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_ids(path: Path) -> set[int]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(payload, list):
        return {
            int(source_id)
            for group in payload
            for source_id in group.get("source_ids", [])
        }
    if isinstance(payload, dict) and "source_ids" in payload:
        values = payload["source_ids"]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return {int(value) for value in values}
    return set()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory-target", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    confirmatory = args.confirmatory_target.resolve()
    payload = torch.load(confirmatory, map_location="cpu", weights_only=False)
    flat = [int(value) for group in payload for value in group["source_ids"]]
    new_ids = set(flat)
    if len(new_ids) != len(flat):
        raise AssertionError("duplicate source IDs within confirmatory target set")

    matrix = {}
    load_failures = {}
    for path in sorted((ROOT / "artifacts").rglob("*targets.pt")):
        if path.resolve() == confirmatory:
            continue
        try:
            old_ids = source_ids(path)
        except (EOFError, KeyError, RuntimeError, TypeError) as error:
            load_failures[str(path.relative_to(ROOT))] = type(error).__name__
            continue
        matrix[str(path.relative_to(ROOT))] = {
            "historical_source_count": len(old_ids),
            "overlap_count": len(new_ids & old_ids),
        }

    max_overlap = max((row["overlap_count"] for row in matrix.values()), default=0)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "inspection_contract": "source IDs only for historical/post-hoc targets",
        "confirmatory_target": str(confirmatory.relative_to(ROOT)),
        "confirmatory_target_sha256": sha256(confirmatory),
        "confirmatory_groups": len(payload),
        "confirmatory_source_count": len(new_ids),
        "unique_within_confirmatory": len(new_ids) == len(flat),
        "historical_target_files_checked": len(matrix),
        "load_failures": load_failures,
        "max_overlap": max_overlap,
        "disjointness_gate": "PASS" if max_overlap == 0 and not load_failures else "FAIL",
        "overlap_matrix": matrix,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "confirmatory_target_sha256", "confirmatory_groups",
        "confirmatory_source_count", "historical_target_files_checked",
        "load_failures", "max_overlap", "disjointness_gate"
    )}, indent=2))


if __name__ == "__main__":
    main()
