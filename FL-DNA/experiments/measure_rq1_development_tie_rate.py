"""Measure the RQ1 tie rate on the registered development groups only.

Candidate selection uses only the stored attacker-visible objective. The
normalized feature-MSE is evaluated after selection and never selects a
candidate. This utility must not be run on post-hoc or confirmatory targets.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import torch

from attacks.inversion_metrics import reconstruction_metrics


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_best(paths: list[Path]) -> tuple[Path, dict]:
    candidates = []
    for path in paths:
        artifact = torch.load(path, map_location="cpu", weights_only=False)
        if artifact.get("method") != "baseline":
            continue
        candidates.append((float(artifact["best_objective"]), str(path), path, artifact))
    if not candidates:
        raise RuntimeError("no baseline candidates found")
    _, _, path, artifact = min(candidates, key=lambda item: (item[0], item[1]))
    return path, artifact


def metric(artifact: dict) -> float:
    original = artifact["original"].detach().cpu().numpy()
    aligned = artifact.get("aligned", artifact["reconstruction"]).detach().cpu().numpy()
    return float(reconstruction_metrics(original, aligned)["feature_mse"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dna-dir", type=Path, required=True)
    parser.add_argument("--dp-root", type=Path, required=True)
    parser.add_argument("--target-file", type=Path, required=True)
    parser.add_argument("--tie-threshold", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    target_groups = torch.load(args.target_file, map_location="cpu", weights_only=False)
    rows = []
    for group_id, target in enumerate(target_groups):
        dna_path, dna = load_best(list((args.dna_dir / f"group_{group_id}").glob("**/baseline.pt")))
        dp_path, dp = load_best(
            list((args.dp_root / f"group_{group_id}_run").glob("**/group_*/restart_*/baseline.pt"))
        )
        target_ids = list(map(int, target["source_ids"]))
        dna_ids = list(map(int, dna["source_ids"]))
        dp_ids = list(map(int, dp["source_ids"]))
        if target_ids != dna_ids or target_ids != dp_ids:
            raise AssertionError(f"source-id mismatch for group {group_id}")
        dna_mse = metric(dna)
        dp_mse = metric(dp)
        difference = dna_mse - dp_mse
        rows.append(
            {
                "group_id": group_id,
                "dna_selected_artifact": str(dna_path),
                "dp_selected_artifact": str(dp_path),
                "dna_selection_objective": float(dna["best_objective"]),
                "dp_selection_objective": float(dp["best_objective"]),
                "dna_normalized_feature_mse": dna_mse,
                "dp_normalized_feature_mse": dp_mse,
                "difference_dna_minus_dp": difference,
                "absolute_difference": abs(difference),
                "tie": abs(difference) <= args.tie_threshold,
            }
        )

    ties = sum(bool(row["tie"]) for row in rows)
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "registered development groups only",
        "selection_rule": "minimum attacker-visible best_objective per method/group",
        "metric": "normalized_feature_mse after attacker-visible selection",
        "tie_rule": "abs(MSE_DNA - MSE_DP) <= tie_threshold_mse",
        "tie_threshold_mse": args.tie_threshold,
        "groups": len(rows),
        "ties": ties,
        "measured_tie_rate": ties / len(rows),
        "target_file": str(args.target_file),
        "target_sha256": sha256(args.target_file),
        "dna_dir": str(args.dna_dir),
        "dp_root": str(args.dp_root),
        "rows_file": str(args.output_dir / "tie_rate_rows.csv"),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "tie_rate_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "tie_rate_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
