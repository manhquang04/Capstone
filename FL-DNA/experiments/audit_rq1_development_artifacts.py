"""Audit Phase-4 development artifacts for the RQ1 Stage-1 protocol.

This script is read-only with respect to historical Phase-4 artifacts.  Any
metadata absent from those artifacts is written as a supplemental record in a
new Stage-1 directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.inversion_metrics import reconstruction_metrics


REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"
DEFAULT_PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"

BRANCH_DIRS = {
    "raw": "harddiff_reparam_development_standard_nonneg_0p001",
    "dna_level1": "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001",
    "clipping_noise_mc": "simple_defense_attack_clipping_noise_mc_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_mc100",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_array(value: object) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def audit_branch(branch: str, directory: Path, targets: list[dict]) -> dict:
    files = sorted(directory.glob("group_*/**/*.pt"))
    counts = {
        "artifact_files": len(files),
        "complete_vectors": 0,
        "source_ids_match": 0,
        "finite_vectors": 0,
        "metric_replay_pass": 0,
    }
    failures: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(directory)
        try:
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            group_id = int(next(part for part in relative.parts if part.startswith("group_")).split("_")[1])
            original = tensor_array(artifact["original"])
            reconstruction = tensor_array(artifact["reconstruction"])
            aligned = tensor_array(artifact.get("aligned", artifact["reconstruction"]))
            labels = tensor_array(artifact["labels"])
            source_ids = [int(value) for value in artifact["source_ids"]]
            if original.shape != reconstruction.shape or original.shape != aligned.shape:
                raise ValueError(f"shape mismatch: {original.shape}, {reconstruction.shape}, {aligned.shape}")
            if labels.shape[0] != original.shape[0]:
                raise ValueError("label/vector row mismatch")
            counts["complete_vectors"] += 1
            if source_ids != [int(value) for value in targets[group_id]["source_ids"]]:
                raise ValueError("source_ids do not match development target")
            counts["source_ids_match"] += 1
            if not all(np.all(np.isfinite(item)) for item in (original, reconstruction, aligned)):
                raise ValueError("non-finite reconstruction vector")
            counts["finite_vectors"] += 1
            first = reconstruction_metrics(original, aligned)
            second = reconstruction_metrics(original, aligned)
            if first != second:
                raise ValueError("metric replay is not bit-deterministic")
            counts["metric_replay_pass"] += 1
        except Exception as error:  # report every corrupt/incomplete candidate
            failures.append({"path": str(relative), "error": f"{type(error).__name__}: {error}"})
    return {
        "branch": branch,
        "directory": str(directory),
        "counts": counts,
        "failures": failures,
        "pass": bool(files) and not failures and all(value == len(files) for value in counts.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--target-file", default="development_gate_targets.pt")
    parser.add_argument("--raw-dir", default=BRANCH_DIRS["raw"])
    parser.add_argument("--dna-dir", default=BRANCH_DIRS["dna_level1"])
    parser.add_argument("--dp-dir", default=BRANCH_DIRS["clipping_noise_mc"])
    parser.add_argument(
        "--branches",
        nargs="+",
        choices=("raw", "dna_level1", "clipping_noise_mc"),
        default=["raw", "dna_level1", "clipping_noise_mc"],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts/rq1/stage1_development_audit_20260912",
    )
    args = parser.parse_args()

    phase4 = args.phase4_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    target_path = phase4 / args.target_file
    targets = torch.load(target_path, map_location="cpu", weights_only=False)
    preprocessing_path = REFERENCE / "preprocessing.json"
    preprocessing = json.loads(preprocessing_path.read_text())

    supplemental = {
        "schema_version": 1,
        "role": "supplemental metadata for immutable historical development artifacts",
        "target_file": str(target_path),
        "target_sha256": sha256(target_path),
        "target_group_count": len(targets),
        "feature_order": preprocessing["feature_names"],
        "numeric_center": preprocessing["numeric_center"],
        "numeric_scale": preprocessing["numeric_scale"],
        "type_categories": preprocessing["type_categories"],
        "preprocessing_file": str(preprocessing_path),
        "preprocessing_sha256": sha256(preprocessing_path),
        "preprocessing_canonical_json_sha256": canonical_json_hash(preprocessing),
        "metric_implementation": "attacks/inversion_metrics.py",
        "metric_sha256": sha256(ROOT / "attacks/inversion_metrics.py"),
        "pseudo_image_implementation": "attacks/pseudo_image.py",
        "pseudo_image_sha256": sha256(ROOT / "attacks/pseudo_image.py"),
        "pseudo_image_contract": "square zero-padding plus joint min-max normalization",
        "checkpoint": str(REFERENCE / "pre_local.pt"),
        "checkpoint_sha256": sha256(REFERENCE / "pre_local.pt"),
        "dataset_sha256": sha256(ROOT / "datasets/creditcard.csv"),
    }
    (output / "supplemental_metadata.json").write_text(json.dumps(supplemental, indent=2) + "\n")

    requested_dirs = {
        "raw": args.raw_dir,
        "dna_level1": args.dna_dir,
        "clipping_noise_mc": args.dp_dir,
    }
    branch_results = [
        audit_branch(branch, phase4 / requested_dirs[branch], targets)
        for branch in args.branches
    ]
    report = {
        "schema_version": 1,
        "run_id": output.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "RQ1 Stage-1 development-only artifact audit; no post-hoc target accessed",
        "phase4_root": str(phase4),
        "target_file": str(target_path),
        "historical_artifacts_modified": False,
        "branch_results": branch_results,
        "supplemental_metadata": str(output / "supplemental_metadata.json"),
        "gate_pass": all(item["pass"] for item in branch_results),
    }
    (output / "artifact_audit_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["gate_pass"] else 1)


if __name__ == "__main__":
    main()
