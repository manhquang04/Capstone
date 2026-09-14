"""Development-only numerical-stability study for the RQ1 MSE tie threshold."""

from __future__ import annotations

import argparse
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


DEFAULT_PHASE4 = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
DEFAULT_DIRS = [
    "dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001",
    "simple_defense_attack_clipping_noise_mc_development_gate_r8_i600_lr0p1_nonneg0p001_rel0p1_mc100",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase4-root", type=Path, default=DEFAULT_PHASE4)
    parser.add_argument("--artifact-dirs", nargs="+", default=DEFAULT_DIRS)
    parser.add_argument("--floor", type=float, default=1e-12)
    parser.add_argument("--safety-factor", type=float, default=10.0)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/rq1/stage1_tie_threshold_20260912/tie_threshold_report.json",
    )
    args = parser.parse_args()

    discrepancies = []
    checked = 0
    for relative in args.artifact_dirs:
        for path in sorted((args.phase4_root / relative).glob("group_*/**/*.pt")):
            artifact = torch.load(path, map_location="cpu", weights_only=False)
            original = artifact["original"].detach().cpu().numpy()
            aligned = artifact.get("aligned", artifact["reconstruction"]).detach().cpu().numpy()
            value_a = reconstruction_metrics(original, aligned)["feature_mse"]
            value_b = reconstruction_metrics(original.copy(), aligned.copy())["feature_mse"]
            # Independent torch implementation, retaining the artifact's
            # float32 arithmetic contract.
            value_c = float(torch.mean((torch.as_tensor(aligned) - torch.as_tensor(original)).square()))
            discrepancies.append(max(abs(value_a - value_b), abs(value_a - value_c)))
            checked += 1

    observed_max = float(max(discrepancies, default=0.0))
    threshold = float(max(args.floor, args.safety_factor * observed_max))
    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only numerical replay stability; no privacy-direction selection",
        "artifact_dirs": args.artifact_dirs,
        "artifacts_checked": checked,
        "metric": "normalized_feature_mse_float32",
        "observed_max_absolute_replay_discrepancy": observed_max,
        "safety_factor": args.safety_factor,
        "absolute_floor": args.floor,
        "selected_tie_threshold_mse": threshold,
        "selection_rule": "max(floor, safety_factor * maximum replay discrepancy)",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
