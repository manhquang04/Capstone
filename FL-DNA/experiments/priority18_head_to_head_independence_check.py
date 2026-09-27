"""Priority 18 independence-assumption robustness analysis.

This script performs statistical re-analysis only.  It reads existing
head-to-head confirmatory JSON artifacts and Priority 17 replay CSVs; it does
not generate targets, train models, or run attacks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


CELLS = {
    "v1_stronger": {
        "label": "Transform v1 stronger vs DP 0.0004",
        "report": "artifacts/priority14_fresh_probe/v1_stronger_confirmatory_20260921/priority14_probe_report.json",
        "replay_csv": "artifacts/priority17_replay_stability/comparisons_20260927/v1_stronger_replay_comparison.csv",
    },
    "v1_medium": {
        "label": "Transform v1 medium vs DP 0.000315",
        "report": "artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_run_20260927/priority14_probe_report.json",
        "replay_csv": "artifacts/priority17_replay_stability/comparisons_20260927/v1_medium_replay_comparison.csv",
    },
    "v2_ratio0p95": {
        "label": "Transform v2 ratio0.95/eta0.01 vs DP 0.00105",
        "report": "artifacts/priority14_fresh_probe/v2_ratio0p95_confirmatory_20260921/priority14_probe_report.json",
        "replay_csv": "artifacts/priority17_replay_stability/comparisons_20260927/v2_ratio0p95_replay_comparison.csv",
    },
}


def sign_tail_unanimous(n: int) -> float:
    return 2.0 ** (-int(n))


def summarize_cell(report_path: Path, permutations: int, rng: np.random.Generator) -> dict:
    payload = json.loads(report_path.read_text())
    selected = payload["summary"]["selected"]
    values = np.asarray([float(row["D_dna_minus_dp"]) for row in selected], dtype=np.float64)
    dp_wins = int(np.sum(values < 0.0))
    dna_wins = int(np.sum(values > 0.0))
    ties = int(np.sum(values == 0.0))
    n = int(dp_wins + dna_wins)
    if n != len(values) or ties != 0:
        raise ValueError(f"Unexpected ties or non-tied count in {report_path}")

    # Under the paired-label permutation null, each target independently flips
    # which side is called DP.  Observing 39/39 DP wins (or more extreme) means
    # every flipped sign is still DP.
    flips = rng.random((permutations, len(values))) < 0.5
    # Original all values are negative (DP wins).  If a row is flipped, DP no
    # longer wins for that target.  Extreme count is therefore all False flips.
    simulated_dp_wins = np.sum(~flips, axis=1)
    empirical_extreme = int(np.sum(simulated_dp_wins >= dp_wins))
    empirical_p = empirical_extreme / float(permutations)

    return {
        "n": len(values),
        "dp_wins": dp_wins,
        "dna_wins": dna_wins,
        "ties": ties,
        "mean_D": float(np.mean(values)),
        "sd_D": float(np.std(values, ddof=1)),
        "mean_abs_D": float(np.mean(np.abs(values))),
        "min_D": float(np.min(values)),
        "max_D": float(np.max(values)),
        "exact_one_sided_p": sign_tail_unanimous(n),
        "permutation_trials": permutations,
        "permutation_extreme_count": empirical_extreme,
        "permutation_empirical_p": empirical_p,
        "permutation_resolution_note": (
            f"0/{permutations} extreme permutations observed; empirical p is "
            f"reported as 0.0 with Monte Carlo resolution 1/{permutations}."
            if empirical_extreme == 0
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    cells = {}
    for cell_id, spec in CELLS.items():
        cells[cell_id] = {
            "label": spec["label"],
            "report": spec["report"],
            "replay_csv": spec["replay_csv"],
            **summarize_cell(ROOT / spec["report"], args.permutations, rng),
        }

    effective_n_values = list(range(1, 11)) + [12, 15, 20, 25, 30, 35, 37, 39]
    effective_n_table = [
        {
            "effective_n": n,
            "unanimous_one_sided_p": sign_tail_unanimous(n),
        }
        for n in effective_n_values
    ]

    payload = {
        "analysis": "Priority 18 head-to-head sign-test independence robustness check",
        "inputs": CELLS,
        "permutation_seed": args.seed,
        "permutation_trials": args.permutations,
        "cells": cells,
        "effective_n_table": effective_n_table,
        "limitations": [
            "The permutation test flips target labels independently and therefore checks the targetwise paired-sign null; it does not by itself model run-level clustering.",
            "The effective-n table is a conservative sensitivity analysis for possible positive within-run correlation.",
        ],
    }
    out = args.output.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
