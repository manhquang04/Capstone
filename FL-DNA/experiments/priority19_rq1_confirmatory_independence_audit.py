"""Priority 19 independence robustness re-analysis for RQ1 sign tests.

This script is intentionally read-only with respect to experimental artifacts:
it reads already-produced Priority 7 CIFAR-10 gate reports and computes
per-control sign summaries, paired-label permutation sanity checks, and
effective-sample-size tables.  It does not generate targets, train models, or
run attacker optimization.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]

CELLS = {
    "cifar_v2_cosine_confirmatory_n39": {
        "path": ROOT
        / "artifacts/priority7_image_domain/confirmatory_v2_cosine_gate_20260917/priority7_image_gate_report.json",
        "cell": "v2_ratio0p95_eta0p01__GEN_COSINE_TV",
        "stage": "confirmatory",
    },
    "cifar_v2_cosine_development_n24": {
        "path": ROOT
        / "artifacts/priority7_image_domain/n24_scope4_v2_cosine_gate_20260917/priority7_image_gate_report.json",
        "cell": "v2_ratio0p95_eta0p01__GEN_COSINE_TV",
        "stage": "development_n24",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sign_tail(wins: int, total: int) -> float:
    if total <= 0:
        return 1.0
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / 2**total


def summarize_values(values: np.ndarray, permutations: int, rng: np.random.Generator) -> dict:
    non_ties = values[values != 0.0]
    wins = int((non_ties < 0.0).sum())
    losses = int((non_ties > 0.0).sum())
    ties = int(len(values) - len(non_ties))
    exact_p = sign_tail(wins, len(non_ties))

    if len(non_ties):
        # Under the paired-label-flip null, signs are independently flipped.
        # We use absolute counts so the direction observed in the data is the
        # tail being tested; for unanimous wins this is equivalent to 2^-n.
        draws = rng.random((permutations, len(non_ties))) < 0.5
        simulated_wins = draws.sum(axis=1)
        extreme = int((simulated_wins >= wins).sum())
        empirical_p = extreme / permutations
    else:
        extreme = permutations
        empirical_p = 1.0

    return {
        "n_values": int(len(values)),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "non_ties": int(len(non_ties)),
        "mean_difference": float(values.mean()) if len(values) else float("nan"),
        "sd_difference": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
        "median_difference": float(np.median(values)) if len(values) else float("nan"),
        "min_difference": float(values.min()) if len(values) else float("nan"),
        "max_difference": float(values.max()) if len(values) else float("nan"),
        "exact_one_sided_p": exact_p,
        "permutation_extreme_count": extreme,
        "permutation_repetitions": permutations,
        "permutation_empirical_p": empirical_p,
        "effective_n_unanimous_p": {
            str(n_eff): 0.5 ** n_eff for n_eff in range(1, int(len(non_ties)) + 1)
        },
    }


def analyze_report(name: str, spec: dict, permutations: int, rng: np.random.Generator) -> dict:
    report = json.loads(spec["path"].read_text())
    cell = report["cells_summary"][spec["cell"]]
    selected = cell["selected"]
    output = {
        "name": name,
        "stage": spec["stage"],
        "path": str(spec["path"].relative_to(ROOT)),
        "sha256": sha256(spec["path"]),
        "target": report.get("target"),
        "target_sha256": report.get("target_sha256"),
        "cell": spec["cell"],
        "groups": report.get("groups"),
        "restarts": report.get("restarts"),
        "iterations": report.get("iterations"),
        "attack_lr": report.get("attack_lr"),
        "local_lr": report.get("local_lr"),
        "selected_count": len(selected),
        "controls": {},
    }
    for control in ("prior", "zero"):
        values = np.asarray([row[f"baseline_minus_{control}"] for row in selected], dtype=float)
        output["controls"][control] = summarize_values(values, permutations, rng)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    result = {
        "purpose": "Priority 19 CIFAR-10 RQ1 sign-test independence robustness re-analysis",
        "no_new_experimental_data": True,
        "permutation_seed": args.seed,
        "permutations": args.permutations,
        "cells": {
            name: analyze_report(name, spec, args.permutations, rng)
            for name, spec in CELLS.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output)}, indent=2))


if __name__ == "__main__":
    main()
