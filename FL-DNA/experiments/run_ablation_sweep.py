"""Run DNA Transform mix-ratio ablation experiments and summarize results."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "ablation"
SUMMARY_PATH = RESULTS_DIR / "ablation_summary.json"
FIXED_KEEP = 0.88
FIXED_SHRINK = 0.45
NUM_ROUNDS = 15
NUM_CLIENTS = 3

MIX_CONFIGS = [
    0.01,
    0.03,
    0.05,
    0.07,
    0.08,
    0.10,
    0.12,
    0.15,
    0.18,
    0.20,
]


def run_experiment(env_overrides: dict[str, object], output_path: str, project_root: Path) -> bool:
    env = os.environ.copy()
    env.update({key: str(value) for key, value in env_overrides.items()})
    env["DNA_TRANSFORM_OUTPUT_PATH"] = output_path
    result = subprocess.run(
        [sys.executable, "experiments/run_fraud_fl_dna_transform.py"],
        env=env,
        cwd=str(project_root),
    )
    return result.returncode == 0


def result_path(mix_ratio: float) -> Path:
    return RESULTS_DIR / f"mix_{mix_ratio:.2f}.json"


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def summarize_result(mix_ratio: float, path: Path) -> dict[str, object] | None:
    if not path.is_file():
        print(f"WARNING: missing ablation result, skipping summary row: {relative(path)}")
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    rounds = data.get("rounds", [])
    if not rounds:
        print(f"WARNING: result has no rounds, skipping summary row: {relative(path)}")
        return None

    final_round = rounds[-1]
    cosine_values = [
        float(round_item["dna_transform_cosine_similarity"])
        for round_item in rounds
        if round_item.get("dna_transform_cosine_similarity") is not None
    ]
    encode_values = [
        float(round_item["dna_encode_decode_ms"])
        for round_item in rounds
        if round_item.get("dna_encode_decode_ms") is not None
    ]

    return {
        "mix_ratio": mix_ratio,
        "final_f1": final_round.get("f1_score"),
        "final_auc_roc": final_round.get("auc_roc"),
        "final_pr_auc": final_round.get("pr_auc"),
        "mean_cosine_similarity": mean(cosine_values) if cosine_values else None,
        "mean_encode_ms": mean(encode_values) if encode_values else None,
        "source_file": relative(path),
    }


def write_summary() -> None:
    rows = [
        row
        for row in (summarize_result(mix_ratio, result_path(mix_ratio)) for mix_ratio in MIX_CONFIGS)
        if row is not None
    ]
    summary = {
        "experiment": "ablation_mix_ratio",
        "fixed_params": {
            "keep_ratio": FIXED_KEEP,
            "shrink_factor": FIXED_SHRINK,
            "num_rounds": NUM_ROUNDS,
            "num_clients": NUM_CLIENTS,
        },
        "results": sorted(rows, key=lambda item: float(item["mix_ratio"])),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    succeeded = 0
    failed = 0
    skipped = 0

    for mix_ratio in MIX_CONFIGS:
        output_path = result_path(mix_ratio)
        description = (
            f"ablation mix_ratio={mix_ratio:.2f}, "
            f"keep={FIXED_KEEP:.2f}, shrink={FIXED_SHRINK:.2f}"
        )
        print(f"=== Running: {description} ===")

        ok = run_experiment(
            {
                "DNA_TRANSFORM_MIX": f"{mix_ratio:.2f}",
                "DNA_TRANSFORM_KEEP": f"{FIXED_KEEP:.2f}",
                "DNA_TRANSFORM_SHRINK": f"{FIXED_SHRINK:.2f}",
                "NUM_ROUNDS": str(NUM_ROUNDS),
                "FL_NUM_CLIENTS": str(NUM_CLIENTS),
            },
            str(output_path),
            PROJECT_ROOT,
        )
        if ok:
            succeeded += 1
        else:
            print(f"ERROR: failed run for {description}")
            failed += 1

    write_summary()
    print("========================================")
    print("Ablation Mix Ratio — DONE")
    print(f"Succeeded: {succeeded}  |  Failed: {failed}  |  Skipped: {skipped}")
    print(f"Output: {relative(SUMMARY_PATH)}")
    print("========================================")


if __name__ == "__main__":
    main()
