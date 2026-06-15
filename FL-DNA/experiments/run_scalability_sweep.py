"""Run DNA Transform client-count scalability experiments and summarize results."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results" / "scalability"
SUMMARY_PATH = RESULTS_DIR / "scalability_summary.json"
CLIENT_CONFIGS = [3, 5, 10]
MAX_ROWS = 500_000
MIX_RATIO = 0.08
KEEP_RATIO = 0.88
SHRINK_FACTOR = 0.45
NUM_ROUNDS = 15


def patch_num_clients() -> None:
    common_path = PROJECT_ROOT / "experiments" / "fraud_fl_common.py"
    source = common_path.read_text(encoding="utf-8")
    if "FL_NUM_CLIENTS" in source:
        return

    patched = re.sub(
        r"^NUM_CLIENTS\s*=\s*\d+",
        'NUM_CLIENTS = int(os.environ.get("FL_NUM_CLIENTS", "3"))',
        source,
        flags=re.MULTILINE,
    )
    if patched == source:
        raise RuntimeError("Could not patch NUM_CLIENTS in experiments/fraud_fl_common.py")

    common_path.write_text(patched, encoding="utf-8")
    print("Patched: NUM_CLIENTS now reads from FL_NUM_CLIENTS env var")


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


def result_path(num_clients: int) -> Path:
    return RESULTS_DIR / f"clients_{num_clients}.json"


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def rounds_to_f1(rounds: list[dict[str, object]], target: float = 0.80) -> int | None:
    for round_item in rounds:
        f1_score = round_item.get("f1_score")
        if f1_score is not None and float(f1_score) >= target:
            return int(round_item.get("round", len(rounds)))
    return None


def summarize_result(num_clients: int, path: Path) -> dict[str, object] | None:
    if not path.is_file():
        print(f"WARNING: missing scalability result, skipping summary row: {relative(path)}")
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    rounds = data.get("rounds", [])
    if not rounds:
        print(f"WARNING: result has no rounds, skipping summary row: {relative(path)}")
        return None

    final_round = rounds[-1]
    encode_values = [
        float(round_item["dna_encode_decode_ms"])
        for round_item in rounds
        if round_item.get("dna_encode_decode_ms") is not None
    ]

    return {
        "num_clients": num_clients,
        "final_f1": final_round.get("f1_score"),
        "final_auc_roc": final_round.get("auc_roc"),
        "rounds_to_f1_08": rounds_to_f1(rounds, target=0.80),
        "mean_encode_ms_per_round": mean(encode_values) if encode_values else None,
        "total_encode_ms": sum(encode_values) if encode_values else None,
        "source_file": relative(path),
    }


def write_summary() -> None:
    rows = [
        row
        for row in (summarize_result(num_clients, result_path(num_clients)) for num_clients in CLIENT_CONFIGS)
        if row is not None
    ]
    summary = {
        "experiment": "scalability_num_clients",
        "fixed_params": {
            "mix_ratio": MIX_RATIO,
            "keep_ratio": KEEP_RATIO,
            "shrink_factor": SHRINK_FACTOR,
            "num_rounds": NUM_ROUNDS,
            "max_rows": MAX_ROWS,
        },
        "results": sorted(rows, key=lambda item: int(item["num_clients"])),
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    patch_num_clients()
    succeeded = 0
    failed = 0
    skipped = 0

    for num_clients in CLIENT_CONFIGS:
        output_path = result_path(num_clients)
        description = (
            f"scalability num_clients={num_clients}, max_rows={MAX_ROWS}, "
            f"mix={MIX_RATIO:.2f}, keep={KEEP_RATIO:.2f}, shrink={SHRINK_FACTOR:.2f}"
        )
        print(f"=== Running: {description} ===")

        ok = run_experiment(
            {
                "FL_NUM_CLIENTS": str(num_clients),
                "DNA_TRANSFORM_MIX": f"{MIX_RATIO:.2f}",
                "DNA_TRANSFORM_KEEP": f"{KEEP_RATIO:.2f}",
                "DNA_TRANSFORM_SHRINK": f"{SHRINK_FACTOR:.2f}",
                "NUM_ROUNDS": str(NUM_ROUNDS),
                "MAX_ROWS": str(MAX_ROWS),
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
    print("Scalability Num Clients — DONE")
    print(f"Succeeded: {succeeded}  |  Failed: {failed}  |  Skipped: {skipped}")
    print(f"Output: {relative(SUMMARY_PATH)}")
    print("========================================")


if __name__ == "__main__":
    main()
