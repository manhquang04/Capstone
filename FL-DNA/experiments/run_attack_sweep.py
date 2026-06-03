"""Run gradient inversion attack sweeps for DNA Transform evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SWEEP_ROOT = PROJECT_ROOT / "artifacts" / "gradient_inversion"
ATTACK_RUNNER = PROJECT_ROOT / "attacks" / "attack_runner.py"

DNA_STRENGTHS = {
    "current": {"mix": "0.05", "keep": "0.90", "shrink": "0.50"},
    "conservative": {"mix": "0.08", "keep": "0.88", "shrink": "0.45"},
    "medium": {"mix": "0.10", "keep": "0.85", "shrink": "0.40"},
    "stronger": {"mix": "0.12", "keep": "0.82", "shrink": "0.35"},
}
ROUND_GROUPS = {
    "early": [1, 2, 3],
    "mid": [7, 8],
    "late": [13, 14, 15],
}
SAMPLE_COUNTS = [10, 20, 30]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["all", "strength", "sample", "round"],
        default="all",
    )
    parser.add_argument("--iterations", type=int, default=int(os.environ.get("ATTACK_ITERATIONS", "300")))
    parser.add_argument("--max-rows", type=int, default=int(os.environ.get("MAX_ROWS", "500000")))
    parser.add_argument("--strength-samples", type=int, default=int(os.environ.get("STRENGTH_SWEEP_SAMPLES", "10")))
    parser.add_argument("--round-samples", type=int, default=int(os.environ.get("ROUND_SWEEP_SAMPLES", "10")))
    args = parser.parse_args()

    started = perf_counter()
    summaries: list[dict[str, object]] = []
    if args.mode in {"all", "strength"}:
        summaries.extend(run_strength_sweep(args))
        write_combined_summary("strength_sweep", summaries)
    if args.mode in {"all", "sample"}:
        sample_rows = run_sample_sweep(args)
        summaries.extend(sample_rows)
        write_combined_summary("sample_sweep", sample_rows)
    if args.mode in {"all", "round"}:
        round_rows = run_round_sweep(args)
        summaries.extend(round_rows)
        write_combined_summary("round_sweep", round_rows)

    write_combined_summary("combined_sweeps", summaries)
    print(f"Total sweep runtime: {perf_counter() - started:.2f} seconds")
    return 0


def run_strength_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    for strength_name, config in DNA_STRENGTHS.items():
        output_dir = SWEEP_ROOT / "strength_sweep" / strength_name
        run_attack(
            output_dir=output_dir,
            iterations=args.iterations,
            num_samples=args.strength_samples,
            warmup_rounds=3,
            max_rows=args.max_rows,
            dna_config=config,
        )
        rows.extend(
            load_summary_rows(
                output_dir,
                {
                    "sweep": "strength",
                    "strength": strength_name,
                    "sample_count": args.strength_samples,
                    "round_group": "warmup_3",
                    "warmup_round": 3,
                    **{f"dna_{key}": value for key, value in config.items()},
                },
            )
        )
    return rows


def run_sample_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    config = DNA_STRENGTHS["current"]
    for sample_count in SAMPLE_COUNTS:
        output_dir = SWEEP_ROOT / "sample_sweep" / f"samples_{sample_count:02d}"
        run_attack(
            output_dir=output_dir,
            iterations=args.iterations,
            num_samples=sample_count,
            warmup_rounds=3,
            max_rows=args.max_rows,
            dna_config=config,
        )
        rows.extend(
            load_summary_rows(
                output_dir,
                {
                    "sweep": "sample",
                    "strength": "current",
                    "sample_count": sample_count,
                    "round_group": "warmup_3",
                    "warmup_round": 3,
                    **{f"dna_{key}": value for key, value in config.items()},
                },
            )
        )
    return rows


def run_round_sweep(args: argparse.Namespace) -> list[dict[str, object]]:
    rows = []
    config = DNA_STRENGTHS["current"]
    for group_name, rounds in ROUND_GROUPS.items():
        for round_number in rounds:
            output_dir = SWEEP_ROOT / "round_sweep" / group_name / f"round_{round_number:02d}"
            run_attack(
                output_dir=output_dir,
                iterations=args.iterations,
                num_samples=args.round_samples,
                warmup_rounds=round_number,
                max_rows=args.max_rows,
                dna_config=config,
            )
            rows.extend(
                load_summary_rows(
                    output_dir,
                    {
                        "sweep": "round",
                        "strength": "current",
                        "sample_count": args.round_samples,
                        "round_group": group_name,
                        "warmup_round": round_number,
                        **{f"dna_{key}": value for key, value in config.items()},
                    },
                )
            )
    return rows


def run_attack(
    output_dir: Path,
    iterations: int,
    num_samples: int,
    warmup_rounds: int,
    max_rows: int,
    dna_config: dict[str, str],
) -> None:
    env = os.environ.copy()
    env.update(
        {
            "ATTACK_OUTPUT_DIR": str(output_dir),
            "ATTACK_ITERATIONS": str(iterations),
            "ATTACK_NUM_SAMPLES": str(num_samples),
            "ATTACK_WARMUP_ROUNDS": str(warmup_rounds),
            "MAX_ROWS": str(max_rows),
            "DNA_TRANSFORM_MIX": dna_config["mix"],
            "DNA_TRANSFORM_KEEP": dna_config["keep"],
            "DNA_TRANSFORM_SHRINK": dna_config["shrink"],
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(SWEEP_ROOT / ".mplconfig"),
        }
    )
    print(
        "RUN",
        f"output={output_dir.relative_to(PROJECT_ROOT)}",
        f"samples={num_samples}",
        f"rounds={warmup_rounds}",
        f"dna={dna_config}",
        flush=True,
    )
    started = perf_counter()
    subprocess.run(
        [sys.executable, "-u", str(ATTACK_RUNNER)],
        cwd=PROJECT_ROOT,
        env=env,
        check=True,
    )
    print(f"COMPLETED runtime={perf_counter() - started:.2f}s", flush=True)


def load_summary_rows(output_dir: Path, extra: dict[str, object]) -> list[dict[str, object]]:
    summary_path = output_dir / "metrics_summary.json"
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return [{**extra, **row} for row in rows]


def write_combined_summary(name: str, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    output_dir = SWEEP_ROOT / name
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "summary.json"
    csv_path = output_dir / "summary.csv"
    json_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved {csv_path.relative_to(PROJECT_ROOT)}", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
