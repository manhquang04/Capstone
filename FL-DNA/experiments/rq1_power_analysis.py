"""Exact-binomial sample-size analysis for the frozen RQ1 sign test.

This is a design utility.  It does not read attack outcomes or target data.
The default arguments reproduce the supervisor-approved planning calculation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any


def binomial_upper_tail(*, n: int, minimum_wins: int, probability: float) -> float:
    """Return P[X >= minimum_wins] for X ~ Binomial(n, probability)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if minimum_wins <= 0:
        return 1.0
    if minimum_wins > n:
        return 0.0
    return math.fsum(
        math.comb(n, wins)
        * probability**wins
        * (1.0 - probability) ** (n - wins)
        for wins in range(minimum_wins, n + 1)
    )


def rejection_threshold(*, n: int, p0: float, alpha: float) -> int | None:
    """Return the smallest win count whose exact upper tail is <= alpha."""
    for minimum_wins in range(0, n + 1):
        if binomial_upper_tail(
            n=n, minimum_wins=minimum_wins, probability=p0
        ) <= alpha:
            return minimum_wins
    return None


def required_non_tied_n(
    *, p0: float, p1: float, alpha: float, target_power: float, maximum_n: int
) -> dict[str, float | int]:
    """Find the smallest exact-binomial n attaining the requested power."""
    if not 0.0 <= p0 < p1 <= 1.0:
        raise ValueError("require 0 <= p0 < p1 <= 1")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target_power must be in (0, 1)")
    if maximum_n < 1:
        raise ValueError("maximum_n must be positive")

    for n in range(1, maximum_n + 1):
        threshold = rejection_threshold(n=n, p0=p0, alpha=alpha)
        if threshold is None:
            continue
        actual_power = binomial_upper_tail(
            n=n, minimum_wins=threshold, probability=p1
        )
        if actual_power >= target_power:
            return {
                "effective_non_tied_n": n,
                "rejection_minimum_wins": threshold,
                "actual_alpha": binomial_upper_tail(
                    n=n, minimum_wins=threshold, probability=p0
                ),
                "actual_power": actual_power,
            }
    raise RuntimeError(f"required n exceeds maximum_n={maximum_n}")


def inflated_draw_count(
    *, effective_non_tied_n: int, tie_rate: float, dropout_rate: float
) -> int:
    """Inflate effective n for independently predeclared ties and dropouts."""
    if not 0.0 <= tie_rate < 1.0 or not 0.0 <= dropout_rate < 1.0:
        raise ValueError("tie_rate and dropout_rate must be in [0, 1)")
    usable_fraction = (1.0 - tie_rate) * (1.0 - dropout_rate)
    return math.ceil(effective_non_tied_n / usable_fraction)


def sensitivity_rows(
    *,
    effective_non_tied_n: int,
    tie_rates: list[float],
    dropout_rates: list[float],
) -> list[dict[str, float | int]]:
    return [
        {
            "tie_rate": tie_rate,
            "dropout_rate": dropout_rate,
            "effective_non_tied_n": effective_non_tied_n,
            "initial_draw_count": inflated_draw_count(
                effective_non_tied_n=effective_non_tied_n,
                tie_rate=tie_rate,
                dropout_rate=dropout_rate,
            ),
        }
        for tie_rate in tie_rates
        for dropout_rate in dropout_rates
    ]


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    primary = required_non_tied_n(
        p0=args.p0,
        p1=args.p1,
        alpha=args.alpha,
        target_power=args.power,
        maximum_n=args.maximum_n,
    )
    effective_n = int(primary["effective_non_tied_n"])
    primary["initial_draw_count"] = inflated_draw_count(
        effective_non_tied_n=effective_n,
        tie_rate=args.tie_rate,
        dropout_rate=args.dropout_rate,
    )
    return {
        "protocol_id": "RQ1-CONFIRMATORY-V1",
        "calculation": "exact_one_sided_binomial_sign_test",
        "inputs": {
            "p0": args.p0,
            "p1": args.p1,
            "alpha": args.alpha,
            "target_power": args.power,
            "tie_rate": args.tie_rate,
            "dropout_rate": args.dropout_rate,
            "maximum_n": args.maximum_n,
        },
        "primary": primary,
        "sensitivity": sensitivity_rows(
            effective_non_tied_n=effective_n,
            tie_rates=args.sensitivity_tie_rates,
            dropout_rates=args.sensitivity_dropout_rates,
        ),
    }


def write_outputs(report: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "rq1_power_analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    rows = report["sensitivity"]
    with (output_dir / "rq1_power_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p0", type=float, default=0.50)
    parser.add_argument("--p1", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--power", type=float, default=0.80)
    parser.add_argument("--tie-rate", type=float, default=0.10)
    parser.add_argument("--dropout-rate", type=float, default=0.05)
    parser.add_argument("--maximum-n", type=int, default=150)
    parser.add_argument(
        "--sensitivity-tie-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10, 0.20],
    )
    parser.add_argument(
        "--sensitivity-dropout-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.05, 0.10],
    )
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_report(args)
    if args.output_dir is not None:
        write_outputs(report, args.output_dir)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
