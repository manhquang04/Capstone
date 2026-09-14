"""Post-hoc pooled analysis for the frozen RQ1 conservative batches."""

from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import beta, chi2


ROOT = Path(__file__).resolve().parents[1]
P1_PRACTICAL_REFERENCE = 0.70


def exact_one_sided_sign_p(wins: int, n: int, p0: float = 0.5) -> float:
    return sum(math.comb(n, k) * (p0 ** k) * ((1 - p0) ** (n - k)) for k in range(wins, n + 1))


def clopper_pearson_ci(wins: int, n: int, alpha: float = 0.05) -> list[float]:
    lower = 0.0 if wins == 0 else float(beta.ppf(alpha / 2, wins, n - wins + 1))
    upper = 1.0 if wins == n else float(beta.ppf(1 - alpha / 2, wins + 1, n - wins))
    return [lower, upper]


def chi_square_heterogeneity(rows: list[dict]) -> dict:
    total_wins = sum(row["wins"] for row in rows)
    total_losses = sum(row["losses"] for row in rows)
    total = total_wins + total_losses
    q = 0.0
    expected_min = float("inf")
    for row in rows:
        row_total = row["wins"] + row["losses"]
        expected_wins = row_total * total_wins / total
        expected_losses = row_total * total_losses / total
        expected_min = min(expected_min, expected_wins, expected_losses)
        q += ((row["wins"] - expected_wins) ** 2) / expected_wins
        q += ((row["losses"] - expected_losses) ** 2) / expected_losses
    df = len(rows) - 1
    return {
        "test": "chi_square_4x2_wins_losses",
        "statistic": q,
        "df": df,
        "p_value": float(chi2.sf(q, df)),
        "minimum_expected_cell_count": expected_min,
        "descriptive_if_expected_lt_5": expected_min < 5,
    }


def load_batch(path: Path, label: str) -> dict:
    summary = json.loads(path.read_text(encoding="utf-8"))
    primary = summary["primary"]
    return {
        "label": label,
        "path": str(path.relative_to(ROOT)),
        "branch_gates_pass": all(summary["branch_gates"][branch]["pass"] for branch in ("raw", "dna", "dp")),
        "wins": int(primary["wins"]),
        "losses": int(primary["losses"]),
        "ties": int(primary["ties"]),
        "non_tied_n": int(primary["non_tied_n"]),
        "one_sided_exact_sign_p": float(primary["one_sided_exact_sign_p"]),
        "reject_h0": bool(primary["reject_h0"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    batches = [
        load_batch(ROOT / "results/rq1/confirmatory_20260913/rq1_summary.json", "original_group2_conservative"),
        load_batch(ROOT / "results/rq1/group3_replication_1_20260913/summary.json", "group3_replication_1"),
        load_batch(ROOT / "results/rq1/group3_replication_2_20260913/summary.json", "group3_replication_2"),
        load_batch(ROOT / "results/rq1/group3_replication_3_20260914/summary.json", "group3_replication_3"),
    ]
    wins = sum(batch["wins"] for batch in batches)
    losses = sum(batch["losses"] for batch in batches)
    ties = sum(batch["ties"] for batch in batches)
    non_tied_n = wins + losses
    win_probability = wins / non_tied_n

    report = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "EXPLORATORY_POST_HOC_POOLED_ANALYSIS",
        "amendment": "protocols/amendments/2026-09-14_group3_replication_3_and_pooled_analysis.md",
        "interpretation_contract": (
            "Does not replace or reverse any individual confirmatory decision; "
            "used only as stability/sensitivity evidence."
        ),
        "batches": batches,
        "pooled": {
            "wins": wins,
            "losses": losses,
            "ties": ties,
            "non_tied_n": non_tied_n,
            "win_probability": win_probability,
            "win_probability_exact_ci95": clopper_pearson_ci(wins, non_tied_n),
            "one_sided_exact_sign_p_p0_0p50": exact_one_sided_sign_p(wins, non_tied_n, 0.5),
            "statistically_above_0p50_alpha_0p05": exact_one_sided_sign_p(wins, non_tied_n, 0.5) < 0.05,
            "practical_effect_reference_p1": P1_PRACTICAL_REFERENCE,
            "meets_practical_effect_reference": win_probability >= P1_PRACTICAL_REFERENCE,
        },
        "heterogeneity": chi_square_heterogeneity(batches),
        "final_draw_rule": "Replication 3 is the final additional draw for this RQ1 conservative comparison.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
