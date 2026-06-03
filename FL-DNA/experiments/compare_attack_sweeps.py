"""Create compact comparison tables for gradient inversion attack sweeps."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_ROOT = PROJECT_ROOT / "artifacts" / "gradient_inversion"
OUTPUTS = {
    "strength": ATTACK_ROOT / "strength_sweep" / "summary.csv",
    "sample": ATTACK_ROOT / "sample_sweep" / "summary.csv",
    "round": ATTACK_ROOT / "round_sweep" / "summary.csv",
}


def main() -> None:
    strength_rows = read_csv(OUTPUTS["strength"])
    sample_rows = read_csv(OUTPUTS["sample"])
    round_rows = read_csv(OUTPUTS["round"])

    write_csv(ATTACK_ROOT / "metrics_by_strength.csv", strength_rows)
    write_csv(ATTACK_ROOT / "metrics_by_sample_count.csv", sample_rows)
    write_csv(ATTACK_ROOT / "metrics_by_round.csv", round_rows)
    grouped_round_rows = aggregate_round_groups(round_rows)
    write_csv(ATTACK_ROOT / "metrics_by_round_group.csv", grouped_round_rows)

    summary = {
        "strength_sweep": strength_rows,
        "sample_sweep": sample_rows,
        "round_sweep": round_rows,
        "round_group_summary": grouped_round_rows,
        "notes": {
            "secureagg": (
                "SecureAgg rows marked not_directly_applicable are the true "
                "server-side threat model: individual raw updates are not visible."
            ),
            "psnr_ssim": (
                "PSNR/SSIM are computed on deterministic pseudo-images from "
                "normalized tabular vectors; feature metrics remain primary."
            ),
        },
    }
    (ATTACK_ROOT / "sweep_report.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Saved {relative(ATTACK_ROOT / 'metrics_by_strength.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'metrics_by_sample_count.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'metrics_by_round.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'metrics_by_round_group.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'sweep_report.json')}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sweep summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_round_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = [
        "mean_mse",
        "mean_psnr",
        "mean_ssim",
        "mean_cosine_similarity",
        "mean_pearson_correlation",
        "mean_sign_match_ratio",
    ]
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (
            row.get("round_group", ""),
            row.get("method", ""),
            row.get("attack_applicability", ""),
        )
        groups[key].append(row)

    aggregated = []
    for (round_group, method, applicability), items in groups.items():
        output: dict[str, object] = {
            "round_group": round_group,
            "method": method,
            "attack_applicability": applicability,
            "rounds": ",".join(item.get("warmup_round", "") for item in items),
            "num_round_snapshots": len(items),
            "sample_count": items[0].get("sample_count", ""),
            "threat_model": items[0].get("threat_model", ""),
        }
        for metric in metrics:
            values = [
                float(item[metric])
                for item in items
                if item.get(metric) not in {"", None}
            ]
            if values:
                output[f"group_{metric}"] = float(np.mean(values))
                output[f"group_std_{metric}"] = float(np.std(values))
        aggregated.append(output)
    return aggregated


def relative(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()

