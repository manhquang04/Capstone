"""Create compact comparison tables for gradient inversion attack sweeps."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ATTACK_ROOT = PROJECT_ROOT / "artifacts" / "gradient_inversion"
OFFICIAL_UTILITY_ROUNDS = 50
OUTPUTS = {
    "strength": ATTACK_ROOT / "strength_sweep" / "summary.csv",
    "sample": ATTACK_ROOT / "sample_sweep" / "summary.csv",
    "round": ATTACK_ROOT / "round_sweep" / "summary.csv",
    "dp": ATTACK_ROOT / "dp_noise_sweep" / "summary.csv",
}
RESULTS_DIR = PROJECT_ROOT / "results" / "fraud"
TRANSFORM_UTILITY = {
    "current": RESULTS_DIR / "dna_transform_metrics.json",
    "conservative": RESULTS_DIR / "dna_transform_conservative_metrics.json",
    "medium": RESULTS_DIR / "dna_transform_medium_metrics.json",
    "stronger": RESULTS_DIR / "dna_transform_stronger_metrics.json",
}
DP_UTILITY = {
    "utility": RESULTS_DIR / "dp_utility_metrics.json",
    "weak": RESULTS_DIR / "dp_weak_metrics.json",
    "mild": RESULTS_DIR / "dp_mild_metrics.json",
    "medium": RESULTS_DIR / "dp_metrics.json",
    "strong": RESULTS_DIR / "dp_strong_metrics.json",
}


def main() -> None:
    strength_rows = normalize_metric_aliases(read_csv(OUTPUTS["strength"]))
    sample_rows = normalize_metric_aliases(read_csv(OUTPUTS["sample"]))
    round_rows = normalize_metric_aliases(read_csv(OUTPUTS["round"]))
    dp_rows = normalize_metric_aliases(read_csv(OUTPUTS["dp"])) if OUTPUTS["dp"].is_file() else []

    write_csv(ATTACK_ROOT / "metrics_by_strength.csv", strength_rows)
    write_csv(ATTACK_ROOT / "metrics_by_sample_count.csv", sample_rows)
    write_csv(ATTACK_ROOT / "metrics_by_round.csv", round_rows)
    if dp_rows:
        write_csv(ATTACK_ROOT / "metrics_by_dp_noise.csv", dp_rows)
    grouped_round_rows = aggregate_round_groups(round_rows)
    write_csv(ATTACK_ROOT / "metrics_by_round_group.csv", grouped_round_rows)
    tradeoff_rows = build_privacy_utility_tradeoff(strength_rows, dp_rows)
    write_csv(ATTACK_ROOT / "privacy_utility_tradeoff.csv", tradeoff_rows)

    summary = {
        "strength_sweep": strength_rows,
        "sample_sweep": sample_rows,
        "round_sweep": round_rows,
        "dp_noise_sweep": dp_rows,
        "round_group_summary": grouped_round_rows,
        "privacy_utility_tradeoff": tradeoff_rows,
        "notes": {
            "secureagg": (
                "SecureAgg rows marked not_directly_applicable are the true "
                "server-side threat model: individual raw updates are not visible."
            ),
            "psnr_ssim": (
                "PSNR/SSIM are computed on deterministic pseudo-images from "
                "normalized tabular vectors; feature metrics remain primary."
            ),
            "dp_accounting": (
                "No privacy accountant is implemented; DP rows are clipping/noise "
                "defenses and must not be reported as formal epsilon-delta DP."
            ),
            "official_utility_rounds": OFFICIAL_UTILITY_ROUNDS,
            "metric_priority": (
                "For tabular PaySim, feature MSE, cosine, Pearson, and sign-match "
                "are primary reconstruction metrics. PSNR/SSIM are reported on "
                "pseudo-images for visual reconstruction context."
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
    if dp_rows:
        print(f"Saved {relative(ATTACK_ROOT / 'metrics_by_dp_noise.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'privacy_utility_tradeoff.csv')}")
    print(f"Saved {relative(ATTACK_ROOT / 'sweep_report.json')}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing sweep summary: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_metric_aliases(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    for row in rows:
        row.setdefault("mean_feature_mse", row.get("mean_mse", ""))
        row.setdefault("std_feature_mse", row.get("std_mse", ""))
    return rows


def load_final_metric(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    rounds = payload.get("rounds", [])
    if not rounds:
        return None
    utility_rounds = payload.get("config", {}).get("num_rounds")
    if utility_rounds != OFFICIAL_UTILITY_ROUNDS:
        return {
            "utility_path": str(path.relative_to(PROJECT_ROOT)),
            "utility_rounds": utility_rounds,
            "utility_round_mismatch": True,
        }
    final = rounds[-1]
    return {
        "utility_path": str(path.relative_to(PROJECT_ROOT)),
        "utility_rounds": utility_rounds,
        "utility_round_mismatch": False,
        "final_f1_score": final.get("f1_score"),
        "final_auc_roc": final.get("auc_roc"),
        "final_pr_auc": final.get("pr_auc"),
        "final_precision": final.get("precision"),
        "final_recall": final.get("recall"),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def build_privacy_utility_tradeoff(
    strength_rows: list[dict[str, str]],
    dp_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in strength_rows:
        if row.get("method") != "FL_DNA_TransformDefense":
            continue
        if row.get("attack_applicability") != "direct_server_side":
            continue
        strength = row.get("strength", "")
        utility = load_final_metric(TRANSFORM_UTILITY.get(strength, Path()))
        rows.append(
            {
                "defense_family": "dna_transform",
                "variant": strength,
                "attack_samples": row.get("sample_count"),
                "attack_round": row.get("warmup_round"),
                "mean_mse": row.get("mean_mse"),
                "std_mse": row.get("std_mse"),
                "mean_feature_mse": row.get("mean_feature_mse", row.get("mean_mse")),
                "std_feature_mse": row.get("std_feature_mse", row.get("std_mse")),
                "mean_cosine_similarity": row.get("mean_cosine_similarity"),
                "std_cosine_similarity": row.get("std_cosine_similarity"),
                "mean_pearson_correlation": row.get("mean_pearson_correlation"),
                "std_pearson_correlation": row.get("std_pearson_correlation"),
                "mean_sign_match_ratio": row.get("mean_sign_match_ratio"),
                "std_sign_match_ratio": row.get("std_sign_match_ratio"),
                "secondary_mean_psnr": row.get("mean_psnr"),
                "secondary_std_psnr": row.get("std_psnr"),
                "secondary_mean_ssim": row.get("mean_ssim"),
                "secondary_std_ssim": row.get("std_ssim"),
                "limitation": limitation_for_utility(utility),
                **(utility or {}),
            }
        )

    for row in dp_rows:
        if row.get("method") != "FL_DP":
            continue
        if row.get("attack_applicability") != "direct_server_side":
            continue
        preset = row.get("dp_noise_preset", "")
        utility = load_final_metric(DP_UTILITY.get(preset, Path()))
        rows.append(
            {
                "defense_family": "dp_noise",
                "variant": preset,
                "dp_noise_multiplier": row.get("dp_noise_multiplier"),
                "attack_samples": row.get("sample_count"),
                "attack_round": row.get("warmup_round"),
                "mean_mse": row.get("mean_mse"),
                "std_mse": row.get("std_mse"),
                "mean_feature_mse": row.get("mean_feature_mse", row.get("mean_mse")),
                "std_feature_mse": row.get("std_feature_mse", row.get("std_mse")),
                "mean_cosine_similarity": row.get("mean_cosine_similarity"),
                "std_cosine_similarity": row.get("std_cosine_similarity"),
                "mean_pearson_correlation": row.get("mean_pearson_correlation"),
                "std_pearson_correlation": row.get("std_pearson_correlation"),
                "mean_sign_match_ratio": row.get("mean_sign_match_ratio"),
                "std_sign_match_ratio": row.get("std_sign_match_ratio"),
                "secondary_mean_psnr": row.get("mean_psnr"),
                "secondary_std_psnr": row.get("std_psnr"),
                "secondary_mean_ssim": row.get("mean_ssim"),
                "secondary_std_ssim": row.get("std_ssim"),
                "limitation": (
                    limitation_for_utility(
                        utility,
                        missing="Missing matching utility artifact for this DP noise preset.",
                    )
                ),
                **(utility or {}),
            }
        )
    return rows


def limitation_for_utility(
    utility: dict[str, object] | None,
    missing: str = "Missing matching utility artifact.",
) -> str:
    if not utility:
        return missing
    if utility.get("utility_round_mismatch"):
        return (
            f"Utility artifact exists but uses {utility.get('utility_rounds')} rounds; "
            f"official utility round count is {OFFICIAL_UTILITY_ROUNDS}."
        )
    return ""


def aggregate_round_groups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = [
        "mean_mse",
        "mean_feature_mse",
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
