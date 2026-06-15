"""Plot the empirical privacy-utility Pareto front.

The primary input is artifacts/gradient_inversion/privacy_utility_tradeoff.csv.
That file contains DNA Transform variants and DP presets. If FL Baseline or
lossless FL + DNA are not present there, this script enriches the plot from
combined_sweeps/summary.json plus the existing utility JSON files.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRADEOFF_CSV = PROJECT_ROOT / "artifacts" / "gradient_inversion" / "privacy_utility_tradeoff.csv"
COMBINED_SWEEPS_JSON = PROJECT_ROOT / "artifacts" / "gradient_inversion" / "combined_sweeps" / "summary.json"
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "gradient_inversion"
OUTPUT_PNG = OUTPUT_DIR / "privacy_utility_pareto_front.png"
OUTPUT_PDF = OUTPUT_DIR / "privacy_utility_pareto_front.pdf"
OUTPUT_CSV = OUTPUT_DIR / "privacy_utility_pareto_front_points.csv"


@dataclass(frozen=True)
class Point:
    method: str
    family: str
    variant: str
    f1: float
    privacy_mse: float
    roc_auc: float | None
    pr_auc: float | None
    source: str


def _float_or_none(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _final_metric(path: Path, key: str) -> float | None:
    if not path.is_file():
        return None
    data = _load_json(path)
    rounds = data.get("rounds", [])
    if not rounds:
        return None
    return _float_or_none(rounds[-1].get(key))


def _display_name(family: str, variant: str) -> str:
    if family == "dna_transform":
        if variant == "current":
            return "DNA Transform"
        return f"DNA Transform {variant}"
    if family == "dp_noise":
        return f"DP {variant}"
    return f"{family} {variant}".strip()


def load_tradeoff_points() -> list[Point]:
    if not TRADEOFF_CSV.is_file():
        raise FileNotFoundError(f"Tradeoff CSV not found: {TRADEOFF_CSV}")

    points: list[Point] = []
    with TRADEOFF_CSV.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            f1 = _float_or_none(row.get("final_f1_score"))
            privacy_mse = _float_or_none(row.get("mean_mse"))
            if f1 is None or privacy_mse is None:
                limitation = row.get("limitation") or "missing utility or privacy metric"
                print(f"WARNING: skipping {row.get('defense_family')} {row.get('variant')}: {limitation}")
                continue

            family = str(row.get("defense_family") or "unknown")
            variant = str(row.get("variant") or "")
            points.append(
                Point(
                    method=_display_name(family, variant),
                    family=family,
                    variant=variant,
                    f1=f1,
                    privacy_mse=privacy_mse,
                    roc_auc=_float_or_none(row.get("final_auc_roc")),
                    pr_auc=_float_or_none(row.get("final_pr_auc")),
                    source=str(TRADEOFF_CSV.relative_to(PROJECT_ROOT)),
                )
            )
    return points


def _combined_attack_row(method: str) -> dict[str, Any] | None:
    if not COMBINED_SWEEPS_JSON.is_file():
        print(f"WARNING: cannot enrich baseline/DNA, missing {COMBINED_SWEEPS_JSON}")
        return None
    rows = _load_json(COMBINED_SWEEPS_JSON)
    if not isinstance(rows, list):
        return None

    candidates = [
        row
        for row in rows
        if row.get("method") == method
        and row.get("sweep") == "strength"
        and row.get("strength") == "current"
        and row.get("round_group") == "warmup_3"
    ]
    if not candidates:
        candidates = [row for row in rows if row.get("method") == method and row.get("mean_mse") is not None]
    return candidates[0] if candidates else None


def enrich_reference_points(points: list[Point]) -> list[Point]:
    existing_names = {point.method for point in points}
    enrich_specs = [
        ("FL Baseline", "baseline", "reference", "FL_Baseline", PROJECT_ROOT / "results" / "fraud" / "baseline_metrics.json"),
        ("FL + DNA", "dna_lossless", "reference", "FL_DNA", PROJECT_ROOT / "results" / "fraud" / "dna_metrics.json"),
    ]

    enriched = list(points)
    for name, family, variant, attack_method, utility_path in enrich_specs:
        if name in existing_names:
            continue
        attack_row = _combined_attack_row(attack_method)
        privacy_mse = _float_or_none(attack_row.get("mean_mse")) if attack_row else None
        f1 = _final_metric(utility_path, "f1_score")
        if privacy_mse is None or f1 is None:
            print(f"WARNING: cannot add {name}: missing attack MSE or utility F1")
            continue
        enriched.append(
            Point(
                method=name,
                family=family,
                variant=variant,
                f1=f1,
                privacy_mse=privacy_mse,
                roc_auc=_final_metric(utility_path, "auc_roc"),
                pr_auc=_final_metric(utility_path, "pr_auc"),
                source=f"{COMBINED_SWEEPS_JSON.relative_to(PROJECT_ROOT)} + {utility_path.relative_to(PROJECT_ROOT)}",
            )
        )
    return enriched


def write_points_csv(points: list[Point]) -> None:
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["method", "family", "variant", "f1", "privacy_mse", "roc_auc", "pr_auc", "source"],
            lineterminator="\n",
        )
        writer.writeheader()
        for point in points:
            writer.writerow(
                {
                    "method": point.method,
                    "family": point.family,
                    "variant": point.variant,
                    "f1": point.f1,
                    "privacy_mse": point.privacy_mse,
                    "roc_auc": point.roc_auc,
                    "pr_auc": point.pr_auc,
                    "source": point.source,
                }
            )


def plot(points: list[Point]) -> None:
    import matplotlib.pyplot as plt

    colors = {
        "dna_transform": "#1f77b4",
        "dp_noise": "#d62728",
        "baseline": "#7f7f7f",
        "dna_lossless": "#2ca02c",
    }
    markers = {
        "dna_transform": "o",
        "dp_noise": "s",
        "baseline": "D",
        "dna_lossless": "^",
    }
    label_map = {
        "dna_transform": "DNA Transform variants",
        "dp_noise": "Differential Privacy",
        "baseline": "FL Baseline",
        "dna_lossless": "FL + DNA lossless",
    }
    short_labels = {
        "FL Baseline": "Baseline",
        "FL + DNA": "FL + DNA",
        "DNA Transform": "DNA current",
        "DNA Transform conservative": "DNA conservative",
        "DNA Transform medium": "DNA medium",
        "DNA Transform stronger": "DNA stronger",
        "DP utility": "DP utility",
        "DP mild": "DP mild",
        "DP medium": "DP medium",
    }
    zoom_offsets = {
        "Baseline": (-58, 22),
        "FL + DNA": (-56, -20),
        "DNA current": (10, -18),
        "DNA conservative": (12, 12),
        "DNA medium": (12, -34),
        "DNA stronger": (12, 30),
        "DP utility": (-74, -28),
    }

    fig, (ax, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(14.6, 6.2),
        gridspec_kw={"width_ratios": [1.05, 1.2]},
    )

    def scatter_points(axis: Any) -> None:
        for point in points:
            axis.scatter(
                point.privacy_mse,
                point.f1,
                s=115,
                c=colors.get(point.family, "#9467bd"),
                marker=markers.get(point.family, "o"),
                edgecolors="black",
                linewidths=0.8,
                alpha=0.92,
                label=point.family,
                zorder=3,
            )

    scatter_points(ax)
    scatter_points(ax_zoom)

    # Keep the full-range panel readable by labeling only the low-utility DP
    # points and directing readers to the zoomed high-utility cluster.
    for point in points:
        short = short_labels.get(point.method, point.method)
        if point.method in {"DP medium", "DP mild"}:
            ax.annotate(
                short,
                (point.privacy_mse, point.f1),
                textcoords="offset points",
                xytext=(8, -14 if point.method == "DP medium" else -18),
                fontsize=8.6,
            )
    ax.annotate(
        "High-utility methods\nshown in detail on right",
        xy=(6305.8, 0.71),
        xytext=(6303.15, 0.57),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#555555"},
        fontsize=9,
        color="#333333",
    )

    cluster_points = [point for point in points if point.f1 >= 0.65 and point.privacy_mse >= 6305.45]
    for point in cluster_points:
        short = short_labels.get(point.method, point.method)
        ax_zoom.annotate(
            short,
            (point.privacy_mse, point.f1),
            textcoords="offset points",
            xytext=zoom_offsets.get(short, (8, 8)),
            fontsize=8.8,
            arrowprops={"arrowstyle": "-", "lw": 0.7, "color": "#777777"},
        )

    conservative = next((point for point in points if point.method == "DNA Transform conservative"), None)
    if conservative is not None:
        for axis in (ax, ax_zoom):
            axis.scatter(
                conservative.privacy_mse,
                conservative.f1,
                s=280,
                facecolors="none",
                edgecolors="#ffbf00",
                linewidths=2.5,
                marker="o",
                zorder=5,
            )
        ax_zoom.annotate(
            "recommended\ntrade-off",
            (conservative.privacy_mse, conservative.f1),
            xytext=(6305.70, 0.733),
            textcoords="data",
            arrowprops={"arrowstyle": "->", "color": "#7a5c00", "lw": 1.4},
            fontsize=9.5,
            color="#7a5c00",
            weight="bold",
            ha="center",
        )

    # Deduplicate legend labels.
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    deduped = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
    fig.legend(
        [item[0] for item in deduped],
        [label_map.get(item[1], item[1]) for item in deduped],
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, -0.01),
    )

    fig.suptitle("Privacy-Utility Pareto Front from Gradient Inversion Evaluation", fontsize=14, weight="bold")
    ax.set_title("Full comparison", fontsize=11, weight="bold")
    ax_zoom.set_title("Zoom: high-utility / high-privacy region", fontsize=11, weight="bold")
    ax.set_xlabel("Privacy proxy: reconstruction MSE\n(higher = harder inversion)")
    ax_zoom.set_xlabel("Privacy proxy: reconstruction MSE\n(higher = harder inversion)")
    ax.set_ylabel("Utility: final F1-score (higher = better fraud detection)")

    for axis in (ax, ax_zoom):
        axis.grid(True, linestyle="--", alpha=0.28)
        axis.text(
            0.02,
            0.03,
            "Top-right is preferred",
            transform=axis.transAxes,
            fontsize=8.8,
            color="#444444",
            bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "#cccccc", "alpha": 0.9},
        )

    ax.set_ylim(-0.025, 0.765)
    ax_zoom.set_xlim(6305.43, 6305.99)
    ax_zoom.set_ylim(0.675, 0.742)
    ax_zoom.set_ylabel("")
    ax_zoom.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax_zoom.tick_params(axis="x", labelrotation=25)

    fig.subplots_adjust(bottom=0.18, top=0.86, wspace=0.24)
    fig.savefig(OUTPUT_PNG, dpi=240, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    points = enrich_reference_points(load_tradeoff_points())
    points = sorted(points, key=lambda point: (point.privacy_mse, point.f1))
    write_points_csv(points)
    plot(points)

    print(f"Loaded plotted points: {len(points)}")
    print(f"Output PNG: {OUTPUT_PNG.relative_to(PROJECT_ROOT)}")
    print(f"Output PDF: {OUTPUT_PDF.relative_to(PROJECT_ROOT)}")
    print(f"Output CSV: {OUTPUT_CSV.relative_to(PROJECT_ROOT)}")
    best_f1 = max(points, key=lambda point: point.f1)
    best_privacy = max(points, key=lambda point: point.privacy_mse)
    conservative = next((point for point in points if point.method == "DNA Transform conservative"), None)
    print(f"Best F1: {best_f1.method} ({best_f1.f1:.4f})")
    print(f"Highest privacy MSE: {best_privacy.method} ({best_privacy.privacy_mse:.2f})")
    if conservative:
        print(
            "DNA Transform conservative: "
            f"F1={conservative.f1:.4f}, privacy_mse={conservative.privacy_mse:.2f}"
        )


if __name__ == "__main__":
    main()
