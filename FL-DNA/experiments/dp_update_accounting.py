"""RDP accounting diagnostics for update-level Gaussian clipping/noise settings."""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path


CONFIGS = [
    {
        "config_id": "rq1_rq2_conservative_distortion_matched",
        "clip_norm": 100.0,
        "noise_multiplier": 0.00025,
        "source": "protocols/config/rq1_confirmatory.yaml and protocols/config/rq2_confirmatory.yaml",
    },
    {
        "config_id": "rq1_medium_distortion_matched",
        "clip_norm": 100.0,
        "noise_multiplier": 0.000315,
        "source": "protocols/config/rq1_medium_confirmatory.json",
    },
    {
        "config_id": "rq1_stronger_distortion_matched",
        "clip_norm": 100.0,
        "noise_multiplier": 0.0004,
        "source": "protocols/config/rq1_stronger_confirmatory.json",
    },
    {
        "config_id": "rq2_development_utility_matched_extension",
        "clip_norm": 100.0,
        "noise_multiplier": 0.00001,
        "source": "artifacts/rq2/utility_extension_20260913/utility_extension_report.json",
    },
    {
        "config_id": "historical_default_medium_preset",
        "clip_norm": 100.0,
        "noise_multiplier": 0.005,
        "source": "privacy/dp_config.py",
    },
]

SENSITIVITY_CONVENTIONS = {
    "add_remove": 1.0,
    "replace_one": 2.0,
}

COMPOSITION_SCENARIOS = {
    "single_release": 1,
    "rq2_same_client_50_rounds": 50,
}


def alpha_grid() -> list[float]:
    values = [1.0 + step * 1e-5 for step in range(1, 1001)]
    values.extend(1.0 + step / 100.0 for step in range(2, 901))
    values.extend(10.0 + step / 2.0 for step in range(1, 1005))
    return sorted(set(values))


def epsilon_from_rdp(
    noise_multiplier: float,
    sensitivity_ratio: float,
    delta: float,
    compositions: int,
    orders: list[float],
) -> dict:
    if noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be positive")
    if not 0 < delta < 1:
        raise ValueError("delta must be in (0, 1)")
    best = None
    log_delta = math.log(1.0 / delta)
    for alpha in orders:
        rdp = compositions * alpha * (sensitivity_ratio**2) / (2.0 * noise_multiplier**2)
        epsilon = rdp + log_delta / (alpha - 1.0)
        if best is None or epsilon < best["epsilon"]:
            best = {"epsilon": epsilon, "alpha": alpha, "rdp_at_alpha": rdp}
    assert best is not None
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--deltas", nargs="+", type=float, default=[1e-5, 1e-6, 1e-8])
    args = parser.parse_args()

    orders = alpha_grid()
    rows = []
    for config in CONFIGS:
        noise_multiplier = float(config["noise_multiplier"])
        for convention, sensitivity_ratio in SENSITIVITY_CONVENTIONS.items():
            for scenario, compositions in COMPOSITION_SCENARIOS.items():
                for delta in args.deltas:
                    result = epsilon_from_rdp(
                        noise_multiplier=noise_multiplier,
                        sensitivity_ratio=sensitivity_ratio,
                        delta=delta,
                        compositions=int(compositions),
                        orders=orders,
                    )
                    rows.append(
                        {
                            "config_id": config["config_id"],
                            "source": config["source"],
                            "clip_norm": config["clip_norm"],
                            "noise_multiplier": noise_multiplier,
                            "noise_std": config["clip_norm"] * noise_multiplier,
                            "sensitivity_convention": convention,
                            "sensitivity_ratio": sensitivity_ratio,
                            "composition_scenario": scenario,
                            "compositions": int(compositions),
                            "delta": delta,
                            "epsilon": result["epsilon"],
                            "optimal_alpha": result["alpha"],
                            "rdp_at_optimal_alpha": result["rdp_at_alpha"],
                        }
                    )

    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "dp_update_accounting.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "accountant": "Gaussian mechanism RDP diagnostic for full-client-update clipping/noise",
        "record_level_dp_claim": False,
        "reason_no_record_level_claim": "implementation clips full client/model update, not per-example gradients",
        "orders": {
            "min": min(orders),
            "max": max(orders),
            "count": len(orders),
            "description": "1.00001..1.01000 step 1e-5, 1.02..10.00 step 0.01, then 10.50..512.00 step 0.50",
        },
        "configs": CONFIGS,
        "sensitivity_conventions": SENSITIVITY_CONVENTIONS,
        "composition_scenarios": COMPOSITION_SCENARIOS,
        "deltas": args.deltas,
        "rows": rows,
    }
    (args.output_dir / "dp_update_accounting.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
