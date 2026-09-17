"""Invert the project Gaussian-RDP diagnostic for fixed update-level budgets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp


ROOT = Path(__file__).resolve().parents[1]
TARGETS = (100.0, 50.0, 10.0, 1.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_multiplier(target: float, sensitivity: float, orders: list[float]) -> dict:
    low, high = 1e-12, 1.0
    while epsilon_from_rdp(high, sensitivity, 1e-5, 1, orders)["epsilon"] > target:
        high *= 2.0
    for _ in range(80):
        middle = (low + high) / 2.0
        if epsilon_from_rdp(middle, sensitivity, 1e-5, 1, orders)["epsilon"] <= target:
            high = middle
        else:
            low = middle
    result = epsilon_from_rdp(high, sensitivity, 1e-5, 1, orders)
    return {"noise_multiplier": high, **result}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    orders = alpha_grid()
    rows = []
    for target in TARGETS:
        primary = find_multiplier(target, 1.0, orders)
        replace = epsilon_from_rdp(primary["noise_multiplier"], 2.0, 1e-5, 1, orders)
        rows.append({
            "target_epsilon_add_remove": target,
            "clip_norm": 100.0,
            "noise_multiplier": primary["noise_multiplier"],
            "noise_std": 100.0 * primary["noise_multiplier"],
            "attained_epsilon_add_remove": primary["epsilon"],
            "optimal_alpha_add_remove": primary["alpha"],
            "epsilon_replace_one_sensitivity": replace["epsilon"],
            "optimal_alpha_replace_one": replace["alpha"],
        })
    with (out / "strong_dp_inverse.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development-only inverse update-level RDP accounting",
        "mechanism": "full-client update L2 clip C=100 then Gaussian noise N(0, (sigma*C)^2 I)",
        "record_level_dp_claim": False,
        "primary_convention": {"adjacency": "add_remove", "sensitivity_ratio": 1.0, "delta": 1e-5, "compositions": 1},
        "secondary_sensitivity": {"adjacency": "replace_one", "sensitivity_ratio": 2.0},
        "selection_rule": "smallest multiplier with calculated epsilon <= requested target", 
        "accountant_script_sha256": sha256(ROOT / "experiments/dp_update_accounting.py"),
        "this_script_sha256": sha256(Path(__file__).resolve()),
        "rows": rows,
    }
    (out / "strong_dp_inverse.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
