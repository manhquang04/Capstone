"""Descriptive per-layer update-energy diagnostic for Priority 10.

This script uses the same existing artifact sources and deterministic
reconstruction seeds as ``priority10_energy_ratio_diagnostic.py``.  It does not
create targets, run attackers, or modify experiment artifacts.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.priority10_energy_ratio_diagnostic import (
    collect_ieee_bundle,
    collect_paysim_targets,
    collect_stored_observed,
)


LINEAR_LAYERS = (
    ("network.0", "input_to_128"),
    ("network.4", "128_to_64"),
    ("network.8", "64_to_32"),
    ("network.12", "32_to_1"),
)


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def load_update_from_row(row: dict) -> dict[str, torch.Tensor]:
    """Recover the update dict for stored-observed rows only."""

    payload = torch.load(ROOT / row["artifact"], weights_only=False, map_location="cpu")
    update = payload.get("observed_raw_update", payload.get("observed"))
    if not isinstance(update, dict):
        raise ValueError(f"stored row has no update dict: {row['artifact']}")
    return update


def layer_ratios(update: dict[str, torch.Tensor]) -> dict[str, float | int]:
    floating = {
        key: value.detach().double().reshape(-1)
        for key, value in update.items()
        if value.is_floating_point()
    }
    total_energy = float(sum(torch.sum(value * value).item() for value in floating.values()))
    if total_energy <= 0.0:
        raise ValueError("total update energy is zero")
    total_params = int(sum(value.numel() for value in floating.values()))
    out: dict[str, float | int] = {
        "total_energy": total_energy,
        "total_params": total_params,
    }
    for prefix, label in LINEAR_LAYERS:
        keys = (f"{prefix}.weight", f"{prefix}.bias")
        missing = [key for key in keys if key not in floating]
        if missing:
            raise KeyError(f"missing layer keys for {prefix}: {missing}")
        energy = float(sum(torch.sum(floating[key] * floating[key]).item() for key in keys))
        params = int(sum(floating[key].numel() for key in keys))
        out[f"{label}_energy"] = energy
        out[f"{label}_ratio"] = energy / total_energy
        out[f"{label}_params"] = params
    return out


def source_signature(item: dict) -> str:
    for key in ("source_rows", "source_ids"):
        if key in item:
            values = item[key]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().reshape(-1).tolist()
            return ",".join(str(int(v)) for v in values)
    return ""


def stored_update_rows(root: Path, dataset: str, limit_unique: int | None = None) -> list[dict]:
    rows = []
    seen = set()
    for path in sorted(root.rglob("*.pt")):
        try:
            payload = torch.load(path, weights_only=False, map_location="cpu")
        except Exception:
            continue
        if not isinstance(payload, dict):
            continue
        update = payload.get("observed_raw_update", payload.get("observed"))
        if not isinstance(update, dict):
            continue
        signature = source_signature(payload) or f"{payload.get('group_id')}:{path.parent}"
        if signature in seen:
            continue
        rows.append(
            {
                "dataset": dataset,
                "source": "stored_observed_update",
                "artifact": str(path.relative_to(ROOT)),
                "group_id": payload.get("group_id", payload.get("group")),
                "source_signature": signature,
                **layer_ratios(update),
            }
        )
        seen.add(signature)
        if limit_unique is not None and len(rows) >= limit_unique:
            break
    return rows


def observed_updates_from_ieee_bundle(bundle_path: Path, run_seed: int, source: str) -> list[dict]:
    from experiments.run_ieee_cis_rq1_development_gate import capture as capture_ieee
    from privacy.seed_manager import derive_seed

    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    rows = []
    for group in bundle["targets"]:
        group_id = int(group["group_id"])
        _, _, _, _, _, observed = capture_ieee(
            group,
            bundle["state_dict"],
            bundle["pos_weight"],
            derive_seed(run_seed, "priority10-energy-ieee-local", group_id),
            bundle["local_lr"],
        )
        rows.append(
            {
                "dataset": "IEEE-CIS",
                "source": source,
                "artifact": str(bundle_path.relative_to(ROOT)),
                "group_id": group_id,
                "source_signature": source_signature(group),
                **layer_ratios(observed),
            }
        )
    return rows


def observed_updates_from_paysim_targets(target_path: Path, run_seed: int, source: str, limit: int | None = None) -> list[dict]:
    from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim
    from privacy.seed_manager import derive_seed

    groups = torch.load(target_path, weights_only=False, map_location="cpu")
    rows = []
    for group_id, group in enumerate(groups):
        _, _, _, _, _, _, observed = capture_paysim(
            group,
            derive_seed(run_seed, "priority10-energy-paysim-local", group_id),
            4,
        )
        rows.append(
            {
                "dataset": "PaySim",
                "source": source,
                "artifact": str(target_path.relative_to(ROOT)),
                "group_id": group_id,
                "source_signature": source_signature(group),
                **layer_ratios(observed),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    rows.extend(
        stored_update_rows(
            ROOT / "artifacts/rq1_v2/confirmatory_run_20260916/dna_v2",
            "PaySim",
            limit_unique=40,
        )
    )
    rows.extend(
        stored_update_rows(
            ROOT / "artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1",
            "IEEE-CIS",
            limit_unique=None,
        )
    )
    rows.extend(
        observed_updates_from_ieee_bundle(
            ROOT / "artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt",
            2026092103,
            "recomputed_from_saved_ieee_bundle_priority6_n24",
        )
    )
    rows.extend(
        observed_updates_from_ieee_bundle(
            ROOT / "artifacts/priority10_v3/ieee_n8_targets_20260921/ieee_priority6_bundle.pt",
            2026092104,
            "recomputed_from_saved_ieee_bundle_priority10_v3_n8",
        )
    )
    rows.extend(
        observed_updates_from_paysim_targets(
            ROOT / "artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt",
            2026092105,
            "recomputed_from_saved_paysim_targets_priority6_n24",
            limit=24,
        )
    )
    rows.extend(
        observed_updates_from_paysim_targets(
            ROOT / "artifacts/priority9_v1_family/stronger_n24_targets_20260917/paysim_priority9_stronger_n24_targets.pt",
            2026092106,
            "recomputed_from_saved_paysim_targets_priority9_stronger_n24",
            limit=24,
        )
    )
    deduped = []
    seen = set()
    for row in rows:
        key = (row["dataset"], row["source_signature"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    return deduped


def summarize(rows: list[dict]) -> list[dict]:
    out = []
    by_dataset: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, dataset_rows in sorted(by_dataset.items()):
        for _prefix, label in LINEAR_LAYERS:
            values = np.asarray([float(row[f"{label}_ratio"]) for row in dataset_rows], dtype=float)
            out.append(
                {
                    "dataset": dataset,
                    "layer": label,
                    "parameter_keys": f"{label.replace('_to_', '->')}",
                    "samples": int(values.size),
                    "mean_ratio": float(values.mean()),
                    "median_ratio": float(np.median(values)),
                    "mean_percent": float(values.mean() * 100.0),
                    "median_percent": float(np.median(values) * 100.0),
                    "params": int(dataset_rows[0][f"{label}_params"]),
                    "total_params_min": int(min(row["total_params"] for row in dataset_rows)),
                    "total_params_max": int(max(row["total_params"] for row in dataset_rows)),
                }
            )
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    torch.set_num_threads(1)
    out = ROOT / "results/priority10_per_layer_energy"
    out.mkdir(parents=True, exist_ok=True)
    rows = collect_rows()
    summary = summarize(rows)
    write_csv(out / "per_sample_layer_energy.csv", rows)
    dump(out / "summary.json", {"summary": summary, "samples": rows})
    print(json.dumps({"summary": summary, "sample_count": len(rows)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
