"""Descriptive final-layer update-energy diagnostic for Priority 10.

This script reads existing artifacts only.  It does not create targets, run an
attacker, or modify any experiment artifact.
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

from attacks.local_update import simulate
from experiments.run_ieee_cis_rq1_development_gate import (
    FeatureContract,
    capture as capture_ieee,
)
from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim
from privacy.seed_manager import derive_seed


FINAL_LAYER_KEYS = ("network.12.weight", "network.12.bias")


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def energy_ratio(update: dict[str, torch.Tensor]) -> dict[str, float | int]:
    floating = {key: value.detach().double().reshape(-1) for key, value in update.items() if value.is_floating_point()}
    missing = [key for key in FINAL_LAYER_KEYS if key not in floating]
    if missing:
        raise KeyError(f"missing final-layer update keys: {missing}")
    total_energy = float(sum(torch.sum(value * value).item() for value in floating.values()))
    final_energy = float(sum(torch.sum(floating[key] * floating[key]).item() for key in FINAL_LAYER_KEYS))
    total_params = int(sum(value.numel() for value in floating.values()))
    final_params = int(sum(floating[key].numel() for key in FINAL_LAYER_KEYS))
    return {
        "ratio": final_energy / total_energy if total_energy > 0.0 else float("nan"),
        "final_energy": final_energy,
        "total_energy": total_energy,
        "final_params": final_params,
        "total_params": total_params,
        "parameter_fraction": final_params / total_params,
    }


def source_signature(item: dict) -> str:
    for key in ("source_rows", "source_ids"):
        if key in item:
            values = item[key]
            if isinstance(values, torch.Tensor):
                values = values.detach().cpu().reshape(-1).tolist()
            return ",".join(str(int(v)) for v in values)
    return ""


def collect_stored_observed(root: Path, dataset: str, limit_unique: int | None = None) -> list[dict]:
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
        metrics = energy_ratio(update)
        rows.append(
            {
                "dataset": dataset,
                "source": "stored_observed_update",
                "artifact": str(path.relative_to(ROOT)),
                "group_id": payload.get("group_id", payload.get("group")),
                "source_signature": signature,
                **metrics,
            }
        )
        seen.add(signature)
        if limit_unique is not None and len(rows) >= limit_unique:
            break
    return rows


def collect_ieee_bundle(bundle_path: Path, run_seed: int, source: str) -> list[dict]:
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
                **energy_ratio(observed),
            }
        )
    return rows


def collect_paysim_targets(target_path: Path, run_seed: int, source: str, limit: int | None = None) -> list[dict]:
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
                **energy_ratio(observed),
            }
        )
        if limit is not None and len(rows) >= limit:
            break
    return rows


def summarize(rows: list[dict]) -> list[dict]:
    grouped: dict[str, list[float]] = defaultdict(list)
    params: dict[str, list[int]] = defaultdict(list)
    fractions: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(float(row["ratio"]))
        params[row["dataset"]].append(int(row["total_params"]))
        fractions[row["dataset"]].append(float(row["parameter_fraction"]))
    out = []
    for dataset, values in sorted(grouped.items()):
        arr = np.asarray(values, dtype=float)
        out.append(
            {
                "dataset": dataset,
                "samples": int(arr.size),
                "mean_ratio": float(arr.mean()),
                "median_ratio": float(np.median(arr)),
                "min_ratio": float(arr.min()),
                "max_ratio": float(arr.max()),
                "mean_ratio_percent": float(arr.mean() * 100.0),
                "median_ratio_percent": float(np.median(arr) * 100.0),
                "min_ratio_percent": float(arr.min() * 100.0),
                "max_ratio_percent": float(arr.max() * 100.0),
                "final_params": 33,
                "total_params_min": int(min(params[dataset])),
                "total_params_max": int(max(params[dataset])),
                "parameter_fraction_mean_percent": float(np.mean(fractions[dataset]) * 100.0),
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
    out = ROOT / "results/priority10_energy_ratio"
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # Stored observed updates are preferred when available.
    rows.extend(
        collect_stored_observed(
            ROOT / "artifacts/rq1_v2/confirmatory_run_20260916/dna_v2",
            "PaySim",
            limit_unique=40,
        )
    )
    rows.extend(
        collect_stored_observed(
            ROOT / "artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1",
            "IEEE-CIS",
            limit_unique=None,
        )
    )

    # Priority 6/8/10 IEEE bundles did not serialize observed updates, so these
    # rows recompute the deterministic local update from saved artifact bundles.
    rows.extend(
        collect_ieee_bundle(
            ROOT / "artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt",
            2026092103,
            "recomputed_from_saved_ieee_bundle_priority6_n24",
        )
    )
    rows.extend(
        collect_ieee_bundle(
            ROOT / "artifacts/priority10_v3/ieee_n8_targets_20260921/ieee_priority6_bundle.pt",
            2026092104,
            "recomputed_from_saved_ieee_bundle_priority10_v3_n8",
        )
    )

    # Add PaySim target artifacts from Priority 6/9 for provenance diversity.
    rows.extend(
        collect_paysim_targets(
            ROOT / "artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt",
            2026092105,
            "recomputed_from_saved_paysim_targets_priority6_n24",
            limit=24,
        )
    )
    rows.extend(
        collect_paysim_targets(
            ROOT / "artifacts/priority9_v1_family/stronger_n24_targets_20260917/paysim_priority9_stronger_n24_targets.pt",
            2026092106,
            "recomputed_from_saved_paysim_targets_priority9_stronger_n24",
            limit=24,
        )
    )

    # Deduplicate within each dataset by source rows to avoid counting baseline
    # and zero-update copies as independent samples.
    deduped = []
    seen = set()
    for row in rows:
        key = (row["dataset"], row["source_signature"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(row)
    rows = deduped
    summary = summarize(rows)
    write_csv(out / "per_sample_energy_ratio.csv", rows)
    dump(out / "summary.json", {"summary": summary, "samples": rows})
    print(json.dumps({"summary": summary, "sample_count": len(rows)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
