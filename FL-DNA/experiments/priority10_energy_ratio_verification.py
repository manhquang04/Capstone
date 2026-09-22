"""Verify full per-key update-energy accounting for Priority 10 diagnostics.

This is descriptive only.  It reads existing artifacts or recomputes the same
local update from saved target bundles; it does not create targets or run
attackers.
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

from experiments.priority10_per_layer_energy_diagnostic import (
    observed_updates_from_ieee_bundle,
    observed_updates_from_paysim_targets,
    source_signature,
)
from privacy.seed_manager import derive_seed


def dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


def stored_update_records(root: Path, dataset: str, limit_unique: int | None = None) -> list[dict]:
    records = []
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
        records.append(
            {
                "dataset": dataset,
                "source": "stored_observed_update",
                "artifact": str(path.relative_to(ROOT)),
                "group_id": payload.get("group_id", payload.get("group")),
                "source_signature": signature,
                "update": update,
            }
        )
        seen.add(signature)
        if limit_unique is not None and len(records) >= limit_unique:
            break
    return records


def ieee_bundle_records(bundle_path: Path, run_seed: int, source: str) -> list[dict]:
    from experiments.run_ieee_cis_rq1_development_gate import capture as capture_ieee

    bundle = torch.load(bundle_path, weights_only=False, map_location="cpu")
    records = []
    for group in bundle["targets"]:
        group_id = int(group["group_id"])
        _, _, _, _, _, observed = capture_ieee(
            group,
            bundle["state_dict"],
            bundle["pos_weight"],
            derive_seed(run_seed, "priority10-energy-ieee-local", group_id),
            bundle["local_lr"],
        )
        records.append(
            {
                "dataset": "IEEE-CIS",
                "source": source,
                "artifact": str(bundle_path.relative_to(ROOT)),
                "group_id": group_id,
                "source_signature": source_signature(group),
                "update": observed,
            }
        )
    return records


def paysim_target_records(target_path: Path, run_seed: int, source: str, limit: int | None = None) -> list[dict]:
    from experiments.run_phase4_dna_v2_iht_attack import _capture_for_iht as capture_paysim

    groups = torch.load(target_path, weights_only=False, map_location="cpu")
    records = []
    for group_id, group in enumerate(groups):
        _, _, _, _, _, _, observed = capture_paysim(
            group,
            derive_seed(run_seed, "priority10-energy-paysim-local", group_id),
            4,
        )
        records.append(
            {
                "dataset": "PaySim",
                "source": source,
                "artifact": str(target_path.relative_to(ROOT)),
                "group_id": group_id,
                "source_signature": source_signature(group),
                "update": observed,
            }
        )
        if limit is not None and len(records) >= limit:
            break
    return records


def collect_records() -> list[dict]:
    records: list[dict] = []
    records.extend(
        stored_update_records(
            ROOT / "artifacts/rq1_v2/confirmatory_run_20260916/dna_v2",
            "PaySim",
            limit_unique=40,
        )
    )
    records.extend(
        stored_update_records(
            ROOT / "artifacts/priority5_ieee_cis/phase53_rq1_development_gate_20260916_rerun1",
            "IEEE-CIS",
            limit_unique=None,
        )
    )
    records.extend(
        ieee_bundle_records(
            ROOT / "artifacts/priority6_sota_attackers/targets_ieee_n24/ieee_priority6_bundle.pt",
            2026092103,
            "recomputed_from_saved_ieee_bundle_priority6_n24",
        )
    )
    records.extend(
        ieee_bundle_records(
            ROOT / "artifacts/priority10_v3/ieee_n8_targets_20260921/ieee_priority6_bundle.pt",
            2026092104,
            "recomputed_from_saved_ieee_bundle_priority10_v3_n8",
        )
    )
    records.extend(
        paysim_target_records(
            ROOT / "artifacts/priority6_sota_attackers/targets_paysim_n24/paysim_priority6_n24_targets.pt",
            2026092105,
            "recomputed_from_saved_paysim_targets_priority6_n24",
            limit=24,
        )
    )
    records.extend(
        paysim_target_records(
            ROOT / "artifacts/priority9_v1_family/stronger_n24_targets_20260917/paysim_priority9_stronger_n24_targets.pt",
            2026092106,
            "recomputed_from_saved_paysim_targets_priority9_stronger_n24",
            limit=24,
        )
    )
    deduped = []
    seen = set()
    for record in records:
        key = (record["dataset"], record["source_signature"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(record)
    return deduped


def classify_key(key: str, tensor: torch.Tensor) -> str:
    if "running_mean" in key or "running_var" in key or "num_batches_tracked" in key:
        return "batchnorm_buffer"
    if key.endswith(".weight") or key.endswith(".bias"):
        return "parameter"
    return "other"


def per_key_rows(records: list[dict]) -> tuple[list[dict], list[dict]]:
    key_rows = []
    sample_checks = []
    for record in records:
        update: dict[str, torch.Tensor] = record["update"]
        floating = {
            key: value.detach().double().reshape(-1)
            for key, value in update.items()
            if value.is_floating_point()
        }
        nonfloating = {
            key: value
            for key, value in update.items()
            if not value.is_floating_point()
        }
        total_energy = float(sum(torch.sum(value * value).item() for value in floating.values()))
        if total_energy <= 0.0:
            raise ValueError("total update energy is zero")
        ratio_sum = 0.0
        for key, value in floating.items():
            energy = float(torch.sum(value * value).item())
            ratio = energy / total_energy
            ratio_sum += ratio
            key_rows.append(
                {
                    "dataset": record["dataset"],
                    "source": record["source"],
                    "artifact": record["artifact"],
                    "group_id": record["group_id"],
                    "source_signature": record["source_signature"],
                    "key": key,
                    "kind": classify_key(key, value),
                    "numel": int(value.numel()),
                    "energy": energy,
                    "ratio": ratio,
                    "percent": ratio * 100.0,
                }
            )
        sample_checks.append(
            {
                "dataset": record["dataset"],
                "source": record["source"],
                "artifact": record["artifact"],
                "group_id": record["group_id"],
                "source_signature": record["source_signature"],
                "floating_keys": len(floating),
                "nonfloating_keys": len(nonfloating),
                "nonfloating_key_names": ";".join(sorted(nonfloating)),
                "total_energy": total_energy,
                "ratio_sum": ratio_sum,
                "percent_sum": ratio_sum * 100.0,
                "absolute_error_from_one": abs(1.0 - ratio_sum),
            }
        )
    return key_rows, sample_checks


def summarize_keys(key_rows: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in key_rows:
        grouped[(row["dataset"], row["key"])].append(row)
    out = []
    for (dataset, key), rows in sorted(grouped.items()):
        ratios = np.asarray([float(row["ratio"]) for row in rows], dtype=float)
        energies = np.asarray([float(row["energy"]) for row in rows], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "key": key,
                "kind": rows[0]["kind"],
                "numel": int(rows[0]["numel"]),
                "samples": int(ratios.size),
                "mean_ratio": float(ratios.mean()),
                "median_ratio": float(np.median(ratios)),
                "mean_percent": float(ratios.mean() * 100.0),
                "median_percent": float(np.median(ratios) * 100.0),
                "min_percent": float(ratios.min() * 100.0),
                "max_percent": float(ratios.max() * 100.0),
                "mean_energy": float(energies.mean()),
            }
        )
    return out


def summarize_checks(checks: list[dict]) -> list[dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in checks:
        grouped[row["dataset"]].append(row)
    out = []
    for dataset, rows in sorted(grouped.items()):
        errors = np.asarray([float(row["absolute_error_from_one"]) for row in rows], dtype=float)
        percent_sums = np.asarray([float(row["percent_sum"]) for row in rows], dtype=float)
        out.append(
            {
                "dataset": dataset,
                "samples": len(rows),
                "min_percent_sum": float(percent_sums.min()),
                "max_percent_sum": float(percent_sums.max()),
                "max_absolute_error_from_one": float(errors.max()),
                "nonfloating_key_names": sorted({name for row in rows for name in row["nonfloating_key_names"].split(";") if name}),
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
    out = ROOT / "results/priority10_energy_verification"
    out.mkdir(parents=True, exist_ok=True)
    records = collect_records()
    key_rows, sample_checks = per_key_rows(records)
    key_summary = summarize_keys(key_rows)
    check_summary = summarize_checks(sample_checks)
    write_csv(out / "per_sample_key_energy.csv", key_rows)
    write_csv(out / "sample_energy_sum_checks.csv", sample_checks)
    write_csv(out / "key_energy_summary.csv", key_summary)
    dump(
        out / "summary.json",
        {
            "key_summary": key_summary,
            "check_summary": check_summary,
            "sample_count": len(records),
            "update_semantics": "full model state delta after local Adam replay; includes trainable parameter deltas and floating BatchNorm running-stat buffer deltas; integer num_batches_tracked buffers are excluded from energy sums",
        },
    )
    print(json.dumps({"check_summary": check_summary, "sample_count": len(records)}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
