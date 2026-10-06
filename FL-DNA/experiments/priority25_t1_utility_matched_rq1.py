"""Priority 25 T1 utility-matched DP confirmatory test."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments import fraud_fl_common as common  # noqa: E402
from experiments.priority24_t1_bn_valid_rq1 import (  # noqa: E402
    BN_KEY,
    capture_groups,
    holm_adjust,
    load_population,
    mse_std,
    qualify,
    recover_mean,
    transform_full_v1,
    transform_full_v2,
    write_csv,
)
from privacy.dp_engine import apply_dp_to_local_state  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority25_rq1_extension.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def exact_p_greater(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def zero_state_like(update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for name, tensor in update.items():
        out[name] = torch.zeros_like(tensor) if torch.is_floating_point(tensor) else tensor.clone()
    return out


def full_state_dp_update(update: dict[str, torch.Tensor], clip_norm: float, noise_multiplier: float, seed: int) -> dict[str, torch.Tensor]:
    dp_state, _norm_before, _norm_after, _noise_std = apply_dp_to_local_state(
        update,
        zero_state_like(update),
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        noise_generator=torch.Generator().manual_seed(seed),
    )
    return dp_state


def branch_signals(captures: list[dict], selections: dict) -> dict[str, list[torch.Tensor]]:
    out = {
        "none": [c["observed"][BN_KEY].detach().cpu() for c in captures],
        "v1_conservative": [transform_full_v1(c["observed"])[BN_KEY].detach().cpu() for c in captures],
        "v2_0p95_eta0p01": [transform_full_v2(c["observed"])[BN_KEY].detach().cpu() for c in captures],
    }
    for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
        selection = selections[transform_id]
        clip = float(selection["selected_clip_norm"])
        sigma = float(selection["selected_noise_multiplier"])
        rows = []
        for capture in captures:
            seed = derive_seed(20260929, "priority25-utility-dp", transform_id, capture["group_id"])
            dp_update = full_state_dp_update(capture["observed"], clip, sigma, seed)
            rows.append(dp_update[BN_KEY].detach().cpu())
        out[f"dp_utility_matched_for_{transform_id}"] = rows
    return out


def score_confirm(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray, selections: dict, tie_band: float) -> dict:
    signals = branch_signals(captures, selections)
    rec = {name: [recover_mean(c["model"], vecs[i]) for i, c in enumerate(captures)] for name, vecs in signals.items()}
    rows = []
    for i, capture in enumerate(captures):
        row = {
            "group": capture["group_id"],
            "source_ids": json.dumps(capture["source_ids"]),
            "prior_mse": mse_std(pop_mean, capture["true_mean"], std),
        }
        for branch in (
            "none",
            "v1_conservative",
            "v2_0p95_eta0p01",
            "dp_utility_matched_for_v1_conservative",
            "dp_utility_matched_for_v2_0p95_eta0p01",
        ):
            row[f"{branch}_mse"] = mse_std(rec[branch][i], capture["true_mean"], std)
            row[f"{branch}_decoy_mse"] = mse_std(rec[branch][(i + 1) % len(captures)], capture["true_mean"], std)
        rows.append(row)
    tests = {}
    for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
        dp_id = f"dp_utility_matched_for_{transform_id}"
        dna_wins = dp_wins = ties = 0
        ratios = []
        for row in rows:
            dna = float(row[f"{transform_id}_mse"])
            dp = float(row[f"{dp_id}_mse"])
            ratios.append(dna / max(dp, 1e-18))
            if dna > dp + tie_band:
                dna_wins += 1
            elif dp > dna + tie_band:
                dp_wins += 1
            else:
                ties += 1
        n = dna_wins + dp_wins
        tests[transform_id] = {
            "dna_wins": dna_wins,
            "dp_wins": dp_wins,
            "ties": ties,
            "non_tied_n": n,
            "dna_greater_p": exact_p_greater(dna_wins, n),
            "dp_greater_p": exact_p_greater(dp_wins, n),
            "median_mse_ratio_dna_over_dp": float(np.median(ratios)),
            "mean_mse_ratio_dna_over_dp": float(np.mean(ratios)),
        }
    return {"rows": rows, "tests": tests}


def make_branch_control_summary(rows: list[dict]) -> list[dict]:
    branches = {
        "none": "none_mse",
        "v1_conservative": "v1_conservative_mse",
        "v2_0p95_eta0p01": "v2_0p95_eta0p01_mse",
        "dp_utility_matched_for_v1_conservative": "dp_utility_matched_for_v1_conservative_mse",
        "dp_utility_matched_for_v2_0p95_eta0p01": "dp_utility_matched_for_v2_0p95_eta0p01_mse",
    }
    out = []
    for branch, col in branches.items():
        decoy_col = "none_decoy_mse" if branch == "none" else col.replace("_mse", "_decoy_mse")
        vals = np.asarray([float(row[col]) for row in rows], dtype=float)
        prior = np.asarray([float(row["prior_mse"]) for row in rows], dtype=float)
        decoy = np.asarray([float(row[decoy_col]) for row in rows], dtype=float)
        wins_prior = int(np.sum(vals < prior))
        wins_decoy = int(np.sum(vals < decoy))
        record = {
            "branch": branch,
            "n": len(rows),
            "mean_mse": float(np.mean(vals)),
            "median_mse": float(np.median(vals)),
            "beats_prior_wins": wins_prior,
            "beats_prior_p_greater": exact_p_greater(wins_prior, len(rows)),
            "beats_decoy_wins": wins_decoy,
            "beats_decoy_p_greater": exact_p_greater(wins_decoy, len(rows)),
        }
        if branch != "none":
            none = np.asarray([float(row["none_mse"]) for row in rows], dtype=float)
            higher = int(np.sum(vals > none))
            record["higher_mse_than_none_wins"] = higher
            record["higher_mse_than_none_p_greater"] = exact_p_greater(higher, len(rows))
        out.append(record)
    return out


def write_union_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_examples(rows: list[dict]) -> list[dict]:
    keys = {
        "group",
        "source_ids",
        "prior_mse",
        "none_mse",
        "v1_conservative_mse",
        "v2_0p95_eta0p01_mse",
        "dp_utility_matched_for_v1_conservative_mse",
        "dp_utility_matched_for_v2_0p95_eta0p01_mse",
    }
    return [{key: row[key] for key in row if key in keys} for row in rows[:4]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-target", type=Path, required=True)
    parser.add_argument("--confirm-target", type=Path, required=True)
    parser.add_argument("--utility-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, default=AMENDMENT)
    parser.add_argument("--seed", type=int, default=2026092951)
    args = parser.parse_args()

    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    utility_summary = json.loads(args.utility_summary.read_text())
    selections = utility_summary["selections"]
    pop_mean, std, feature_names = load_population()

    dev = capture_groups(args.dev_target.resolve(), args.seed, "priority25_t1_utility_dev")
    dev_q = qualify(dev, pop_mean, std)
    dev_a = score_confirm(dev, pop_mean, std, selections, tie_band=0.0)
    dev_b = score_confirm(dev, pop_mean, std, selections, tie_band=0.0)
    tie_band = 0.0
    for row_a, row_b in zip(dev_a["rows"], dev_b["rows"]):
        for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
            dp_id = f"dp_utility_matched_for_{transform_id}"
            d_a = row_a[f"{transform_id}_mse"] - row_a[f"{dp_id}_mse"]
            d_b = row_b[f"{transform_id}_mse"] - row_b[f"{dp_id}_mse"]
            tie_band = max(tie_band, abs(d_a - d_b))

    confirm = {"rows": [], "tests": {}}
    if dev_q["summary"]["qualified"]:
        captures = capture_groups(args.confirm_target.resolve(), args.seed, "priority25_t1_utility_confirm")
        confirm = score_confirm(captures, pop_mean, std, selections, tie_band=tie_band)
        pvals_dna = {key: value["dna_greater_p"] for key, value in confirm["tests"].items()}
        pvals_dp = {key: value["dp_greater_p"] for key, value in confirm["tests"].items()}
        holm_dna = holm_adjust(pvals_dna)
        holm_dp = holm_adjust(pvals_dp)
        for key, value in confirm["tests"].items():
            value["holm_dna_greater_p_within_priority25_utility"] = holm_dna[key]
            value["holm_dp_greater_p_within_priority25_utility"] = holm_dp[key]

    write_csv(out / "dev_t1_qualification.csv", dev_q["rows"])
    write_csv(out / "confirm_t1_per_target.csv", confirm["rows"])
    write_csv(out / "confirm_t1_examples.csv", make_examples(confirm["rows"]))
    write_union_csv(out / "confirm_t1_branch_vs_controls.csv", make_branch_control_summary(confirm["rows"]))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(args.amendment.resolve().relative_to(ROOT)),
        "amendment_sha256": sha256(args.amendment.resolve()),
        "utility_summary": str(args.utility_summary.resolve().relative_to(ROOT)),
        "utility_summary_sha256": sha256(args.utility_summary.resolve()),
        "dev_target": str(args.dev_target.resolve().relative_to(ROOT)),
        "dev_target_sha256": sha256(args.dev_target.resolve()),
        "confirm_target": str(args.confirm_target.resolve().relative_to(ROOT)),
        "confirm_target_sha256": sha256(args.confirm_target.resolve()),
        "feature_names": feature_names,
        "instrument": "T1_BN_RUNNING_MEAN_BATCH_MEAN_RECOVERY",
        "matching": "utility_matched_dp_full_floating_state",
        "selected_dp": selections,
        "dev_qualification": dev_q["summary"],
        "tie_band_from_development_replay": tie_band,
        "confirmatory_tests": confirm["tests"],
    }
    dump(out / "priority25_t1_utility_summary.json", summary)
    manifest = {
        "summary_sha256": sha256(out / "priority25_t1_utility_summary.json"),
        "dev_csv_sha256": sha256(out / "dev_t1_qualification.csv"),
        "confirm_csv_sha256": sha256(out / "confirm_t1_per_target.csv"),
        "examples_csv_sha256": sha256(out / "confirm_t1_examples.csv"),
        "branch_vs_controls_csv_sha256": sha256(out / "confirm_t1_branch_vs_controls.csv"),
    }
    dump(out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
