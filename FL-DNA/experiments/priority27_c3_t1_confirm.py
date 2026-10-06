"""Priority 27 C3 T1 BN-channel RQ1-extension confirmatory tests."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.transform_defense import DNATransformConfig, transform_update_array  # noqa: E402
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2  # noqa: E402
from experiments.priority16_v1_medium_dna_vs_dp_probe import capture_paysim  # noqa: E402
from experiments.priority24_t1_bn_valid_rq1 import load_population, recover_mean  # noqa: E402
from experiments.priority27_dp_variants import apply_dp_variant_to_local_state  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority27_gaps_and_harness.md"
BN_KEY = "network.1.running_mean"
V1 = DNATransformConfig(block_size=256, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=681958327)
V2 = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=20260916)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def exact_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def mse_std(a: np.ndarray, b: np.ndarray, std: np.ndarray) -> float:
    return float(np.mean(((a - b) / std) ** 2))


def update_to_local_state(model: torch.nn.Module, update: dict[str, torch.Tensor]) -> OrderedDict[str, torch.Tensor]:
    global_state = model.state_dict()
    local = OrderedDict()
    for name, tensor in global_state.items():
        if name in update:
            local[name] = tensor.detach().cpu() + update[name].detach().cpu()
        else:
            local[name] = tensor.detach().cpu().clone()
    return local


def transform_full_v1(update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    direct = {}
    debiased = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        if not torch.is_floating_point(tensor):
            direct[name] = tensor.detach().cpu().clone()
            debiased[name] = tensor.detach().cpu().clone()
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        transformed, _ = transform_update_array(arr, V1, tensor_index=tensor_index)
        direct[name] = torch.from_numpy(transformed).to(dtype=tensor.dtype)
        # Pre-registered adaptive Level-1 debias estimator for v1.  It removes
        # the global mean component of the residual mix but does not use the
        # secret block permutation.
        mean = float(np.mean(transformed))
        estimate = (transformed - V1.mix_ratio * mean) / max(1.0 - V1.mix_ratio, 1e-12)
        debiased[name] = torch.from_numpy(estimate.astype(np.float32, copy=False)).to(dtype=tensor.dtype)
    return {"direct": direct, "debias": debiased}


def transform_full_v2(update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        if not torch.is_floating_point(tensor):
            out[name] = tensor.detach().cpu().clone()
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        _, reconstructed, _, _ = transform_and_reconstruct_array_v2(
            arr,
            V2,
            tensor_index=tensor_index,
            quantization_seed=derive_seed(V2.seed, "priority27-c3-v2", tensor_index),
        )
        out[name] = torch.from_numpy(reconstructed).to(dtype=tensor.dtype)
    return out


def apply_dp(update: dict[str, torch.Tensor], model: torch.nn.Module, variant: str, clip_spec, sigma: float, seed: int) -> dict[str, torch.Tensor]:
    global_state = OrderedDict((name, tensor.detach().cpu().clone()) for name, tensor in model.state_dict().items())
    local_state = update_to_local_state(model, update)
    dp_state, _diag = apply_dp_variant_to_local_state(
        local_state,
        global_state,
        variant=variant,
        clip_spec=clip_spec,
        noise_multiplier=sigma,
        noise_generator=torch.Generator().manual_seed(seed),
    )
    return {name: dp_state[name] - global_state[name] for name in dp_state}


def capture_rows(target: Path, seed: int, cell: str) -> list[dict]:
    groups = torch.load(target, map_location="cpu", weights_only=False)
    rows = []
    for group_id, group in enumerate(groups):
        local_seed = derive_seed(seed, "priority27-c3-local", cell, group_id)
        model, _criterion, x, _y, _batches, _rng, observed = capture_paysim(group, local_seed, 4)
        rows.append(
            {
                "group": group_id,
                "source_ids": group["source_ids"],
                "model": model,
                "true_mean": x.detach().cpu().numpy().astype(np.float64).mean(axis=0),
                "observed": observed,
            }
        )
    return rows


def score_branches(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray, config: dict, seed: int) -> tuple[list[dict], dict]:
    rows = []
    branch_vectors: dict[str, list[np.ndarray]] = {}
    branch_mses: dict[str, list[float]] = {}
    for item in captures:
        group = int(item["group"])
        model = item["model"]
        observed = item["observed"]
        true_mean = item["true_mean"]
        v1_options = transform_full_v1(observed)
        # Choose best attacker estimator for v1 per target (lowest MSE),
        # because C3 evaluates adaptive attacker strength.
        v1_candidates = {}
        for estimator, state in v1_options.items():
            v1_candidates[estimator] = recover_mean(model, state[BN_KEY])
        v1_best_name, v1_best = min(v1_candidates.items(), key=lambda kv: mse_std(kv[1], true_mean, std))
        v2_state = transform_full_v2(observed)
        recs = {
            "none": recover_mean(model, observed[BN_KEY]),
            "v1_conservative": v1_best,
            "v2_0p95_eta0p01": recover_mean(model, v2_state[BN_KEY]),
            "population_mean": pop_mean,
        }
        estimator_names = {"v1_conservative": v1_best_name, "v2_0p95_eta0p01": "key_holder_lift"}
        for variant, selected_by_transform in config["selected_dp"].items():
            for transform_id, selected in selected_by_transform.items():
                if not selected["bracketed"]:
                    continue
                sigma = float(selected["selected_noise_multiplier"])
                clip_spec = config["clip_specs"][variant]
                dp_update = apply_dp(
                    observed,
                    model,
                    variant,
                    clip_spec,
                    sigma,
                    derive_seed(seed, "priority27-c3-dp", variant, transform_id, group),
                )
                recs[f"dp_{variant}_for_{transform_id}"] = recover_mean(model, dp_update[BN_KEY])
        base = {
            "group": group,
            "source_ids": json.dumps(item["source_ids"]),
            "v1_adaptive_estimator": v1_best_name,
        }
        for branch, rec in recs.items():
            mse = mse_std(rec, true_mean, std)
            rows.append({**base, "branch": branch, "mse": mse})
            branch_vectors.setdefault(branch, []).append(rec)
            branch_mses.setdefault(branch, []).append(mse)
    summary = {
        branch: {
            "mean_mse": float(np.mean(values)),
            "median_mse": float(np.median(values)),
        }
        for branch, values in branch_mses.items()
    }
    return rows, summary


def compare(rows: list[dict], config: dict) -> dict:
    by_group_branch = {(int(row["group"]), row["branch"]): float(row["mse"]) for row in rows}
    groups = sorted({int(row["group"]) for row in rows})
    tests = {}
    for variant, selected_by_transform in config["selected_dp"].items():
        for transform_id, selected in selected_by_transform.items():
            if not selected["bracketed"]:
                continue
            dp_branch = f"dp_{variant}_for_{transform_id}"
            dna_wins = dp_wins = ties = 0
            per_target = []
            for group in groups:
                dna_mse = by_group_branch[(group, transform_id)]
                dp_mse = by_group_branch[(group, dp_branch)]
                if dna_mse > dp_mse:
                    dna_wins += 1
                    direction = "dna"
                elif dp_mse > dna_mse:
                    dp_wins += 1
                    direction = "dp"
                else:
                    ties += 1
                    direction = "tie"
                per_target.append({"group": group, "transform": transform_id, "variant": variant, "dna_mse": dna_mse, "dp_mse": dp_mse, "direction": direction})
            n = dna_wins + dp_wins
            tests[f"{transform_id}__{variant}"] = {
                "transform": transform_id,
                "variant": variant,
                "dna_wins": dna_wins,
                "dp_wins": dp_wins,
                "ties": ties,
                "non_tied_n": n,
                "p_dna_greater": exact_p(dna_wins, n),
                "p_dp_greater": exact_p(dp_wins, n),
                "per_target": per_target,
            }
    return tests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-target", type=Path, required=True)
    parser.add_argument("--confirm-target", type=Path, required=True)
    parser.add_argument("--utility-summary", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    utility = json.loads(args.utility_summary.read_text())
    config_path = ROOT / utility["config"]
    base_config = json.loads(config_path.read_text())
    config = {
        "clip_specs": base_config["clip_specs"],
        "selected_dp": utility["selections"],
    }
    pop_mean, std, features = load_population()

    dev = capture_rows(args.dev_target, args.seed, "dev")
    dev_rows, _dev_summary = score_branches(dev, pop_mean, std, config, args.seed)
    # Replay is deterministic for this closed-form computation.
    tie_band = 0.0

    confirm = capture_rows(args.confirm_target, args.seed, "confirm")
    confirm_rows, branch_summary = score_branches(confirm, pop_mean, std, config, args.seed)
    tests = compare(confirm_rows, config)
    comparison_rows = []
    for key, test in tests.items():
        comparison_rows.extend(test.pop("per_target"))
    write_csv(args.output_dir / "dev_t1_rows.csv", dev_rows)
    write_csv(args.output_dir / "confirm_t1_branch_rows.csv", confirm_rows)
    write_csv(args.output_dir / "confirm_t1_comparisons.csv", comparison_rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "dev_target": str(args.dev_target),
        "dev_target_sha256": sha256(args.dev_target),
        "confirm_target": str(args.confirm_target),
        "confirm_target_sha256": sha256(args.confirm_target),
        "utility_summary": str(args.utility_summary),
        "utility_summary_sha256": sha256(args.utility_summary),
        "feature_names": features,
        "tie_band": tie_band,
        "branch_summary": branch_summary,
        "tests": tests,
    }
    (args.output_dir / "c3_t1_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

