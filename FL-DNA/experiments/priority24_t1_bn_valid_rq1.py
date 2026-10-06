"""Priority 24 valid-instrument RQ1 test for T1 BN-statistics channel."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.load_creditcard import load_creditcard_data  # noqa: E402
from dna_encoder.transform_defense import DNATransformConfig, transform_update_array  # noqa: E402
from dna_encoder.transform_defense_v2 import DNATransformV2Config, transform_and_reconstruct_array_v2  # noqa: E402
from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp  # noqa: E402
from experiments.priority16_v1_medium_dna_vs_dp_probe import capture_paysim  # noqa: E402
from experiments import fraud_fl_common as common  # noqa: E402
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority24_rq1_valid_instrument.md"
CLIP_SAFETY = 1.01
V1 = DNATransformConfig(block_size=256, mix_ratio=0.08, keep_ratio=0.88, shrink_factor=0.45, seed=681958327)
V2 = DNATransformV2Config(compression_ratio=0.95, quantization_eta=0.01, seed=20260916)
BN_KEY = "network.1.running_mean"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_p_greater(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def load_population(seed: int = 42) -> tuple[np.ndarray, np.ndarray, list[str]]:
    loaders, _, _, _, _, meta = load_creditcard_data(
        batch_size=1024,
        num_clients=3,
        seed=seed,
        max_rows=500000,
    )
    arrays = [loader.dataset.tensors[0].detach().cpu().numpy().astype(np.float64) for loader in loaders]
    train = np.concatenate(arrays, axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return train.mean(axis=0), std, list(meta.feature_names)


def transform_full_v1(update: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {}
    for tensor_index, (name, tensor) in enumerate(update.items()):
        if not torch.is_floating_point(tensor):
            out[name] = tensor.detach().cpu().clone()
            continue
        arr = tensor.detach().cpu().numpy().astype(np.float32, copy=False)
        transformed, _ = transform_update_array(arr, V1, tensor_index=tensor_index)
        out[name] = torch.from_numpy(transformed).to(dtype=tensor.dtype)
    return out


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
            quantization_seed=derive_seed(V2.seed, "priority24-v2", tensor_index),
        )
        out[name] = torch.from_numpy(reconstructed).to(dtype=tensor.dtype)
    return out


def recover_mean(model: torch.nn.Module, delta_rm: torch.Tensor) -> np.ndarray:
    linear = model.network[0]
    bn = model.network[1]
    momentum = float(bn.momentum if bn.momentum is not None else 0.1)
    rhs = bn.running_mean.detach().cpu().double() + delta_rm.detach().cpu().double() / momentum
    rhs = rhs - linear.bias.detach().cpu().double()
    sol = torch.linalg.lstsq(linear.weight.detach().cpu().double(), rhs).solution
    return sol.detach().cpu().numpy().astype(np.float64)


def mse_std(a: np.ndarray, b: np.ndarray, std: np.ndarray) -> float:
    return float(np.mean(((a - b) / std) ** 2))


def capture_groups(target: Path, seed: int, cell: str) -> list[dict]:
    groups = torch.load(target, map_location="cpu", weights_only=False)
    captures = []
    for group_id, group in enumerate(groups):
        local_seed = derive_seed(seed, "priority24-local", cell, group_id)
        model, criterion, x, y, batches, rng, observed = capture_paysim(group, local_seed, 4)
        captures.append(
            {
                "group_id": group_id,
                "source_ids": group["source_ids"],
                "model": model,
                "x": x.detach().cpu().numpy().astype(np.float64),
                "true_mean": x.detach().cpu().numpy().astype(np.float64).mean(axis=0),
                "observed": observed,
            }
        )
    return captures


def defended_bn_vectors(captures: list[dict], calibration: dict | None = None) -> dict[str, list[torch.Tensor]]:
    raw = [c["observed"][BN_KEY].detach().cpu() for c in captures]
    v1 = [transform_full_v1(c["observed"])[BN_KEY].detach().cpu() for c in captures]
    v2 = [transform_full_v2(c["observed"])[BN_KEY].detach().cpu() for c in captures]
    out = {"none": raw, "v1_conservative": v1, "v2_0p95_eta0p01": v2}
    if calibration is not None:
        for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
            clip = float(calibration[transform_id]["clip_norm"])
            mult = float(calibration[transform_id]["noise_multiplier"])
            dp = []
            for c in captures:
                vec = c["observed"][BN_KEY].detach().cpu().double()
                norm = float(torch.linalg.vector_norm(vec))
                clipped = vec * min(1.0, clip / max(norm, 1e-12))
                gen = torch.Generator().manual_seed(derive_seed(20260929, "priority24-dp", transform_id, c["group_id"]))
                noise = torch.randn(vec.shape, generator=gen, dtype=torch.float64) * (clip * mult)
                dp.append((clipped + noise).to(dtype=c["observed"][BN_KEY].dtype))
            out[f"dp_distortion_matched_for_{transform_id}"] = dp
    return out


def qualify(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray) -> dict:
    signals = defended_bn_vectors(captures)["none"]
    rec = [recover_mean(c["model"], signals[i]) for i, c in enumerate(captures)]
    rows = []
    for i, c in enumerate(captures):
        decoy = rec[(i + 1) % len(rec)]
        prior_mse = mse_std(pop_mean, c["true_mean"], std)
        decoy_mse = mse_std(decoy, c["true_mean"], std)
        recovered_mse = mse_std(rec[i], c["true_mean"], std)
        rows.append(
            {
                "group": c["group_id"],
                "source_ids": json.dumps(c["source_ids"]),
                "recovered_mse": recovered_mse,
                "prior_mse": prior_mse,
                "decoy_mse": decoy_mse,
                "win_vs_prior": recovered_mse < prior_mse,
                "win_vs_decoy": recovered_mse < decoy_mse,
            }
        )
    wins_prior = sum(r["win_vs_prior"] for r in rows)
    wins_decoy = sum(r["win_vs_decoy"] for r in rows)
    return {
        "rows": rows,
        "summary": {
            "n": len(rows),
            "wins_vs_prior": wins_prior,
            "p_vs_prior": exact_p_greater(wins_prior, len(rows)),
            "wins_vs_decoy": wins_decoy,
            "p_vs_decoy": exact_p_greater(wins_decoy, len(rows)),
            "qualified": exact_p_greater(wins_prior, len(rows)) < 0.05 and exact_p_greater(wins_decoy, len(rows)) < 0.05,
        },
    }


def calibrate(captures: list[dict]) -> dict:
    signals = defended_bn_vectors(captures)
    raw = signals["none"]
    norms = np.asarray([float(torch.linalg.vector_norm(v.double())) for v in raw], dtype=np.float64)
    clip_norm = float(max(np.max(norms) * CLIP_SAFETY, 1e-12))
    out = {}
    orders = alpha_grid()
    dim = int(raw[0].numel())
    for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
        dna_dist = np.asarray(
            [float(torch.linalg.vector_norm((signals[transform_id][i] - raw[i]).double())) for i in range(len(raw))],
            dtype=np.float64,
        )
        gaussian_norms = []
        for i, _ in enumerate(raw):
            gen = torch.Generator().manual_seed(derive_seed(20260929, "priority24-dp", transform_id, i))
            z = torch.randn(raw[i].shape, generator=gen, dtype=torch.float64)
            gaussian_norms.append(float(torch.linalg.vector_norm(z)))
        denom = clip_norm * float(np.median(gaussian_norms))
        multiplier = float(np.median(dna_dist) / max(denom, 1e-18))
        eps1 = epsilon_from_rdp(multiplier, 1.0, 1e-5, 1, orders)
        eps50 = epsilon_from_rdp(multiplier, 1.0, 1e-5, 50, orders)
        out[transform_id] = {
            "clip_norm": clip_norm,
            "noise_multiplier": multiplier,
            "bn_dim": dim,
            "median_bn_norm": float(np.median(norms)),
            "max_bn_norm": float(np.max(norms)),
            "median_dna_l2_distortion": float(np.median(dna_dist)),
            "epsilon_single_release_delta_1e_minus_5": eps1["epsilon"],
            "epsilon_50_releases_delta_1e_minus_5": eps50["epsilon"],
        }
    return out


def score_confirm(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray, calibration: dict, tie_band: float) -> dict:
    signals = defended_bn_vectors(captures, calibration=calibration)
    rec = {name: [recover_mean(c["model"], vecs[i]) for i, c in enumerate(captures)] for name, vecs in signals.items()}
    rows = []
    for i, c in enumerate(captures):
        base = {
            "group": c["group_id"],
            "source_ids": json.dumps(c["source_ids"]),
            "prior_mse": mse_std(pop_mean, c["true_mean"], std),
        }
        for branch in ("none", "v1_conservative", "v2_0p95_eta0p01", "dp_distortion_matched_for_v1_conservative", "dp_distortion_matched_for_v2_0p95_eta0p01"):
            base[f"{branch}_mse"] = mse_std(rec[branch][i], c["true_mean"], std)
            base[f"{branch}_decoy_mse"] = mse_std(rec[branch][(i + 1) % len(captures)], c["true_mean"], std)
        rows.append(base)
    tests = {}
    for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
        dp_id = f"dp_distortion_matched_for_{transform_id}"
        dna_wins = dp_wins = ties = 0
        ratios = []
        for row in rows:
            dna = row[f"{transform_id}_mse"]
            dp = row[f"{dp_id}_mse"]
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


def holm_adjust(pvalues: dict[str, float]) -> dict[str, float]:
    ordered = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(ordered)
    adjusted = {}
    running = 0.0
    for rank, (key, p) in enumerate(ordered, start=1):
        val = min(1.0, (m - rank + 1) * p)
        running = max(running, val)
        adjusted[key] = running
    return adjusted


def make_examples(confirm: dict, feature_names: list[str]) -> list[dict]:
    rows = []
    for row in confirm["rows"][:4]:
        rows.append({key: row[key] for key in row if key in {"group", "source_ids", "prior_mse", "none_mse", "v1_conservative_mse", "v2_0p95_eta0p01_mse", "dp_distortion_matched_for_v1_conservative_mse", "dp_distortion_matched_for_v2_0p95_eta0p01_mse"}})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-target", type=Path, required=True)
    parser.add_argument("--pilot-target", type=Path, required=True)
    parser.add_argument("--confirm-target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092901)
    args = parser.parse_args()
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=False)
    pop_mean, std, feature_names = load_population()
    dev = capture_groups(args.dev_target.resolve(), args.seed, "priority24_t1_dev")
    dev_q = qualify(dev, pop_mean, std)
    calibration = calibrate(dev)
    # Deterministic replay stability for D on development.
    dev_confirm_1 = score_confirm(dev, pop_mean, std, calibration, tie_band=0.0)
    dev_confirm_2 = score_confirm(dev, pop_mean, std, calibration, tie_band=0.0)
    max_replay = 0.0
    for r1, r2 in zip(dev_confirm_1["rows"], dev_confirm_2["rows"]):
        for transform_id in ("v1_conservative", "v2_0p95_eta0p01"):
            dp_id = f"dp_distortion_matched_for_{transform_id}"
            d1 = r1[f"{transform_id}_mse"] - r1[f"{dp_id}_mse"]
            d2 = r2[f"{transform_id}_mse"] - r2[f"{dp_id}_mse"]
            max_replay = max(max_replay, abs(d1 - d2))
    tie_band = float(max_replay)
    pilot_q = {"summary": {"qualified": False}, "rows": []}
    confirm = {"rows": [], "tests": {}}
    answer = "UNANSWERABLE"
    if dev_q["summary"]["qualified"]:
        pilot = capture_groups(args.pilot_target.resolve(), args.seed, "priority24_t1_pilot")
        pilot_q = qualify(pilot, pop_mean, std)
        if pilot_q["summary"]["qualified"]:
            confirmation = capture_groups(args.confirm_target.resolve(), args.seed, "priority24_t1_confirm")
            confirm = score_confirm(confirmation, pop_mean, std, calibration, tie_band=tie_band)
            pvals_dna = {key: value["dna_greater_p"] for key, value in confirm["tests"].items()}
            pvals_dp = {key: value["dp_greater_p"] for key, value in confirm["tests"].items()}
            holm_dna = holm_adjust(pvals_dna)
            holm_dp = holm_adjust(pvals_dp)
            for key, value in confirm["tests"].items():
                value["holm_dna_greater_p"] = holm_dna[key]
                value["holm_dp_greater_p"] = holm_dp[key]
            any_dna = any(v["holm_dna_greater_p"] < 0.05 for v in confirm["tests"].values())
            any_dp = any(v["holm_dp_greater_p"] < 0.05 for v in confirm["tests"].values())
            if any_dp:
                answer = "NO_DP_SIGNIFICANTLY_BEATS_DNA"
            elif any_dna:
                answer = "YES_DNA_SIGNIFICANTLY_BEATS_DP"
            else:
                answer = "NO_NOT_SHOWN"
    write_csv(out / "dev_t1_qualification.csv", dev_q["rows"])
    write_csv(out / "pilot_t1_qualification.csv", pilot_q["rows"])
    write_csv(out / "confirm_t1_per_target.csv", confirm["rows"])
    write_csv(out / "confirm_t1_examples.csv", make_examples(confirm, feature_names))
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "dev_target": str(args.dev_target.resolve().relative_to(ROOT)),
        "dev_target_sha256": sha256(args.dev_target.resolve()),
        "pilot_target": str(args.pilot_target.resolve().relative_to(ROOT)),
        "pilot_target_sha256": sha256(args.pilot_target.resolve()),
        "confirm_target": str(args.confirm_target.resolve().relative_to(ROOT)),
        "confirm_target_sha256": sha256(args.confirm_target.resolve()),
        "feature_names": feature_names,
        "instrument": "T1_BN_RUNNING_MEAN_BATCH_MEAN_RECOVERY",
        "dev_qualification": dev_q["summary"],
        "pilot_qualification": pilot_q["summary"],
        "calibration": calibration,
        "tie_band_from_development_replay": tie_band,
        "confirmatory_tests": confirm["tests"],
        "rq1_answer_reduced_plan": answer,
        "reduced_plan_note": "Only T1 fraud-domain distortion-matched tests were authorized in this reduced Priority 24 amendment.",
    }
    dump(out / "priority24_t1_summary.json", summary)
    manifest = {
        "summary_sha256": sha256(out / "priority24_t1_summary.json"),
        "dev_csv_sha256": sha256(out / "dev_t1_qualification.csv"),
        "pilot_csv_sha256": sha256(out / "pilot_t1_qualification.csv"),
        "confirm_csv_sha256": sha256(out / "confirm_t1_per_target.csv"),
        "examples_csv_sha256": sha256(out / "confirm_t1_examples.csv"),
    }
    dump(out / "manifest.json", manifest)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
