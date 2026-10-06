"""Priority 26 T1-only privacy evaluation for v2-SB."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dna_encoder.transform_defense_v2_server_blind import (  # noqa: E402
    ServerBlindV2Config,
    lift_raw_sketch_v2sb,
    lift_sketch_array_v2sb,
    sketch_update_array_v2sb,
)
from experiments.priority24_t1_bn_valid_rq1 import (  # noqa: E402
    BN_KEY,
    capture_groups,
    load_population,
    mse_std,
    recover_mean,
)
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority26_v2sb_server_blind.md"
CFG = ServerBlindV2Config(compression_ratio=0.95, quantization_eta=0.01)
V2_UTILITY_DP_CLIP = 259.0841131896973
V2_UTILITY_DP_SIGMA = 0.00003


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def exact_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def round_key(seed: int, group_id: int) -> bytes:
    payload = f"priority26-v2sb-t1|{seed}|{group_id}".encode()
    return hashlib.sha256(payload).digest()


def state_norm(update: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for tensor in update.values():
        if torch.is_floating_point(tensor):
            arr = tensor.detach().cpu().double().reshape(-1)
            total += float(torch.dot(arr, arr))
    return total**0.5


def sketch_bn(capture: dict, seed: int):
    names = list(capture["observed"].keys())
    tensor_index = names.index(BN_KEY)
    key = round_key(seed, capture["group_id"])
    bn = capture["observed"][BN_KEY].detach().cpu().numpy().astype(np.float32, copy=False)
    return sketch_update_array_v2sb(
        bn,
        CFG,
        round_key=key,
        round_number=1,
        tensor_index=tensor_index,
        client_id=0,
    ), key, tensor_index


def sketch_features(sketch: np.ndarray) -> np.ndarray:
    q = np.asarray(sketch, dtype=np.float64).reshape(-1)
    hist, _ = np.histogram(q, bins=16)
    stats = np.asarray(
        [
            np.linalg.norm(q),
            q.mean(),
            q.std(),
            np.mean(np.abs(q)),
            np.min(q),
            np.max(q),
            np.quantile(q, 0.25),
            np.quantile(q, 0.5),
            np.quantile(q, 0.75),
        ],
        dtype=np.float64,
    )
    return np.concatenate([stats, hist.astype(np.float64) / max(q.size, 1)])


def dp_distortion_calibration(captures: list[dict], seed: int) -> dict:
    raw = [c["observed"][BN_KEY].detach().cpu().double() for c in captures]
    norms = np.asarray([float(torch.linalg.vector_norm(v)) for v in raw])
    clip = float(np.quantile(norms, 0.95) * 1.01)
    distortions = []
    gaussian_norms = []
    for c in captures:
        sketch, key, _ = sketch_bn(c, seed)
        lifted = torch.from_numpy(lift_sketch_array_v2sb(sketch, round_key=key)).double()
        distortions.append(float(torch.linalg.vector_norm(lifted - c["observed"][BN_KEY].detach().cpu().double())))
        gen = torch.Generator().manual_seed(derive_seed(seed, "dp-distortion-calib", c["group_id"]))
        gaussian_norms.append(float(torch.linalg.vector_norm(torch.randn(raw[0].shape, generator=gen, dtype=torch.float64))))
    sigma = float(np.median(distortions) / max(clip * np.median(gaussian_norms), 1e-18))
    return {
        "clip_norm": clip,
        "noise_multiplier": sigma,
        "median_v2sb_lift_distortion": float(np.median(distortions)),
        "median_bn_norm": float(np.median(norms)),
        "bn_dim": int(raw[0].numel()),
    }


def dp_bn_signal(capture: dict, clip: float, sigma: float, seed: int, label: str) -> torch.Tensor:
    vec = capture["observed"][BN_KEY].detach().cpu().double()
    factor = min(1.0, clip / max(state_norm(capture["observed"]), 1e-12))
    gen = torch.Generator().manual_seed(derive_seed(seed, "dp", label, capture["group_id"]))
    noise = torch.randn(vec.shape, generator=gen, dtype=torch.float64) * (clip * sigma)
    return (vec * factor + noise).to(dtype=capture["observed"][BN_KEY].dtype)


def predictions(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray, seed: int, ridge_train=None) -> dict:
    rows = []
    features = []
    true = []
    for c in captures:
        sketch, key, tensor_index = sketch_bn(c, seed)
        q = sketch.sketch
        a1 = recover_mean(c["model"], torch.from_numpy(lift_sketch_array_v2sb(sketch, round_key=key)))
        padded = np.zeros(sketch.metadata.padded_size, dtype=np.float32)
        padded[: q.size] = q
        a2_identity = recover_mean(c["model"], torch.from_numpy(padded[: sketch.metadata.original_size]))
        wrong_key = round_key(seed + 99173, c["group_id"])
        a2_random = recover_mean(
            c["model"],
            torch.from_numpy(
                lift_raw_sketch_v2sb(
                    q,
                    original_shape=tuple(c["observed"][BN_KEY].shape),
                    config=CFG,
                    round_key=wrong_key,
                    round_number=1,
                    tensor_index=tensor_index,
                )
            ),
        )
        features.append(sketch_features(q))
        true.append(c["true_mean"])
        rows.append(
            {
                "group": c["group_id"],
                "source_ids": json.dumps(c["source_ids"]),
                "a1_seed_leak_mse": mse_std(a1, c["true_mean"], std),
                "a2_identity_mse": mse_std(a2_identity, c["true_mean"], std),
                "a2_random_surrogate_mse": mse_std(a2_random, c["true_mean"], std),
                "prior_mse": mse_std(pop_mean, c["true_mean"], std),
            }
        )
    x = np.vstack(features)
    y = np.vstack(true)
    if ridge_train is None:
        loo_pred = np.zeros_like(y)
        for train_idx, test_idx in LeaveOneOut().split(x):
            model = Ridge(alpha=1.0)
            model.fit(x[train_idx], y[train_idx])
            loo_pred[test_idx] = model.predict(x[test_idx])
        train_model = Ridge(alpha=1.0).fit(x, y)
        a3_pred = loo_pred
    else:
        train_model = ridge_train
        a3_pred = train_model.predict(x)
    for i, c in enumerate(captures):
        rows[i]["a3_invariant_ridge_mse"] = mse_std(a3_pred[i], c["true_mean"], std)
        rows[i]["a3_decoy_mse"] = mse_std(a3_pred[(i + 1) % len(captures)], c["true_mean"], std)
    return {"rows": rows, "ridge": train_model}


def gate(rows: list[dict], attack_key: str) -> dict:
    wins_prior = sum(r[attack_key] < r["prior_mse"] for r in rows)
    if attack_key == "a3_invariant_ridge_mse":
        wins_decoy = sum(r[attack_key] < r["a3_decoy_mse"] for r in rows)
    else:
        vals = [r[attack_key] for r in rows]
        wins_decoy = sum(vals[i] < vals[(i + 1) % len(vals)] for i in range(len(vals)))
    n = len(rows)
    return {
        "wins_vs_prior": int(wins_prior),
        "p_vs_prior": exact_p(wins_prior, n),
        "wins_vs_decoy": int(wins_decoy),
        "p_vs_decoy": exact_p(wins_decoy, n),
        "qualified": bool(exact_p(wins_prior, n) < 0.05 and exact_p(wins_decoy, n) < 0.05),
        "median_mse": float(np.median([r[attack_key] for r in rows])),
    }


def score_confirm(captures: list[dict], pop_mean: np.ndarray, std: np.ndarray, seed: int, ridge, attack: str, calib: dict) -> dict:
    base = predictions(captures, pop_mean, std, seed, ridge_train=ridge)
    rows = []
    wins = {"distortion": {"v2sb": 0, "dp": 0, "ties": 0}, "utility": {"v2sb": 0, "dp": 0, "ties": 0}}
    for i, c in enumerate(captures):
        row = dict(base["rows"][i])
        v2sb = row[attack]
        for label, clip, sigma in (
            ("distortion", calib["clip_norm"], calib["noise_multiplier"]),
            ("utility", V2_UTILITY_DP_CLIP, V2_UTILITY_DP_SIGMA),
        ):
            dp_signal = dp_bn_signal(c, clip, sigma, seed, label)
            rec = recover_mean(c["model"], dp_signal)
            dp_mse = mse_std(rec, c["true_mean"], std)
            row[f"dp_{label}_mse"] = dp_mse
            if v2sb > dp_mse:
                wins[label]["v2sb"] += 1
            elif dp_mse > v2sb:
                wins[label]["dp"] += 1
            else:
                wins[label]["ties"] += 1
        rows.append(row)
    tests = {}
    for label, counts in wins.items():
        n = counts["v2sb"] + counts["dp"]
        tests[label] = {
            "v2sb_wins": counts["v2sb"],
            "dp_wins": counts["dp"],
            "ties": counts["ties"],
            "non_tied_n": n,
            "v2sb_greater_p": exact_p(counts["v2sb"], n),
            "dp_greater_p": exact_p(counts["dp"], n),
        }
    return {"rows": rows, "tests": tests}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-target", type=Path, required=True)
    parser.add_argument("--confirm-target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026092604)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    pop_mean, std, feature_names = load_population()
    dev = capture_groups(args.dev_target.resolve(), args.seed, "priority26_v2sb_t1_dev")
    dev_pred = predictions(dev, pop_mean, std, args.seed)
    gates = {key: gate(dev_pred["rows"], key) for key in ("a1_seed_leak_mse", "a2_identity_mse", "a2_random_surrogate_mse", "a3_invariant_ridge_mse")}
    level1 = {key: value for key, value in gates.items() if key != "a1_seed_leak_mse" and value["qualified"]}
    selected = min(level1, key=lambda k: level1[k]["median_mse"]) if level1 else None
    calib = dp_distortion_calibration(dev, args.seed)
    confirm = {"rows": [], "tests": {}, "selected_level1_attack": selected}
    if selected is not None:
        conf = capture_groups(args.confirm_target.resolve(), args.seed, "priority26_v2sb_t1_confirm")
        confirm = score_confirm(conf, pop_mean, std, args.seed, dev_pred["ridge"], selected, calib)
        confirm["selected_level1_attack"] = selected
    write_csv(output / "dev_t1_v2sb_rows.csv", dev_pred["rows"])
    write_csv(output / "confirm_t1_v2sb_rows.csv", confirm["rows"])
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "dev_target": str(args.dev_target.resolve().relative_to(ROOT)),
        "dev_target_sha256": sha256(args.dev_target.resolve()),
        "confirm_target": str(args.confirm_target.resolve().relative_to(ROOT)),
        "confirm_target_sha256": sha256(args.confirm_target.resolve()),
        "feature_names": feature_names,
        "gates": gates,
        "selected_level1_attack": selected,
        "dp_distortion_calibration": calib,
        "dp_utility_config_from_priority25": {"clip_norm": V2_UTILITY_DP_CLIP, "noise_multiplier": V2_UTILITY_DP_SIGMA},
        "confirmatory": {"executed": selected is not None, **confirm},
    }
    (output / "priority26_t1_v2sb_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {
        "summary_sha256": sha256(output / "priority26_t1_v2sb_summary.json"),
        "dev_csv_sha256": sha256(output / "dev_t1_v2sb_rows.csv"),
        "confirm_csv_sha256": sha256(output / "confirm_t1_v2sb_rows.csv"),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
