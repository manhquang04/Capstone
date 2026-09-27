"""Priority 20 multi-checkpoint DP-vs-DNA head-to-head replication.

This runner reuses the Priority 16 clean-vector v1-medium vs DP comparator but
loads an explicitly supplied pre-local checkpoint instead of the historical
shared Phase 3 checkpoint.  It supports creating independently initialized
warmup checkpoints, executing one 8-target head-to-head probe per checkpoint,
and aggregating checkpoint-level sign summaries.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate
from data.load_creditcard import load_creditcard_data
from data import load_creditcard as data_module
from dna_encoder.transform_defense import DNATransformConfig
from experiments import fraud_fl_common as common
from experiments.phase3_bounded_validation import _align_for_evaluation, update_objective
from experiments.run_phase3_full_client import checksum, dump, score
from experiments.run_phase4_dna_level1_forward_attack import (
    _apply_surrogate_realization_torch,
    _surrogate_plan_from_state,
    _transmit_observed,
)
from experiments.run_phase4_harddiff_reparam_for_misselected import (
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _max_balance_residual,
    _nonnegative_penalty,
)
from experiments.run_phase4_simple_defense_attack import (
    _apply_candidate_defense,
    _apply_observed_defense,
    _plan as simple_defense_plan,
)
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed


AMENDMENT = "protocols/amendments/2026-09-27_priority20_multi_checkpoint_head_to_head_replication.md"
CELL = "v1_medium_vs_dp_0p000315"
DNA_CONFIG = dict(block_size=256, mix_ratio=0.10, keep_ratio=0.85, shrink_factor=0.40, seed=681958327)
DP_CONFIG = dict(clip_norm=100.0, noise_multiplier=0.000315, mc_noise_samples=100, defense_seed=314159265)


def _sha(path: Path) -> str:
    return checksum(path)


def _seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))


def _trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def _filter_trainable(state: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    missing = [name for name in names if name not in state]
    if missing:
        raise KeyError(f"missing trainable keys: {missing[:5]}")
    return {name: state[name] for name in names}


def _mse(candidate: dict[str, torch.Tensor], signal: dict[str, torch.Tensor]) -> float:
    parts = [
        (candidate[key].detach().double() - signal[key].detach().double()).reshape(-1).square()
        for key, value in signal.items()
        if value.is_floating_point()
    ]
    return float(torch.cat(parts).mean())


def create_checkpoints(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    data_module.DEFAULT_NUM_WORKERS = 0
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": args.amendment,
        "data_seed": args.data_seed,
        "max_rows": args.max_rows,
        "checkpoint_seeds": args.checkpoint_seeds,
        "batch_size": 1024,
        "num_clients": 3,
        "local_epochs": 1,
        "optimizer": "Adam",
        "torch_num_threads": 1,
        "checkpoints": [],
    }

    for checkpoint_id, seed in enumerate(args.checkpoint_seeds):
        _seed_everything(seed)
        loaders, _, _, dim, pos_weight, metadata = load_creditcard_data(
            batch_size=1024,
            num_clients=3,
            max_rows=args.max_rows,
            seed=args.data_seed,
        )
        base = FraudMLP(dim).train()
        states = []
        sample_counts = []
        losses = []
        for client_id, loader in enumerate(loaders):
            local = copy.deepcopy(base)
            torch.manual_seed(derive_seed(seed, "warmup", client_id))
            loss = common.train_local_model(local, loader, pos_weight, local_epochs=1)
            states.append(copy.deepcopy(local.state_dict()))
            sample_counts.append(len(loader.dataset))
            losses.append(float(loss))
        base.load_state_dict(common.fed_avg(states, sample_counts))
        folder = output / f"ckpt_{checkpoint_id}"
        folder.mkdir()
        checkpoint_path = folder / "pre_local.pt"
        torch.save(base.state_dict(), checkpoint_path)
        metadata_path = folder / "preprocessing.json"
        dump(metadata_path, asdict(metadata))
        record = {
            "checkpoint_id": f"ckpt_{checkpoint_id}",
            "checkpoint_seed": int(seed),
            "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
            "checkpoint_sha256": _sha(checkpoint_path),
            "preprocessing_path": str(metadata_path.relative_to(ROOT)),
            "preprocessing_sha256": _sha(metadata_path),
            "dataset_sha256": _sha(ROOT / "datasets/creditcard.csv"),
            "client_sample_counts": sample_counts,
            "warmup_losses": losses,
        }
        dump(folder / "checkpoint_manifest.json", record)
        manifest["checkpoints"].append(record)
        print(json.dumps(record), flush=True)
    dump(output / "priority20_checkpoint_manifest.json", manifest)
    print(json.dumps({"manifest": str(output / "priority20_checkpoint_manifest.json")}, indent=2))


def _capture_with_checkpoint(group: dict, local_seed: int, batch_size: int, checkpoint_path: Path):
    x = torch.from_numpy(group["x"])
    y = torch.from_numpy(group["y"])
    batches = [slice(start, min(start + batch_size, len(x))) for start in range(0, len(x), batch_size)]
    model = FraudMLP(x.shape[1]).train()
    model.load_state_dict(torch.load(checkpoint_path, weights_only=False, map_location="cpu"))
    initial = copy.deepcopy(model.state_dict())
    criterion = common.BinaryFocalLoss()
    rng = torch.Generator().manual_seed(local_seed).get_state()
    observed = simulate(model, criterion, x, y, batches, rng)

    native = copy.deepcopy(model)
    optimizer = torch.optim.Adam(native.parameters(), lr=0.001)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        for ids in batches:
            optimizer.zero_grad()
            criterion(native(x[ids]), y[ids]).backward()
            optimizer.step()
    for key, value in native.state_dict().items():
        torch.testing.assert_close(observed[key], value - initial[key], atol=1e-5, rtol=5e-3)
    return model, criterion, x, y, batches, rng, observed


def _dna_payload(observed: dict[str, torch.Tensor], group: int, restart: int):
    config = DNATransformConfig(**DNA_CONFIG)
    transmitted = _transmit_observed(observed, config)
    plan = _surrogate_plan_from_state(observed, config, group, restart)
    return transmitted, plan


def _candidate_dna(delta: dict[str, torch.Tensor], plan):
    return _apply_surrogate_realization_torch(delta, plan)


def _dp_payload(observed: dict[str, torch.Tensor], group: int):
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    args = SimpleNamespace(
        target_rel_l2=0.1,
        clip_factor=0.95,
        clip_norm=float(DP_CONFIG["clip_norm"]),
        noise_multiplier=float(DP_CONFIG["noise_multiplier"]),
        defense_seed=int(DP_CONFIG["defense_seed"]),
        mc_noise_samples=int(DP_CONFIG["mc_noise_samples"]),
        topk_temperature=1e-3,
    )
    plan = simple_defense_plan("clipping_noise_mc", observed, keys, group, args)
    signal = _apply_observed_defense(observed, keys, "clipping_noise_mc", plan)
    return signal, (keys, plan)


def _candidate_dp(delta: dict[str, torch.Tensor], payload):
    keys, plan = payload
    return _apply_candidate_defense(delta, keys, "clipping_noise_mc", plan)


def _optimize_branch(
    job: dict,
    branch: str,
    signal: dict[str, torch.Tensor],
    payload,
    model,
    criterion,
    original,
    labels,
    batches,
    rng,
    trainable,
    distribution,
    folder: Path,
) -> dict:
    meta = distribution[0]
    initial = _initial(
        original.shape,
        derive_seed(job["seed"], "priority20-init", job["checkpoint_id"], branch, job["group"], job["restart"]),
        "standard",
        distribution,
    )
    latent = initial.clone().requires_grad_(True)
    optimizer = torch.optim.Adam([latent], lr=0.1)
    best = float("inf")
    best_step = 0
    best_latent = latent.detach().clone()
    history = []
    for step in range(601):
        reconstruction = _decode_harddiff(latent, meta)
        delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
        delta = _filter_trainable(delta_full, trainable)
        candidate = _candidate_dna(delta, payload) if branch == "dna" else _candidate_dp(delta, payload)
        penalty = 0.001 * _nonnegative_penalty(latent, meta)
        loss = update_objective(candidate, signal, list(signal), reference=signal, mode="balanced_tensor") + penalty
        value = float(loss.detach())
        history.append(value)
        if value < best:
            best = value
            best_step = step
            best_latent = latent.detach().clone()
        if step < 600:
            (gradient,) = torch.autograd.grad(loss, latent)
            optimizer.zero_grad()
            latent.grad = gradient
            optimizer.step()
    reconstruction = _decode_harddiff(best_latent, meta).detach()
    aligned = _align_for_evaluation(original, reconstruction, labels)
    delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
    delta = _filter_trainable(delta_full, trainable)
    candidate = _candidate_dna(delta, payload) if branch == "dna" else _candidate_dp(delta, payload)
    mse = _mse(candidate, signal)
    metrics = score(original, aligned, labels, meta, folder / f"{branch}.csv")
    artifact = {
        "method": branch,
        "cell": CELL,
        "checkpoint_id": job["checkpoint_id"],
        "group_id": job["group"],
        "restart": job["restart"],
        "source_ids": job["source_ids"],
        "original": original,
        "labels": labels,
        "initial": initial,
        "reconstruction": reconstruction,
        "aligned": aligned,
        "best_objective": best,
        "best_step": best_step,
        "history": history,
        "clean_vector_mse": mse,
        "vector_scope": "trainable_parameters_only",
        "trainable_keys": trainable,
        "max_balance_residual": _max_balance_residual(reconstruction, meta),
        "feature_metrics": metrics,
    }
    torch.save(artifact, folder / f"{branch}.pt")
    return artifact


def _job(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    groups = torch.load(job["target"], map_location="cpu", weights_only=False)
    group = groups[job["group"]]
    distribution = _feature_distribution()
    local_seed = derive_seed(job["seed"], "priority20-local", job["checkpoint_id"], job["group"])
    model, criterion, original, labels, batches, rng, observed_full = _capture_with_checkpoint(
        group, local_seed, 4, Path(job["checkpoint"])
    )
    trainable = _trainable_names(model)
    observed = _filter_trainable(observed_full, trainable)
    dna_signal, dna_payload = _dna_payload(observed, job["group"], job["restart"])
    dp_signal, dp_payload = _dp_payload(observed, job["group"])
    folder = Path(job["folder"])
    folder.mkdir(parents=True, exist_ok=True)
    context = {
        "seed": job["seed"],
        "checkpoint_id": job["checkpoint_id"],
        "group": job["group"],
        "restart": job["restart"],
        "source_ids": group["source_ids"],
    }
    dna = _optimize_branch(context, "dna", dna_signal, dna_payload, model, criterion, original, labels, batches, rng, trainable, distribution, folder)
    dp = _optimize_branch(context, "dp", dp_signal, dp_payload, model, criterion, original, labels, batches, rng, trainable, distribution, folder)
    result = {
        "cell": CELL,
        "checkpoint_id": job["checkpoint_id"],
        "group": job["group"],
        "restart": job["restart"],
        "source_ids": group["source_ids"],
        "dna_mse": dna["clean_vector_mse"],
        "dp_mse": dp["clean_vector_mse"],
        "D_dna_minus_dp": dna["clean_vector_mse"] - dp["clean_vector_mse"],
        "status": "SUCCESS",
    }
    dump(folder / "result.json", result)
    return result


def _summarize(records: list[dict]) -> dict:
    selected = []
    for group in sorted({int(row["group"]) for row in records}):
        rows = [row for row in records if int(row["group"]) == group]
        dna_best = min(rows, key=lambda row: row["dna_mse"])
        dp_best = min(rows, key=lambda row: row["dp_mse"])
        D = float(dna_best["dna_mse"] - dp_best["dp_mse"])
        selected.append(
            {
                "group_id": int(group),
                "dna_restart": int(dna_best["restart"]),
                "dp_restart": int(dp_best["restart"]),
                "dna_mse": float(dna_best["dna_mse"]),
                "dp_mse": float(dp_best["dp_mse"]),
                "D_dna_minus_dp": D,
                "winner": "DNA" if D > 0 else ("DP" if D < 0 else "TIE"),
            }
        )
    ds = np.asarray([row["D_dna_minus_dp"] for row in selected], dtype=float)
    dna_wins = int((ds > 0).sum())
    dp_wins = int((ds < 0).sum())
    ties = int((ds == 0).sum())
    non_tied = dna_wins + dp_wins
    return {
        "selected": selected,
        "zero_threshold": {
            "dna_wins": dna_wins,
            "dp_wins": dp_wins,
            "exact_ties": ties,
            "non_tied_n": non_tied,
            "dp_advantage_p": float(binomtest(dp_wins, non_tied, 0.5, alternative="greater").pvalue) if non_tied else 1.0,
            "dna_advantage_p": float(binomtest(dna_wins, non_tied, 0.5, alternative="greater").pvalue) if non_tied else 1.0,
            "mean_D_dna_minus_dp": float(ds.mean()) if len(ds) else float("nan"),
            "sd_D_dna_minus_dp": float(ds.std(ddof=1)) if len(ds) > 1 else 0.0,
            "median_D_dna_minus_dp": float(np.median(ds)) if len(ds) else float("nan"),
            "min_D_dna_minus_dp": float(ds.min()) if len(ds) else float("nan"),
            "max_D_dna_minus_dp": float(ds.max()) if len(ds) else float("nan"),
            "dp_won_every_target": bool(dp_wins == len(selected) and ties == 0),
            "dp_strict_majority": bool(dp_wins > dna_wins),
            "mean_D_negative": bool(float(ds.mean()) < 0) if len(ds) else False,
        },
    }


def execute(args: argparse.Namespace) -> None:
    torch.set_num_threads(1)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": args.amendment,
        "cell": CELL,
        "checkpoint_id": args.checkpoint_id,
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": _sha(args.checkpoint.resolve()),
        "target": str(args.target),
        "target_sha256": _sha(args.target.resolve()),
        "groups": args.groups,
        "restarts": args.restarts,
        "jobs": args.groups * args.restarts,
        "workers": args.workers,
        "torch_num_threads": 1,
        "vector_scope": "trainable_parameters_only",
        "attacker": "simple_balanced_tensor_raw_lift_clean_vector",
        "dp_config": DP_CONFIG,
        "dna_config": DNA_CONFIG,
    }
    dump(output / "execution_manifest.json", manifest)
    jobs = []
    for group in range(args.groups):
        for restart in range(args.restarts):
            jobs.append(
                {
                    "checkpoint_id": args.checkpoint_id,
                    "checkpoint": str(args.checkpoint.resolve()),
                    "group": group,
                    "restart": restart,
                    "seed": args.seed,
                    "target": str(args.target.resolve()),
                    "folder": str(output / CELL / f"group_{group}" / f"restart_{restart}"),
                }
            )
    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_job, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            records.append(row)
            print(json.dumps({key: row[key] for key in ("checkpoint_id", "group", "restart", "status")}), flush=True)
    summary = {
        **manifest,
        "completed_jobs": len(records),
        "records": sorted(records, key=lambda r: (r["group"], r["restart"])),
        "summary": _summarize(records),
    }
    dump(output / "priority20_probe_report.json", summary)
    print(json.dumps(summary["summary"]["zero_threshold"], indent=2))


def analyze(args: argparse.Namespace) -> None:
    reports = [json.loads(path.read_text()) for path in args.reports]
    rows = []
    all_selected = []
    for report in reports:
        z = report["summary"]["zero_threshold"]
        rows.append(
            {
                "checkpoint_id": report["checkpoint_id"],
                "targets": report["groups"],
                "dp_wins": z["dp_wins"],
                "dna_wins": z["dna_wins"],
                "ties": z["exact_ties"],
                "mean_D": z["mean_D_dna_minus_dp"],
                "sd_D": z["sd_D_dna_minus_dp"],
                "median_D": z["median_D_dna_minus_dp"],
                "dp_won_every_target": z["dp_won_every_target"],
                "dp_strict_majority": z["dp_strict_majority"],
                "mean_D_negative": z["mean_D_negative"],
                "report": str(Path(report.get("report_path", "")).as_posix()),
            }
        )
        for item in report["summary"]["selected"]:
            all_selected.append({**item, "checkpoint_id": report["checkpoint_id"]})
    ckpt_n = len(rows)
    majority_success = sum(row["dp_strict_majority"] for row in rows)
    mean_success = sum(row["mean_D_negative"] for row in rows)
    aggregate_dp = sum(row["dp_wins"] for row in rows)
    aggregate_dna = sum(row["dna_wins"] for row in rows)
    aggregate_ties = sum(row["ties"] for row in rows)
    aggregate_non_tied = aggregate_dp + aggregate_dna
    output = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": args.amendment,
        "cell": CELL,
        "reports": [str(path) for path in args.reports],
        "per_checkpoint": rows,
        "checkpoint_level": {
            "checkpoints": ckpt_n,
            "dp_majority_checkpoints": majority_success,
            "dp_majority_p": float(binomtest(majority_success, ckpt_n, 0.5, alternative="greater").pvalue) if ckpt_n else 1.0,
            "mean_D_negative_checkpoints": mean_success,
            "mean_D_negative_p": float(binomtest(mean_success, ckpt_n, 0.5, alternative="greater").pvalue) if ckpt_n else 1.0,
            "dp_won_every_target_every_checkpoint": bool(all(row["dp_won_every_target"] for row in rows)),
        },
        "aggregate_targetwise_descriptive": {
            "dp_wins": aggregate_dp,
            "dna_wins": aggregate_dna,
            "ties": aggregate_ties,
            "non_tied_n": aggregate_non_tied,
            "dp_advantage_p_if_targetwise_independent": float(binomtest(aggregate_dp, aggregate_non_tied, 0.5, alternative="greater").pvalue) if aggregate_non_tied else 1.0,
        },
        "selected": all_selected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump(args.output, output)
    print(json.dumps(output["checkpoint_level"], indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    ckpt = sub.add_parser("create-checkpoints")
    ckpt.add_argument("--output", type=Path, required=True)
    ckpt.add_argument("--checkpoint-seeds", type=int, nargs="+", required=True)
    ckpt.add_argument("--data-seed", type=int, default=20260907)
    ckpt.add_argument("--max-rows", type=int, default=500000)
    ckpt.add_argument("--amendment", default=AMENDMENT)

    run = sub.add_parser("execute")
    run.add_argument("--checkpoint-id", required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--target", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--seed", type=int, required=True)
    run.add_argument("--groups", type=int, default=8)
    run.add_argument("--restarts", type=int, default=4)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--amendment", default=AMENDMENT)

    ana = sub.add_parser("analyze")
    ana.add_argument("--reports", type=Path, nargs="+", required=True)
    ana.add_argument("--output", type=Path, required=True)
    ana.add_argument("--amendment", default=AMENDMENT)

    args = parser.parse_args()
    if args.command == "create-checkpoints":
        create_checkpoints(args)
    elif args.command == "execute":
        if not 1 <= args.workers <= 9:
            raise ValueError("--workers must be in [1, 9]")
        execute(args)
    else:
        analyze(args)


if __name__ == "__main__":
    main()
