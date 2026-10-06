"""Priority 25b T2 gradient-only gate attempt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.local_update import simulate  # noqa: E402
from experiments import fraud_fl_common as common  # noqa: E402
from experiments.phase3_bounded_validation import _align_for_evaluation  # noqa: E402
from experiments.priority24_t1_bn_valid_rq1 import load_population  # noqa: E402
from experiments.priority16_v1_medium_dna_vs_dp_probe import capture_paysim  # noqa: E402
from experiments.run_phase4_harddiff_reparam_for_misselected import (  # noqa: E402
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _max_balance_residual,
    _nonnegative_penalty,
)
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority25b_bracketed_utility.md"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def filter_trainable(delta: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    return {name: delta[name] for name in names}


def filter_trainable_detached(delta: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    return {name: delta[name].detach().clone() for name in names}


def cosine_objective(candidate: dict[str, torch.Tensor], observed: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    cand = torch.cat([candidate[name].reshape(-1) for name in names])
    obs = torch.cat([observed[name].reshape(-1) for name in names])
    return 1.0 - torch.nn.functional.cosine_similarity(cand, obs, dim=0, eps=1e-12)


def mse_std(candidate: torch.Tensor, original: torch.Tensor, std: np.ndarray) -> float:
    diff = (candidate.detach().cpu().numpy().astype(np.float64) - original.detach().cpu().numpy().astype(np.float64)) / std
    return float(np.mean(diff**2))


def run_group(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    target = Path(job["target"])
    groups = torch.load(target, map_location="cpu", weights_only=False)
    group = groups[int(job["group"])]
    distribution = _feature_distribution()
    meta = distribution[0]
    local_seed = derive_seed(int(job["seed"]), "priority25b-t2-local", int(job["group"]))
    model, criterion, original, labels, batches, rng, observed_full = capture_paysim(group, local_seed, 4)
    names = trainable_names(model)
    observed = filter_trainable_detached(observed_full, names)
    best_value = float("inf")
    best = None
    restart_rows = []
    reconstructions = []
    initials = []
    for restart in range(int(job["restarts"])):
        initial = _initial(
            original.shape,
            derive_seed(int(job["seed"]), "priority25b-t2-init", int(job["group"]), restart),
            "standard",
            distribution,
        )
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=float(job["attacker_lr"]))
        history = []
        for step in range(int(job["steps"])):
            reconstruction = _decode_harddiff(latent, meta)
            delta_full = simulate(model, criterion, reconstruction, labels, batches, rng)
            delta = filter_trainable(delta_full, names)
            loss = cosine_objective(delta, observed, names) + 0.001 * _nonnegative_penalty(latent, meta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite T2 loss group={job['group']} restart={restart} step={step}")
            history.append(float(loss.detach()))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        reconstruction = _decode_harddiff(latent, meta).detach()
        delta_full = simulate(model, criterion, reconstruction, labels, batches, rng, create_graph=False)
        delta = filter_trainable(delta_full, names)
        objective = float(cosine_objective(delta, observed, names).detach())
        with torch.no_grad():
            aligned = _align_for_evaluation(original, reconstruction, labels)
            input_mse = mse_std(aligned, original, np.asarray(job["std"], dtype=np.float64))
            prior_aligned = _align_for_evaluation(original, _decode_harddiff(initial, meta), labels)
            prior_mse = mse_std(prior_aligned, original, np.asarray(job["std"], dtype=np.float64))
        reconstructions.append(aligned.detach().cpu())
        initials.append(prior_aligned.detach().cpu())
        row = {
            "group": int(job["group"]),
            "restart": restart,
            "objective": objective,
            "input_mse": input_mse,
            "prior_mse": prior_mse,
            "max_balance_residual": _max_balance_residual(reconstruction, meta),
            "final_loss": history[-1],
        }
        restart_rows.append(row)
        if objective < best_value:
            best_value = objective
            best = {
                "restart": restart,
                "objective": objective,
                "input_mse": input_mse,
                "prior_mse": prior_mse,
                "reconstruction": aligned.detach().cpu(),
                "prior": prior_aligned.detach().cpu(),
                "history": history,
            }
    assert best is not None
    pooled = torch.median(torch.stack(reconstructions), dim=0).values
    pooled_mse = mse_std(pooled, original, np.asarray(job["std"], dtype=np.float64))
    return {
        "group": int(job["group"]),
        "source_ids": group["source_ids"],
        "labels": labels.detach().cpu(),
        "original": original.detach().cpu(),
        "selected_restart": best["restart"],
        "selected_objective": best["objective"],
        "selected_mse": best["input_mse"],
        "prior_mse": best["prior_mse"],
        "pooled_median_mse": pooled_mse,
        "reconstruction": best["reconstruction"],
        "prior": best["prior"],
        "restart_rows": restart_rows,
    }


def exact_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--attacker-lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    groups = torch.load(args.target.resolve(), map_location="cpu", weights_only=False)
    _pop_mean, std, features = load_population()
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(args.target.resolve().relative_to(ROOT)),
        "target_sha256": sha256(args.target.resolve()),
        "seed": args.seed,
        "steps": args.steps,
        "restarts": args.restarts,
        "attacker_lr": args.attacker_lr,
        "workers": args.workers,
    }
    (output / "execution_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    jobs = [
        {
            "target": str(args.target.resolve()),
            "group": group_id,
            "seed": args.seed,
            "steps": args.steps,
            "restarts": args.restarts,
            "attacker_lr": args.attacker_lr,
            "std": std.tolist(),
        }
        for group_id in range(len(groups))
    ]
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_group, job) for job in jobs]
        for future in as_completed(futures):
            row = future.result()
            results.append(row)
            print(json.dumps({"group": row["group"], "selected_mse": row["selected_mse"], "prior_mse": row["prior_mse"]}), flush=True)
    results = sorted(results, key=lambda row: row["group"])
    rows = []
    restart_rows = []
    selected_mses = [row["selected_mse"] for row in results]
    for index, row in enumerate(results):
        decoy = results[(index + 1) % len(results)]["reconstruction"]
        decoy_mse = mse_std(decoy, row["original"], std)
        rows.append(
            {
                "group": row["group"],
                "source_ids": json.dumps(row["source_ids"]),
                "selected_restart": row["selected_restart"],
                "selected_objective": row["selected_objective"],
                "selected_mse": row["selected_mse"],
                "prior_mse": row["prior_mse"],
                "decoy_mse": decoy_mse,
                "pooled_median_mse": row["pooled_median_mse"],
                "win_vs_prior": row["selected_mse"] < row["prior_mse"],
                "win_vs_decoy": row["selected_mse"] < decoy_mse,
            }
        )
        for rr in row["restart_rows"]:
            restart_rows.append(rr)
    wins_prior = sum(row["win_vs_prior"] for row in rows)
    wins_decoy = sum(row["win_vs_decoy"] for row in rows)
    summary = {
        **manifest,
        "feature_names": features,
        "n": len(rows),
        "wins_vs_prior": int(wins_prior),
        "p_vs_prior": exact_p(wins_prior, len(rows)),
        "wins_vs_decoy": int(wins_decoy),
        "p_vs_decoy": exact_p(wins_decoy, len(rows)),
        "qualified": bool(exact_p(wins_prior, len(rows)) < 0.05 and exact_p(wins_decoy, len(rows)) < 0.05),
        "mean_selected_mse": float(np.mean(selected_mses)),
        "median_selected_mse": float(np.median(selected_mses)),
    }
    write_csv(output / "t2_gate_rows.csv", rows)
    write_csv(output / "t2_restart_rows.csv", restart_rows)
    torch.save(
        {
            "rows": rows,
            "summary": summary,
            "reconstructions": [row["reconstruction"] for row in results],
            "originals": [row["original"] for row in results],
        },
        output / "t2_gate_artifact.pt",
    )
    (output / "t2_gate_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
