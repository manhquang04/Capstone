"""Priority 27 C1 gradient-only instrument gate.

Runs a TabLeak-style trainable-parameter attacker for one local-update setting
and one batch size.  The script is intentionally development/pilot gate only;
confirmatory RQ1 extension tests are handled by separate C3 scripts.
"""

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

from attacks.local_update import simulate, simulate_sgd  # noqa: E402
from data.load_creditcard import load_creditcard_data  # noqa: E402
from experiments import fraud_fl_common as common  # noqa: E402
from experiments.phase3_bounded_validation import _align_for_evaluation  # noqa: E402
from experiments.priority16_v1_medium_dna_vs_dp_probe import capture_paysim  # noqa: E402
from experiments.run_phase4_harddiff_reparam_for_misselected import (  # noqa: E402
    _decode_harddiff,
    _feature_distribution,
    _initial,
    _nonnegative_penalty,
)
from privacy.seed_manager import derive_seed  # noqa: E402


AMENDMENT = ROOT / "protocols/amendments/2026-09-29_priority27_gaps_and_harness.md"


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


def trainable_names(model: torch.nn.Module) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def filter_trainable(delta: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    return {name: delta[name] for name in names}


def filter_trainable_detached(delta: dict[str, torch.Tensor], names: list[str]) -> dict[str, torch.Tensor]:
    return {name: delta[name].detach().clone() for name in names}


def flatten(state: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    return torch.cat([state[name].reshape(-1) for name in names])


def cosine_objective(candidate: dict[str, torch.Tensor], observed: dict[str, torch.Tensor], names: list[str]) -> torch.Tensor:
    cand = flatten(candidate, names)
    obs = flatten(observed, names)
    return 1.0 - torch.nn.functional.cosine_similarity(cand, obs, dim=0, eps=1e-12)


def mse_std(candidate: torch.Tensor, original: torch.Tensor, std: np.ndarray) -> float:
    diff = (candidate.detach().cpu().numpy().astype(np.float64) - original.detach().cpu().numpy().astype(np.float64)) / std
    return float(np.mean(diff**2))


def exact_p(wins: int, n: int) -> float:
    return float(binomtest(wins, n, 0.5, alternative="greater").pvalue) if n else 1.0


def load_std() -> tuple[np.ndarray, list[str]]:
    loaders, *_rest, meta = load_creditcard_data(batch_size=1024, num_clients=3, seed=42, max_rows=500000)
    arrays = [loader.dataset.tensors[0].detach().cpu().numpy().astype(np.float64) for loader in loaders]
    train = np.concatenate(arrays, axis=0)
    std = train.std(axis=0)
    std[std < 1e-12] = 1.0
    return std, list(meta.feature_names)


def simulate_update(setting: str, model, criterion, x, labels, batches, rng, create_graph: bool = True):
    if setting == "adam":
        return simulate(model, criterion, x, labels, batches, rng, create_graph=create_graph)
    if setting == "sgd":
        return simulate_sgd(model, criterion, x, labels, batches, rng, lr=0.01, create_graph=create_graph)
    raise ValueError(setting)


def adam_sign_diagnostic(model, criterion, original, labels, batches, rng, observed: dict[str, torch.Tensor], names: list[str]) -> dict:
    params = dict(model.named_parameters())
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(rng)
        ids = batches[0]
        logits = torch.func.functional_call(model, (params, dict(model.named_buffers())), (original[ids],))
        grads = torch.autograd.grad(criterion(logits, labels[ids]), tuple(params.values()), create_graph=False)
    grad = torch.cat([g.detach().reshape(-1).double() for g in grads])
    step = flatten(observed, names).detach().double()
    sign_step = -0.001 * torch.sign(grad)
    cosine = float(torch.nn.functional.cosine_similarity(step, sign_step, dim=0, eps=1e-12))
    sign_agreement = float((torch.sign(step) == torch.sign(sign_step)).double().mean())
    g_abs = grad.abs().cpu().numpy()
    step_abs = step.abs().cpu().numpy()
    if np.std(g_abs) < 1e-18 or np.std(step_abs) < 1e-18:
        mag_corr = float("nan")
    else:
        mag_corr = float(np.corrcoef(g_abs, step_abs)[0, 1])
    return {
        "adam_step_vs_negative_lr_sign_grad_cosine": cosine,
        "adam_step_vs_negative_lr_sign_grad_sign_agreement": sign_agreement,
        "abs_step_abs_grad_correlation": mag_corr,
        "n_coordinates": int(step.numel()),
    }


def run_group(job: dict) -> dict:
    torch.set_num_threads(1)
    common.DEVICE = torch.device("cpu")
    groups = torch.load(Path(job["target"]), map_location="cpu", weights_only=False)
    group = groups[int(job["group"])]
    distribution = _feature_distribution()
    meta = distribution[0]
    local_seed = derive_seed(int(job["seed"]), "priority27-c1-local", job["setting"], int(job["batch_size"]), int(job["group"]))
    model, criterion, original, labels, batches, rng, observed_full = capture_paysim(group, local_seed, int(job["batch_size"]))
    names = trainable_names(model)
    if job["setting"] == "sgd":
        observed_full = simulate_sgd(model, criterion, original, labels, batches, rng, lr=0.01, create_graph=False)
    observed = filter_trainable_detached(observed_full, names)
    diagnostics = adam_sign_diagnostic(model, criterion, original, labels, batches, rng, observed, names) if job["setting"] == "adam" else {}

    best_value = float("inf")
    best = None
    restart_rows = []
    reconstructions = []
    for restart in range(int(job["restarts"])):
        initial = _initial(
            original.shape,
            derive_seed(int(job["seed"]), "priority27-c1-init", job["setting"], int(job["batch_size"]), int(job["group"]), restart),
            "standard",
            distribution,
        )
        latent = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent], lr=float(job["attacker_lr"]))
        final_loss = float("nan")
        for step in range(int(job["steps"])):
            reconstruction = _decode_harddiff(latent, meta)
            delta_full = simulate_update(job["setting"], model, criterion, reconstruction, labels, batches, rng)
            delta = filter_trainable(delta_full, names)
            loss = cosine_objective(delta, observed, names) + 0.001 * _nonnegative_penalty(latent, meta)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"nonfinite loss group={job['group']} restart={restart} step={step}")
            final_loss = float(loss.detach())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        reconstruction = _decode_harddiff(latent, meta).detach()
        delta_full = simulate_update(job["setting"], model, criterion, reconstruction, labels, batches, rng, create_graph=False)
        delta = filter_trainable(delta_full, names)
        objective = float(cosine_objective(delta, observed, names).detach())
        with torch.no_grad():
            aligned = _align_for_evaluation(original, reconstruction, labels)
            input_mse = mse_std(aligned, original, np.asarray(job["std"], dtype=np.float64))
            prior_aligned = _align_for_evaluation(original, _decode_harddiff(initial, meta), labels)
            prior_mse = mse_std(prior_aligned, original, np.asarray(job["std"], dtype=np.float64))
        reconstructions.append(aligned.detach().cpu())
        row = {
            "group": int(job["group"]),
            "restart": restart,
            "objective": objective,
            "input_mse": input_mse,
            "prior_mse": prior_mse,
            "final_loss": final_loss,
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
        "diagnostics": diagnostics,
        "restart_rows": restart_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--setting", choices=["adam", "sgd"], required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--attacker-lr", type=float, default=0.05)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    std, feature_names = load_std()
    groups = torch.load(args.target, map_location="cpu", weights_only=False)
    jobs = [
        {
            "target": str(args.target),
            "group": group_id,
            "seed": args.seed,
            "setting": args.setting,
            "batch_size": args.batch_size,
            "steps": args.steps,
            "restarts": args.restarts,
            "attacker_lr": args.attacker_lr,
            "std": std.tolist(),
        }
        for group_id in range(len(groups))
    ]
    rows = []
    restarts = []
    diagnostics = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_group, job): job for job in jobs}
        for future in as_completed(futures):
            row = future.result()
            rows.append(row)
            restarts.extend(
                {
                    **restart_row,
                    "setting": args.setting,
                    "batch_size": args.batch_size,
                }
                for restart_row in row.pop("restart_rows")
            )
            if row["diagnostics"]:
                diagnostics.append({"group": row["group"], **row["diagnostics"]})
            print(json.dumps({"group": row["group"], "selected_mse": row["selected_mse"], "prior_mse": row["prior_mse"]}))
    rows = sorted(rows, key=lambda item: item["group"])
    recon = [row["reconstruction"] for row in rows]
    for i, row in enumerate(rows):
        decoy = recon[(i + 1) % len(recon)]
        row["decoy_mse"] = mse_std(decoy, row["original"], std)
        row["win_vs_prior"] = row["selected_mse"] < row["prior_mse"]
        row["win_vs_decoy"] = row["selected_mse"] < row["decoy_mse"]
        row["source_ids_json"] = json.dumps(row["source_ids"])
        del row["source_ids"]
        del row["labels"]
        del row["original"]
        del row["reconstruction"]
        del row["diagnostics"]
    wins_prior = sum(bool(row["win_vs_prior"]) for row in rows)
    wins_decoy = sum(bool(row["win_vs_decoy"]) for row in rows)
    summary = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "amendment": str(AMENDMENT.relative_to(ROOT)),
        "amendment_sha256": sha256(AMENDMENT),
        "target": str(args.target),
        "target_sha256": sha256(args.target),
        "seed": args.seed,
        "setting": args.setting,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "restarts": args.restarts,
        "attacker_lr": args.attacker_lr,
        "workers": args.workers,
        "feature_names": feature_names,
        "n": len(rows),
        "wins_vs_prior": wins_prior,
        "p_vs_prior": exact_p(wins_prior, len(rows)),
        "wins_vs_decoy": wins_decoy,
        "p_vs_decoy": exact_p(wins_decoy, len(rows)),
        "qualified": exact_p(wins_prior, len(rows)) < 0.05 and exact_p(wins_decoy, len(rows)) < 0.05,
        "mean_selected_mse": float(np.mean([row["selected_mse"] for row in rows])),
        "median_selected_mse": float(np.median([row["selected_mse"] for row in rows])),
        "adam_sign_diagnostic_mean": {
            key: float(np.nanmean([item[key] for item in diagnostics]))
            for key in diagnostics[0]
            if key != "group"
        } if diagnostics else None,
    }
    write_csv(args.output_dir / "c1_gate_rows.csv", rows)
    write_csv(args.output_dir / "c1_restart_rows.csv", restarts)
    write_csv(args.output_dir / "adam_sign_diagnostic.csv", diagnostics)
    torch.save({"rows": rows, "diagnostics": diagnostics}, args.output_dir / "c1_gate_artifact.pt")
    (args.output_dir / "c1_gate_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

