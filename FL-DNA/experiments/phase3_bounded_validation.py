"""Prospective bounded-client update-inversion validation for Phase 3.

This validates an attack on a real one-step Adam client update.  It deliberately
does not claim inversion of the full 70k+ record clients used by utility runs.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

from attacks.local_update import simulate, simulate_sgd
from data.load_creditcard import BASE_FEATURE_COLUMNS, NUMERIC_COLUMNS, _build_features
from experiments import fraud_fl_common as common
from experiments.run_phase3_full_client import checksum, dump, score
from models.fraud_mlp import FraudMLP
from privacy.seed_manager import derive_seed, generate_run_seed


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "artifacts/phase3/full_20260908T143837588533Z"
FOLLOWUP = ROOT / "artifacts/phase3_followup/final_checks_v1"


def update_objective(candidate, observed, keys, reference=None, mode="equal_mse"):
    """Compare transmitted floating parameters and BN buffers."""
    if mode == "equal_mse":
        return torch.stack([(candidate[key] - observed[key]).square().mean() for key in keys]).mean()
    if mode == "balanced_tensor":
        terms = []
        for key in keys:
            candidate_value = candidate[key].reshape(-1)
            observed_value = observed[key].reshape(-1)
            if reference is None:
                reference_value = observed_value
            else:
                reference_value = reference[key].reshape(-1)
            scale = reference_value.square().sum().clamp_min(torch.finfo(candidate_value.dtype).tiny)
            if float(observed_value.square().sum().detach()) == 0.0:
                terms.append(candidate_value.square().sum() / scale)
            else:
                cosine = 1 - torch.nn.functional.cosine_similarity(
                    candidate_value, observed_value, dim=0, eps=1e-12
                )
                magnitude = (candidate_value - observed_value).square().sum() / scale
                terms.append(cosine + 0.1 * magnitude)
        return torch.stack(terms).mean()
    if mode != "cosine_magnitude":
        raise ValueError(f"Unknown objective mode: {mode}")
    flat_candidate=torch.cat([candidate[key].reshape(-1) for key in keys])
    flat_observed=torch.cat([observed[key].reshape(-1) for key in keys])
    scale_source=flat_observed if reference is None else torch.cat([reference[key].reshape(-1) for key in keys])
    scale=scale_source.square().sum().clamp_min(torch.finfo(flat_candidate.dtype).tiny)
    if float(flat_observed.square().sum().detach()) == 0.0:
        return flat_candidate.square().sum()
    cosine=1-torch.nn.functional.cosine_similarity(flat_candidate,flat_observed,dim=0,eps=1e-12)
    magnitude=(flat_candidate-flat_observed).square().sum()/scale
    return cosine + .1*magnitude


def _binomial_tail(wins, total):
    return sum(math.comb(total, k) for k in range(wins, total + 1)) / (2 ** total)


def _decode(latent):
    return torch.cat((latent[:, :8], latent[:, 8:].softmax(-1)), dim=1)


def _align_for_evaluation(original, candidate, labels):
    """Match unordered reconstructed records within known-label groups."""
    aligned=torch.empty_like(candidate)
    flat_labels=labels.reshape(-1)
    for label in flat_labels.unique(sorted=True):
        ids=torch.where(flat_labels==label)[0]
        costs=torch.cdist(original[ids].double(),candidate[ids].double()).square().cpu().numpy()
        rows,cols=linear_sum_assignment(costs)
        aligned[ids[torch.as_tensor(rows)]]=candidate[ids[torch.as_tensor(cols)]]
    return aligned


def _load_metadata():
    return SimpleNamespace(**json.loads((REFERENCE / "preprocessing.json").read_text()))


def _make_groups(output, count, seed, excluded_ids):
    metadata = _load_metadata()
    frame = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=BASE_FEATURE_COLUMNS + ["isFraud"])
    available = np.setdiff1d(np.arange(len(frame)), np.asarray(sorted(excluded_ids), dtype=np.int64))
    labels = frame.isFraud.to_numpy()
    fraud = available[labels[available] == 1]
    normal = available[labels[available] == 0]
    rng = np.random.default_rng(seed)
    selected_fraud = rng.choice(fraud, count, replace=False)
    selected_normal = rng.choice(normal, count * 3, replace=False).reshape(count, 3)
    groups = []
    for index in range(count):
        ids = np.concatenate(([selected_fraud[index]], selected_normal[index]))
        rng.shuffle(ids)
        part = frame.iloc[ids].copy()
        features = _build_features(part)
        scaled = (features[NUMERIC_COLUMNS].to_numpy() - np.asarray(metadata.numeric_center)) / np.asarray(metadata.numeric_scale)
        onehot = np.column_stack([(features.type.to_numpy() == value).astype(float) for value in metadata.type_categories])
        groups.append(dict(source_ids=ids.tolist(), x=np.hstack((scaled, onehot)).astype("float32"),
                           y=labels[ids].astype("float32").reshape(-1, 1)))
    torch.save(groups, output)
    return groups


def prepare(out, development_targets, confirmation_targets):
    out.mkdir(parents=True, exist_ok=False)
    seed = generate_run_seed()
    old_ids = set()
    for run_name in ("full_20260908T143837588533Z", "full_20260908T143843502722Z", "full_20260908T143850567575Z"):
        run = ROOT / "artifacts/phase3" / run_name
        client = int(run_name != "full_20260908T143837588533Z")
        # The exact old source IDs are not present in evaluator files; exclude both
        # previously documented 500k subsets below instead.
        del client
    labels = pd.read_csv(ROOT / "datasets/creditcard.csv", usecols=["isFraud"]).isFraud.to_numpy()
    from sklearn.model_selection import train_test_split
    ids = np.arange(len(labels))
    _, original = train_test_split(ids, test_size=500000, random_state=20260907, stratify=labels)
    heldout_seed = json.loads((FOLLOWUP / "heldout_provenance.json").read_text())["seed"]
    remaining = np.setdiff1d(ids, original)
    _, second = train_test_split(remaining, test_size=500000, random_state=heldout_seed, stratify=labels[remaining])
    old_ids.update(original.tolist()); old_ids.update(second.tolist())
    for previous in (ROOT / 'artifacts/phase3_bounded').glob('*/*targets.pt'):
        for group in torch.load(previous, weights_only=False):
            old_ids.update(group['source_ids'])
    dev = _make_groups(out / "development_targets.pt", development_targets, derive_seed(seed, "development-data"), old_ids)
    old_ids.update(i for group in dev for i in group["source_ids"])
    confirm = _make_groups(out / "confirmation_targets.pt", confirmation_targets, derive_seed(seed, "confirmation-data"), old_ids)
    assert not ({i for group in dev for i in group["source_ids"]} & {i for group in confirm for i in group["source_ids"]})
    protocol = dict(run_seed=seed, client_records=4, composition="1 fraud + 3 non-fraud",
                    local_optimizer="SGD", local_lr=.001, local_steps=1, local_epochs=1,
                    batch_size=4, model_mode="train", known_labels=True, known_order=True,
                    known_local_rng=True, observation="individual full model-state delta",
                    objective="cosine direction plus 0.1 normalized magnitude over parameters and floating BatchNorm buffers",
                    parameterization="numeric unconstrained; transaction type softmax simplex",
                    evaluation_alignment="Hungarian minimum squared distance within known-label groups; evaluation only",
                    development_targets=development_targets, confirmation_targets=confirmation_targets,
                    development_grid={"attack_lr":[.01,.05,.1], "iterations":[300]},
                    confirmation_iterations=300, confirmation_restarts=3,
                    selection="lowest development mean baseline-minus-own-prior MSE",
                    candidate_selection="minimum attacker objective; ground truth evaluation-only",
                    control_selection="prior averaged over all restarts; zero selects its own objective minimum; zero scale is constant 1",
                    statistics="one-sided sign test excludes exact ties; client group is unit; exploratory sequential development",
                    gate="paired one-sided exact sign p<0.05, negative mean and median MSE difference versus both prior and zero-update",
                    limitation="bounded one-batch SGD client-update validation, diagnostic-only; not full utility-client Adam inversion",
                    dataset_sha256=checksum(ROOT / "datasets/creditcard.csv"),
                    checkpoint_sha256=checksum(REFERENCE / "pre_local.pt"))
    dump(out / "protocol_lock.json", protocol)


def _capture(group, seed):
    x = torch.from_numpy(group["x"]); y = torch.from_numpy(group["y"])
    model = FraudMLP(x.shape[1]).train()
    model.load_state_dict(torch.load(REFERENCE / "pre_local.pt", weights_only=False))
    criterion = common.BinaryFocalLoss()
    rng = torch.Generator().manual_seed(seed).get_state()
    delta = simulate_sgd(model, criterion, x, y, [slice(0, len(x))], rng)
    return model, criterion, x, y, rng, delta


def _attack(group, group_id, restart, seed, attack_lr, iterations, output):
    model, criterion, original, labels, rng, observed = _capture(group, derive_seed(seed, "local", group_id))
    keys = [key for key, value in observed.items() if value.is_floating_point()]
    initial = torch.randn(original.shape, generator=torch.Generator().manual_seed(derive_seed(seed, "init", group_id, restart)))
    metadata = _load_metadata()
    prior_candidate=_decode(initial)
    prior_aligned=_align_for_evaluation(original,prior_candidate,labels)
    prior_metrics = score(original, prior_aligned, labels, metadata, output / "prior.csv")
    records = []
    for method in ("baseline", "zero_update"):
        signal = observed if method == "baseline" else {key: torch.zeros_like(value) for key, value in observed.items()}
        z = initial.clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=attack_lr)
        history=[]; best=float("inf"); candidate=None; best_step=0
        started=time.perf_counter()
        for step in range(iterations + 1):
            simulated = simulate_sgd(model, criterion, _decode(z), labels, [slice(0, len(z))], rng)
            loss = update_objective(simulated, signal, keys, reference=observed, mode="cosine_magnitude")
            value=float(loss.detach()); history.append(value)
            if value < best:
                best=value; candidate=z.detach().clone(); best_step=step
            if step < iterations:
                gradient, = torch.autograd.grad(loss, z)
                optimizer.zero_grad(); z.grad=gradient; optimizer.step()
        decoded=_decode(candidate)
        aligned=_align_for_evaluation(original,decoded,labels)
        metrics=score(original, aligned, labels, metadata, output / f"{method}.csv")
        artifact=dict(method=method, group_id=group_id, restart=restart, seed=seed, attack_lr=attack_lr,
                      iterations=iterations, best_step=best_step, best_objective=best, history=history,
                      original=original, labels=labels, source_ids=group["source_ids"], initial=initial,
                      latent=candidate, reconstruction=decoded, aligned_reconstruction=aligned,
                      observed=observed, keys=keys)
        torch.save(artifact, output / f"{method}.pt")
        loaded=torch.load(output / f"{method}.pt", weights_only=False)
        checked=update_objective(simulate_sgd(model, criterion, loaded["reconstruction"], labels,
                                          [slice(0,len(labels))], rng), signal, keys,
                                          reference=observed,mode="cosine_magnitude")
        torch.testing.assert_close(checked.detach(), torch.tensor(best), rtol=1e-5, atol=1e-11)
        records.append(dict(method=method, group_id=group_id, restart=restart, attack_lr=attack_lr,
                            best_step=best_step, objective=best, metrics=metrics, prior=prior_metrics,
                            elapsed_seconds=time.perf_counter()-started))
    assert torch.equal(torch.load(output/"baseline.pt",weights_only=False)["initial"],
                       torch.load(output/"zero_update.pt",weights_only=False)["initial"])
    dump(output / "results.json", records)
    return records


def development(out):
    protocol=json.loads((out/"protocol_lock.json").read_text()); groups=torch.load(out/"development_targets.pt",weights_only=False)
    all_records=[]
    for lr in protocol["development_grid"]["attack_lr"]:
        for group_id,group in enumerate(groups):
            folder=out/"development"/f"lr_{lr}"/f"group_{group_id}"; folder.mkdir(parents=True)
            all_records.extend(_attack(group,group_id,0,protocol["run_seed"],lr,300,folder))
    choices=[]
    for lr in protocol["development_grid"]["attack_lr"]:
        rows=[r for r in all_records if r["attack_lr"]==lr and r["method"]=="baseline"]
        choices.append(dict(attack_lr=lr, mean_delta=float(np.mean([r["metrics"]["mean_mse"]-r["prior"]["mean_mse"] for r in rows]))))
    selected=min(choices,key=lambda row:row["mean_delta"])
    dump(out/"development_summary.json",dict(candidates=choices,selected=selected,records=all_records))
    dump(out/"frozen.json",dict(attack_lr=selected["attack_lr"],iterations=protocol["confirmation_iterations"],
                                  restarts=protocol["confirmation_restarts"],frozen_at_unix=time.time(),
                                  protocol_sha256=checksum(out/"protocol_lock.json")))


def confirm(out):
    protocol=json.loads((out/"protocol_lock.json").read_text()); frozen=json.loads((out/"frozen.json").read_text())
    assert checksum(out/"protocol_lock.json")==frozen["protocol_sha256"]
    groups=torch.load(out/"confirmation_targets.pt",weights_only=False); all_records=[]
    for group_id,group in enumerate(groups):
        for restart in range(frozen["restarts"]):
            folder=out/"confirmation"/f"group_{group_id}"/f"restart_{restart}";folder.mkdir(parents=True)
            all_records.extend(_attack(group,group_id,restart,protocol["run_seed"],frozen["attack_lr"],frozen["iterations"],folder))
    dump(out/"confirmation_records.json",all_records)


def summarize(out):
    rows=json.loads((out/"confirmation_records.json").read_text())
    selected=[]
    for group_id in sorted({r["group_id"] for r in rows}):
        baseline=min((r for r in rows if r["group_id"]==group_id and r["method"]=="baseline"),key=lambda r:r["objective"])
        same=min((r for r in rows if r['group_id']==group_id and r['method']=='zero_update'),key=lambda r:r['objective'])
        prior=float(np.mean([r['prior']['mean_mse'] for r in rows if r['group_id']==group_id and r['method']=='baseline']))
        selected.append(dict(group_id=group_id,restart=baseline["restart"],baseline_mse=baseline["metrics"]["mean_mse"],
                             prior_mse=prior,zero_mse=same["metrics"]["mean_mse"],
                             baseline_minus_prior=baseline["metrics"]["mean_mse"]-prior,
                             baseline_minus_zero=baseline["metrics"]["mean_mse"]-same["metrics"]["mean_mse"]))
    result={}
    for control in ("prior","zero"):
        values=np.asarray([r[f"baseline_minus_{control}"] for r in selected])
        wins=int((values<0).sum())
        non_ties=int((values!=0).sum())
        probability=_binomial_tail(wins,non_ties) if non_ties else 1.0
        result[control]=dict(n=len(values),wins=wins,mean_difference=float(values.mean()),median_difference=float(np.median(values)),
                             non_ties=non_ties, ties=len(values)-non_ties, one_sided_sign_p=probability,
                             gate=bool(values.mean()<0 and np.median(values)<0 and probability<.05))
    result["bounded_evidence_gate_passed"]=result["prior"]["gate"] and result["zero"]["gate"]
    result["phase3_gate_passed"]=False
    result['limitation']='Exploratory bounded SGD evidence cannot close full-client Adam gate; sequential development p-values are descriptive.'
    result["scope"]="known-label/order/RNG, one-batch four-record SGD client update; parameters and BatchNorm buffers visible; diagnostic-only"
    dump(out/"bounded_validation_report.json",dict(summary=result,selected=selected,
          development=json.loads((out/"development_summary.json").read_text()),
          frozen=json.loads((out/"frozen.json").read_text())))
    with (out/"bounded_validation_summary.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(selected[0]));writer.writeheader();writer.writerows(selected)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser();parser.add_argument("stage",choices=("prepare","development","confirm","summarize"));parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--development-targets",type=int,default=4);parser.add_argument("--confirmation-targets",type=int,default=10)
    args=parser.parse_args();torch.set_num_threads(1);common.DEVICE=torch.device("cpu")
    if args.stage=="prepare": prepare(args.output,args.development_targets,args.confirmation_targets)
    elif args.stage=="development": development(args.output)
    elif args.stage=="confirm": confirm(args.output)
    else: summarize(args.output)


if __name__=="__main__": main()
