"""Independent reload of immutable qualification and descriptive diagnostic evidence."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
from collections import OrderedDict
import math
import numpy as np
import torch
from experiments.priority35a_runtime import ROOT, Q, sha, load, check_hashes
from experiments.priority35a_verify import independent_quality, compare_metrics, independent_gate
from experiments.priority35a_firewall import DATASETS, validate
from experiments.priority35a_checkpoints import PREPARED, arrays_partitions, exact_states
from experiments.priority35a_evaluation import reference_statistics, marginal
from experiments.priority35a_qualification import cells, cell_id, stage_targets, target_id, valid_receipt, client_step
from experiments.priority35a_estimator import schema_blocks, candidate_rows, observable, objective, flatten
from experiments.priority35a_nuisance import estimator_model
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_labels import canonical_labels
from experiments.priority35a_diagnostic import D, frozen, result_receipt, selected_cells, no_live_audit
from experiments.priority35a_diagnostic_estimator import oracle_model, operator
from experiments import priority34a_local_bn as old


def metric_audit(rows, saved_rows, medians):
    if len(rows) != len(saved_rows):
        raise ValueError("missing independent metric units")
    for calculated, saved in zip(rows, saved_rows):
        if calculated["index"] != saved["index"]:
            raise ValueError("target ordering mismatch")
        for arm in ("estimator", "mean_mode", "empirical_marginal", "decoy"):
            compare_metrics(saved[arm], calculated[arm])
    for arm in medians:
        for metric in medians[arm]:
            expected = float(np.median([r[arm][metric] for r in rows]))
            if not math.isclose(medians[arm][metric], expected, rel_tol=1e-12, abs_tol=1e-14):
                raise ValueError("independent median mismatch")
    return independent_gate(rows, len(rows))


def reload_metrics(bundles, guesses, stats):
    rows = []
    for i, bundle in enumerate(bundles):
        truth = bundle["features"].numpy()
        prior = independent_quality(truth, bundle["mean_mode"], stats)
        mse = prior["standardized_mse"]
        rows.append(dict(index=i, mean_mode=prior, estimator=independent_quality(truth, guesses[i], stats, mse),
                         empirical_marginal=independent_quality(truth, bundle["empirical_marginal"], stats, mse),
                         decoy=independent_quality(truth, guesses[(i+1) % len(guesses)], stats, mse)))
    return rows


def budget_check(recon, oracle=False):
    tensors = dict(rows=recon["rows"], raw=recon["raw"], labels=recon["labels"],
                   **recon["oracle_affine" if oracle else "nuisance"])
    require_finite_cpu(tensors, "independent reconstructed tensors")
    if recon["synthetic"] or recon["torch_threads"] != 1 or len(recon["start_traces"]) != 2:
        raise ValueError("scientific thread/start contract")
    if any(t[-1]["iteration"] != 4000 for t in recon["start_traces"]):
        raise ValueError("scientific iteration contract")
    if any(not math.isfinite(v) for v in recon["start_objectives"]) or not math.isfinite(recon["final_objective"]):
        raise ValueError("nonfinite start/objective")
    if recon["chosen_start"] != min(range(2), key=lambda i: (recon["start_objectives"][i], i)):
        raise ValueError("observable start selection changed")
    if not torch.equal(recon["labels"], canonical_labels(len(recon["labels"]), int(recon["labels"].sum()))):
        raise ValueError("unordered label contract")


def independently_capture(public, private):
    """Separate native autograd capture, not the production backward routine."""
    with torch.random.fork_rng(devices=[]):
        model = old.p.FraudMLP(private["features"].shape[1]).cpu()
        state = model.state_dict()
        state.update(private["client_bn"])
        state.update(public["shared"])
        model.load_state_dict(state, strict=True)
        model.train()
        torch.manual_seed(private["dropout_seed"])
        prediction = model(private["features"])
        require_finite_cpu(model.state_dict(), "independent actual gradient forward")
        loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(prediction, private["labels"])
        if not torch.isfinite(loss):
            raise ValueError("independent gradient focal nonfinite")
        params = OrderedDict((k, p) for k, p in model.named_parameters() if k in public["shared"])
        grad = OrderedDict(zip(params, [v.detach() for v in torch.autograd.grad(loss, tuple(params.values()))]))
    require_finite_cpu(grad, "independent actual gradients")
    return grad


def cosine_audit(saved, gradients, upload):
    # Separate NumPy implementation, retaining the prescribed eps clamp.
    g = flatten(gradients).numpy().astype(np.float64)
    delta = flatten(upload).numpy().astype(np.float64)
    sign = np.sign(g)
    def cos(value):
        return float(np.dot(value, sign) / (max(np.linalg.norm(value), 1e-12)*max(np.linalg.norm(sign), 1e-12)))
    expected = dict(gradient_vs_sign=cos(g), adam_vs_sign=cos(delta), descent_aligned_adam_vs_sign=cos(-delta),
                    zero_gradient_norm=bool(np.linalg.norm(g) == 0), zero_upload_norm=bool(np.linalg.norm(delta) == 0))
    if set(saved) != set(expected):
        raise ValueError("cosine schema")
    for k, value in expected.items():
        if isinstance(value, bool):
            if saved[k] != value:
                raise ValueError("cosine zero flag")
        elif not math.isfinite(saved[k]) or not math.isclose(saved[k], value, rel_tol=2e-6, abs_tol=2e-6):
            # Float32 production versus float64 dot audit; this is not a gate tolerance.
            raise ValueError("independent cosine mismatch")


def verify_measurements():
    torch.set_num_threads(1)
    no_live_audit()
    frozen()
    history = load(Q / "history.json")
    check_hashes(history["verified_input_hashes"])
    dataset_info = {}
    for dataset in DATASETS:
        x, y, parts = arrays_partitions(dataset)
        ids = np.load(PREPARED / dataset / "train_source_ids.npy")
        validate(load(Q / "reservations" / (dataset+".json")), [ids[p] for p in parts], history["exclusions"][dataset])
        stats = load(Q / "evaluator" / (dataset+".json"))
        if stats != reference_statistics(x, load(PREPARED / dataset / "preprocessing.json")):
            raise ValueError("training reference statistics drift")
        checkpoint = load(Q / "checkpoints" / dataset / "verified.json")
        check_hashes(checkpoint["hashes"])
        historical = load(ROOT / "artifacts/priority34a/jobs" / dataset / "baseline/321000.json")
        oldpath = next(ROOT / "artifacts/priority34a" / p for p in historical["artifact_sha256"] if p.endswith("final_checkpoint.pt"))
        exact_states(torch.load(ROOT / checkpoint["snapshots"]["50"], map_location="cpu", weights_only=False),
                     torch.load(oldpath, map_location="cpu", weights_only=False))
        snapshots = {r: torch.load(ROOT / checkpoint["snapshots"][str(r)], map_location="cpu", weights_only=False) for r in (10, 30)}
        dataset_info[dataset] = (x, y, ids, np.argsort(ids), stats, snapshots)
    table = load(Q / "qualification_table.json")
    if [r["cell"] for r in table["cells"]] != cells() or table["qualified_count"] != 0:
        raise ValueError("immutable qualification grid")
    uploads, qualification_rows = {}, {}
    for saved_cell in table["cells"]:
        cell = saved_cell["cell"]
        x, y, ids, order, stats, snapshots = dataset_info[cell["dataset"]]
        saved = load(Q / "per_target" / cell_id(cell) / "n8.json")
        bundles, guesses = [], []
        for target in stage_targets(cell, "n8"):
            identity = target_id(cell["dataset"], target)
            if identity not in uploads:
                receipt = valid_receipt(Q / "uploads" / identity / "receipt.json")
                path = ROOT / receipt["private_path"]
                if path.stat().st_mode & 0o077:
                    raise ValueError("private truth permissions")
                private = torch.load(path, map_location="cpu", weights_only=False)
                public = torch.load(ROOT / receipt["public_path"], map_location="cpu", weights_only=False)
                if private["target"] != target or set(public) != {"shared", "upload", "meta", "batch", "ordinal"}:
                    raise ValueError("source ID/public whitelist")
                indices = order[np.searchsorted(ids[order], target["source_ids"])]
                if ids[indices].tolist() != target["source_ids"] or not np.array_equal(x[indices], private["features"].numpy()) or not np.array_equal(y[indices], private["labels"].numpy().reshape(-1)):
                    raise ValueError("prepared source-ID binding")
                snapshot = snapshots[target["checkpoint"]]
                if any(not torch.equal(v, snapshot["global_non_bn"][k]) for k, v in public["shared"].items()) or any(not torch.equal(v, snapshot["client_bn"][target["client"]][k]) for k, v in private["client_bn"].items()):
                    raise ValueError("checkpoint BN/shared binding")
                replayed = client_step(public["shared"], private["client_bn"], private["features"], private["labels"], private["dropout_seed"])
                if any(not torch.equal(replayed[k], public["upload"][k]) for k in replayed):
                    raise ValueError("native Adam upload recalculation")
                if not np.array_equal(private["mean_mode"], np.repeat(np.array(stats["prior"])[None, :], target["batch"], axis=0)) or not np.array_equal(private["empirical_marginal"], marginal(x, stats, target["batch"], identity)):
                    raise ValueError("reference recalculation")
                uploads[identity] = (public, private)
            public, private = uploads[identity]
            rec = valid_receipt(Q / "jobs" / cell_id(cell) / "n8" / f"{target['index']:02d}.json")
            if rec["job"]["cell"] != cell or rec["job"]["index"] != target["index"] or rec["job"]["public_sha256"] != sha(ROOT / rec["job"]["public_path"]):
                raise ValueError("qualification job binding")
            reconstruction = torch.load(ROOT / rec["reconstruction_path"], map_location="cpu", weights_only=False)
            budget_check(reconstruction)
            if cell["label_setting"] == "given" and int(reconstruction["labels"].sum()) != int(private["labels"].sum()):
                raise ValueError("known label multiset")
            if cell["label_setting"] == "estimated":
                choices = reconstruction["label_receipt"]["candidates"]
                if [r["fraud_count"] for r in choices] != list(range(cell["batch"]+1)) or any(not math.isfinite(r["observable_objective"]) for r in choices):
                    raise ValueError("label enumeration")
                best = min(choices, key=lambda r: (r["observable_objective"], r["fraud_count"]))
                if int(reconstruction["labels"].sum()) != best["fraud_count"]:
                    raise ValueError("label count selection")
            if not torch.equal(reconstruction["rows"], candidate_rows(reconstruction["raw"], schema_blocks(public["meta"]), hard=True)):
                raise ValueError("categorical decode")
            model, nuisance = estimator_model(public["shared"], len(public["meta"]["feature_names"]), 0)
            with torch.no_grad():
                for k, p in nuisance.items():
                    p.copy_(reconstruction["nuisance"][k])
            value = objective(observable(model, candidate_rows(reconstruction["raw"], schema_blocks(public["meta"])), reconstruction["labels"], public["shared"]), flatten(public["upload"]), cell["instrument"])
            if float(value.detach()) != reconstruction["final_objective"]:
                raise ValueError("qualification objective reload")
            bundles.append(private)
            guesses.append(reconstruction["rows"].numpy())
        rows = reload_metrics(bundles, guesses, stats)
        independent = metric_audit(rows, saved["rows"], saved["medians"])
        if independent != saved["gate"] or independent != saved_cell["n8"] or independent["passed"] or saved_cell["n24"] is not None or not saved_cell["n24_not_run_reason"]:
            raise ValueError("independent qualification gate")
        qualification_rows[cell_id(cell)] = rows
    diagnostic = load(D / "summary.json")
    if len(diagnostic["summaries"]) != 18 or diagnostic["qualification_count"] != 0:
        raise ValueError("diagnostic completeness")
    diagnostic_rows, observations = {}, {}
    for cell in selected_cells():
        stats = dataset_info[cell["dataset"]][4]
        bundles = [uploads[f"{cell['dataset']}_r10_b4_n8_{i:02d}"][1] for i in range(8)]
        for condition in ("D1", "D2", "D3"):
            saved = load(D / "per_target" / condition / (cell_id(cell)+".json"))
            if saved not in diagnostic["summaries"] or saved["cell"] != cell or saved["condition"] != condition:
                raise ValueError("diagnostic summary binding")
            if condition == "D3":
                rows = qualification_rows[cell_id(cell)]
                if saved["rows"] != load(Q / "per_target" / cell_id(cell) / "n8.json")["rows"]:
                    raise ValueError("D3 must be reused unchanged")
            else:
                guesses = []
                for i in range(8):
                    identity = f"{cell['dataset']}_r10_b4_n8_{i:02d}"
                    public, private = uploads[identity]
                    obs_rec = result_receipt(D / "observations" / identity / "receipt.json")
                    payloadpath = ROOT / obs_rec["payload_path"]
                    if payloadpath.stat().st_mode & 0o077:
                        raise ValueError("privileged BN permissions")
                    payload = torch.load(payloadpath, map_location="cpu", weights_only=False)
                    if set(payload) != {"shared", "D1", "D2", "bn", "meta", "ordinal", "batch", "known_count"} or payload["known_count"] != int(private["labels"].sum()):
                        raise ValueError("oracle whitelist/label count")
                    if identity not in observations:
                        gradients = independently_capture(public, private)
                        if any(not torch.equal(v, payload["D1"][k]) for k, v in gradients.items()) or any(not torch.equal(v, payload["D2"][k]) for k, v in public["upload"].items()):
                            raise ValueError("independent gradient/Adam payload")
                        if any(not torch.equal(v, payload["bn"][k]) for k, v in private["client_bn"].items()) or any(not torch.equal(v, payload["shared"][k]) for k, v in public["shared"].items()):
                            raise ValueError("oracle BN/shared binding")
                        cosine_audit(obs_rec["cosines"], gradients, public["upload"])
                        observations[identity] = obs_rec["cosines"]
                    rec = result_receipt(D / "jobs" / condition / cell_id(cell) / f"{i:02d}.json")
                    if rec["job"]["cell"] != cell or rec["job"]["condition"] != condition or rec["job"]["index"] != i or rec["job"]["payload_sha256"] != sha(payloadpath):
                        raise ValueError("oracle job binding")
                    recon = torch.load(ROOT / rec["reconstruction_path"], map_location="cpu", weights_only=False)
                    budget_check(recon, oracle=True)
                    if not torch.equal(recon["labels"], canonical_labels(4, int(private["labels"].sum()))) or recon["dropout_masks_supplied"] or recon["condition"] != condition:
                        raise ValueError("oracle knowledge condition")
                    if not torch.equal(recon["rows"], candidate_rows(recon["raw"], schema_blocks(public["meta"]), hard=True)):
                        raise ValueError("oracle categorical decode")
                    model = oracle_model(public["shared"], private["client_bn"], len(public["meta"]["feature_names"]), 0)
                    if any(not torch.equal(v, private["client_bn"][k]) for k, v in recon["oracle_affine"].items()):
                        raise ValueError("known affine changed")
                    final = objective(operator(model, candidate_rows(recon["raw"], schema_blocks(public["meta"])), recon["labels"], public["shared"], condition), flatten(payload[condition]), cell["instrument"])
                    if float(final.detach()) != recon["final_objective"]:
                        raise ValueError("oracle objective reload")
                    guesses.append(recon["rows"].numpy())
                rows = reload_metrics(bundles, guesses, stats)
            independent = metric_audit(rows, saved["rows"], saved["medians"])
            if independent != saved["comparisons"] or independent["passed"] != saved["diagnostic_beats_all_controls"]:
                raise ValueError("independent descriptive intersection")
            if saved["cosines"] != [observations[f"{cell['dataset']}_r10_b4_n8_{i:02d}"] for i in range(8)]:
                raise ValueError("cosine summary binding")
            diagnostic_rows[(cell_id(cell), condition)] = rows
    if len(uploads) != 144 or len(qualification_rows) != 72 or len(diagnostic_rows) != 18:
        raise ValueError("independent output count")
    return dict(status="PASS", qualification_cells=72, qualification_reconstructions=576, original_uploads=144,
                diagnostic_new_reconstructions=96, diagnostic_reused_reconstructions=48, diagnostic_batches=24,
                qualified_cells=0, no_confirmatory_measurements=True, fixed_holm_family=6048,
                metric_method="separate vectorized TabLeak cost and SciPy Hungarian",
                sign_method="SciPy binomtest; production integer-combinatorial tests checked exactly",
                cosine_method="NumPy float64 independently compared to float32 cosine; audit-only 2e-6 tolerance",
                firewall=True, checkpoint_and_upload_bit_exact=True, budgets_and_knowledge=True)
