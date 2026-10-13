"""Separate vectorized metrics and SciPy gate audit of complete qualification."""
import itertools
import math
from pathlib import Path
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import binomtest
from experiments.priority35a_runtime import ROOT, OUT, Q, load, write, sha, freeze_check, check_hashes, now, scoped
from experiments.priority35a_firewall import DATASETS, validate
from experiments.priority35a_adam import require_finite_cpu
from experiments.priority35a_checkpoints import PREPARED, arrays_partitions, exact_states
from experiments.priority35a_estimator import schema_blocks, candidate_rows
from experiments.priority35a_evaluation import reference_statistics, marginal


def independent_quality(truth, guess, stats, prior_mse=None):
    truth, guess = np.asarray(truth, dtype=np.float64), np.asarray(guess, dtype=np.float64)
    if truth.shape != guess.shape or truth.ndim != 2 or not truth.size or not np.isfinite(truth).all() or not np.isfinite(guess).all():
        raise ValueError("independent metric shape/finite failure")
    cost = np.zeros((len(truth), len(guess)))
    categorical = 0
    for block, tolerance in zip(stats["blocks"], stats["tolerances"]):
        indices = block["indices"]
        if block["categories"] is None:
            left, right = truth[:, indices[0]][:, None], guess[:, indices[0]][None, :]
            cost += ~((left - tolerance <= right) & (right <= left + tolerance))
        else:
            categorical += 1
            cost += truth[:, indices].argmax(1)[:, None] != guess[:, indices].argmax(1)[None, :]
    denominator = max(categorical, 1) + max(len(stats["blocks"]) - categorical, 1)
    cost = cost / denominator
    rows, columns = linear_sum_assignment(cost)
    std = np.asarray(stats["std"])
    scale = np.where(std == 0, 1., std)
    mse = float(np.square((guess[columns] - truth) / scale).mean())
    return dict(feature_accuracy=1-float(cost[rows, columns].mean()),
                standardized_mse=mse, bounded_score=min(mse, mse if prior_mse is None else prior_mse),
                assignment=columns.tolist(), absent_type_denominator=denominator)


def compare_metrics(saved, recomputed):
    if set(saved) != set(recomputed):
        raise ValueError("metric fields differ")
    for key in saved:
        if key in ("assignment", "absent_type_denominator"):
            if saved[key] != recomputed[key]:
                raise ValueError("independent matching mismatch")
        elif not math.isfinite(saved[key]) or not math.isclose(saved[key], recomputed[key], rel_tol=1e-12, abs_tol=1e-14):
            raise ValueError("independent metric mismatch: " + key)


def independent_gate(rows, expected):
    if len(rows) != expected or [r["index"] for r in rows] != list(range(expected)):
        raise ValueError("incomplete gate units")
    tests = {}
    for control in ("mean_mode", "empirical_marginal", "decoy"):
        differences = np.array([r["estimator"]["feature_accuracy"] - r[control]["feature_accuracy"] for r in rows])
        wins, losses = int((differences > 0).sum()), int((differences < 0).sum())
        tests[control] = dict(n=expected, wins=wins, losses=losses, ties=expected-wins-losses,
            p_one_sided=float(binomtest(wins, wins+losses, alternative="greater").pvalue) if wins+losses else 1.,
            median_difference=float(np.median(differences)))
    return dict(n=expected, tests=tests, passed=all(t["p_one_sided"] < .05 for t in tests.values()),
                worst_p=max(t["p_one_sided"] for t in tests.values()))


def verify():
    from experiments.priority35a_qualification import cells, cell_id, target_id, stage_targets, client_step, valid_receipt
    from experiments.priority35a_estimator import observable, objective, flatten
    from experiments.priority35a_nuisance import estimator_model
    from experiments.priority35a_labels import canonical_labels
    torch.set_num_threads(1)
    freeze_check()
    history = load(Q / "history.json")
    datasets = {}
    uploads_verified = set()
    for dataset in DATASETS:
        x, y, partitions = arrays_partitions(dataset)
        ids = np.load(PREPARED / dataset / "train_source_ids.npy")
        reservations = load(Q / "reservations" / (dataset + ".json"))
        validate(reservations, [ids[p] for p in partitions], history["exclusions"][dataset])
        stats = load(Q / "evaluator" / (dataset + ".json"))
        if stats != reference_statistics(x, load(PREPARED / dataset / "preprocessing.json")):
            raise ValueError("independent evaluator population-statistic mismatch")
        checkpoint = load(Q / "checkpoints" / dataset / "verified.json")
        check_hashes(checkpoint["hashes"])
        historical = load(ROOT / "artifacts/priority34a/jobs" / dataset / "baseline/321000.json")
        path = next(ROOT / "artifacts/priority34a" / p for p in historical["artifact_sha256"] if p.endswith("final_checkpoint.pt"))
        exact_states(torch.load(ROOT / checkpoint["snapshots"]["50"], map_location="cpu", weights_only=False),
                     torch.load(path, map_location="cpu", weights_only=False))
        datasets[dataset] = (x, y, ids, stats)
    table = load(Q / "qualification_table.json")
    if table["cell_count"] != 72 or [row["cell"] for row in table["cells"]] != cells():
        raise ValueError("72-cell table contract")
    recomputed_table, reconstructions_verified = [], 0
    for row in table["cells"]:
        cell = row["cell"]
        x, y, ids, stats = datasets[cell["dataset"]]
        order = np.argsort(ids)
        verified_stages = {}
        for which, expected in (("n8", 8), ("n24", 24)):
            if which == "n24" and not verified_stages["n8"]["passed"]:
                if row["n24"] is not None or (Q / "per_target" / cell_id(cell) / "n24.json").exists():
                    raise ValueError("ineligible n24 measured")
                continue
            saved = load(Q / "per_target" / cell_id(cell) / (which + ".json"))
            bundles, guesses = [], []
            for target in stage_targets(cell, which):
                identity = target_id(cell["dataset"], target)
                receipt = valid_receipt(Q / "uploads" / identity / "receipt.json")
                private_path = ROOT / receipt["private_path"]
                if private_path.stat().st_mode & 0o077:
                    raise ValueError("private bundle permissions not 0600")
                private = torch.load(private_path, map_location="cpu", weights_only=False)
                public = torch.load(ROOT / receipt["public_path"], map_location="cpu", weights_only=False)
                if set(public) != {"shared", "upload", "meta", "batch", "ordinal"} or public["batch"] != target["batch"] or public["ordinal"] != identity:
                    raise ValueError("public knowledge firewall")
                if private["target"] != target or target["stage"] == "n39":
                    raise ValueError("target reservation mismatch/forbidden confirmation")
                indices = order[np.searchsorted(ids[order], target["source_ids"])]
                if not np.array_equal(x[indices], private["features"].numpy()) or not np.array_equal(y[indices], private["labels"].numpy().reshape(-1)):
                    raise ValueError("private truth/source-ID mismatch")
                if identity not in uploads_verified:
                    checkpoint = load(Q / "checkpoints" / cell["dataset"] / "verified.json")
                    snapshot = torch.load(ROOT / checkpoint["snapshots"][str(cell["checkpoint"])], map_location="cpu", weights_only=False)
                    for k in public["shared"]:
                        if not torch.equal(public["shared"][k], snapshot["global_non_bn"][k]):
                            raise ValueError("public shared checkpoint mismatch")
                    for k in private["client_bn"]:
                        if not torch.equal(private["client_bn"][k], snapshot["client_bn"][target["client"]][k]):
                            raise ValueError("private client BN mismatch")
                    raw = client_step(public["shared"], private["client_bn"], private["features"], private["labels"], private["dropout_seed"])
                    if tuple(raw) != tuple(public["upload"]) or any(not torch.equal(raw[k], public["upload"][k]) for k in raw):
                        raise ValueError("bit-exact one-step upload recalculation mismatch")
                    expected_prior = np.repeat(np.array(stats["prior"])[None, :], target["batch"], axis=0)
                    if not np.array_equal(private["mean_mode"], expected_prior) or not np.array_equal(private["empirical_marginal"], marginal(x, stats, target["batch"], identity)):
                        raise ValueError("reference construction mismatch")
                    uploads_verified.add(identity)
                result = valid_receipt(Q / "jobs" / cell_id(cell) / which / f"{target['index']:02d}.json")
                job = result["job"]
                if job["cell"] != cell or job["stage"] != which or job["index"] != target["index"] or job["public_sha256"] != sha(ROOT / receipt["public_path"]):
                    raise ValueError("reconstruction job/input binding mismatch")
                correct_count = int(private["labels"].sum()) if cell["label_setting"] == "given" else None
                if job["known_count"] != correct_count:
                    raise ValueError("label knowledge contract")
                recon = torch.load(ROOT / result["reconstruction_path"], map_location="cpu", weights_only=False)
                require_finite_cpu(dict(rows=recon["rows"], raw=recon["raw"], labels=recon["labels"], **recon["nuisance"]), "independent reconstruction")
                if recon["synthetic"] or recon["torch_threads"] != 1 or len(recon["start_traces"]) != 2 or any(trace[-1]["iteration"] != 4000 for trace in recon["start_traces"]):
                    raise ValueError("budget/thread/scientific-mode mismatch")
                if recon["chosen_start"] != min(range(2), key=lambda i: (recon["start_objectives"][i], i)):
                    raise ValueError("observable solution-selection mismatch")
                if not torch.equal(recon["rows"], candidate_rows(recon["raw"], schema_blocks(public["meta"]), hard=True)):
                    raise ValueError("post-optimization category decode mismatch")
                if not torch.equal(recon["labels"], canonical_labels(target["batch"], int(recon["labels"].sum()))):
                    raise ValueError("unordered canonical label multiset mismatch")
                surrogate, nuisance = estimator_model(public["shared"], len(public["meta"]["feature_names"]), 0)
                if tuple(nuisance) != tuple(recon["nuisance"]):
                    raise ValueError("nuisance schema mismatch")
                with torch.no_grad():
                    for name, param in nuisance.items():
                        param.copy_(recon["nuisance"][name])
                final = objective(observable(surrogate, candidate_rows(recon["raw"], schema_blocks(public["meta"])),
                                            recon["labels"], public["shared"]), flatten(public["upload"]), cell["instrument"])
                if float(final.detach()) != recon["final_objective"]:
                    raise ValueError("final observable objective reload mismatch")
                if cell["label_setting"] == "estimated":
                    label_doc = recon["label_receipt"]
                    candidates = label_doc["candidates"]
                    if [c["fraud_count"] for c in candidates] != list(range(target["batch"]+1)) or any(not math.isfinite(c["observable_objective"]) for c in candidates):
                        raise ValueError("missing/nonfinite label enumeration")
                    best = min(candidates, key=lambda c: (c["observable_objective"], c["fraud_count"]))
                    if int(recon["labels"].sum()) != best["fraud_count"]:
                        raise ValueError("label observable argmin mismatch")
                bundles.append(private)
                guesses.append(recon)
                reconstructions_verified += 1
            recomputed_rows = []
            for i, saved_row in enumerate(saved["rows"]):
                bundle, recon = bundles[i], guesses[i]
                truth = bundle["features"].numpy()
                prior = independent_quality(truth, bundle["mean_mode"], stats)
                prior_mse = prior["standardized_mse"]
                calculated = dict(mean_mode=prior, estimator=independent_quality(truth, recon["rows"].numpy(), stats, prior_mse),
                    empirical_marginal=independent_quality(truth, bundle["empirical_marginal"], stats, prior_mse),
                    decoy=independent_quality(truth, guesses[(i+1) % expected]["rows"].numpy(), stats, prior_mse))
                for key, metric in calculated.items():
                    compare_metrics(saved_row[key], metric)
                if saved_row["source_ids"] != bundle["target"]["source_ids"] or saved_row["fraud_count"] != int(bundle["labels"].sum()):
                    raise ValueError("per-target metadata mismatch")
                recomputed_rows.append(dict(index=i, **calculated))
            medians = {arm: {metric: float(np.median([r[arm][metric] for r in recomputed_rows]))
                            for metric in ("feature_accuracy", "standardized_mse", "bounded_score")}
                       for arm in ("estimator", "mean_mode", "empirical_marginal", "decoy")}
            for arm in medians:
                for metric in medians[arm]:
                    if not math.isclose(medians[arm][metric], saved["medians"][arm][metric], rel_tol=1e-12, abs_tol=1e-14):
                        raise ValueError("independent summary median mismatch")
            independent = independent_gate(recomputed_rows, expected)
            if independent != saved["gate"] or independent != row[which]:
                raise ValueError("SciPy independent gate mismatch")
            verified_stages[which] = independent
        recomputed_table.append(dict(cell=cell, n8=verified_stages["n8"], n24=verified_stages.get("n24"),
            n24_not_run_reason=None if "n24" in verified_stages else "n8 failed at least one prespecified control gate",
            qualified=bool(verified_stages.get("n24", {}).get("passed", False))))
    if recomputed_table != table["cells"] or sum(r["qualified"] for r in recomputed_table) != table["qualified_count"]:
        raise ValueError("independent qualification table mismatch")
    if any("n39" in path.parts for path in (Q / "jobs").rglob("*.json")):
        raise ValueError("forbidden n39 workload")
    audit_doc = dict(status="PASS", at=now(), cell_count=72, reconstructions_verified=reconstructions_verified,
                     uploads_verified=len(uploads_verified), independent_metric="vectorized native TabLeak cost + SciPy Hungarian",
                     independent_sign="SciPy binomtest versus integer-combinatorial production gate",
                     qualified_count=table["qualified_count"], no_confirmatory_or_defended_uploads=True)
    write(Q / "independent_verification.json", audit_doc, exclusive=True)
    report = ["# Priority35A qualification-only results", "", f"Qualified cells: {table['qualified_count']}/72. Independent verification PASS.",
        "", "Only unprotected uploads were measured. n24 is fresh and conditional on all three n8 gates.",
        "No confirmatory, DP, v1 or v2 measurements; the full evaluation remains incomplete.",
        "The experimental unit is a batch, not a row. Exact one-sided sign tests exclude only exact ties from their binomial count.",
        "The gates are an unadjusted prespecified intersection; no transform protection or full Holm-family claim is made.",
        "Private client BN affine/statistics and dropout masks were unavailable to estimators. Free/default BN affine and dropout-only identity are fixed approximations.",
        "The empirical controls are update-free but training-derived. Frequency-encoded categories are assessed as continuous encoded coordinates.",
        "", "| Cell | n8 worst p | n24 worst p | Decision |", "|---|---:|---:|---|"]
    for row in recomputed_table:
        report.append(f"| {cell_id(row['cell'])} | {row['n8']['worst_p']:.8g} | {row['n24']['worst_p'] if row['n24'] else 'NOT_RUN: n8 failed'} | {'QUALIFIED' if row['qualified'] else 'NOT_QUALIFIED'} |")
    closest = sorted(recomputed_table, key=lambda r: (0 if r["n24"] else 1, (r["n24"] or r["n8"])["worst_p"], cell_id(r["cell"])))[:5]
    report += ["", "## Closest-to-qualifying evidence", ""]
    for row in closest:
        endpoint = row["n24"] or row["n8"]
        report.append(f"- {cell_id(row['cell'])}, n={endpoint['n']}: " + "; ".join(f"{c}: W/L/T={t['wins']}/{t['losses']}/{t['ties']}, p={t['p_one_sided']:.8g}, median accuracy delta={t['median_difference']:.8g}" for c, t in endpoint["tests"].items()))
    if table["qualified_count"] == 0:
        report += ["", "No qualified instrument recovered records from realistic Adam uploads even without a defense.",
                   "This is limited to the fixed tested budgets, dropout approximation and nuisance treatment; it does not establish zero leakage or immunity to other instruments."]
    report += ["", f"Protocol SHA-256: {freeze_check()['protocol_sha256']}.", f"Execution freeze SHA-256: {sha(Q / 'execution_freeze.json')}.",
               "Exact execution command is preserved in launch_receipt.json; per-target outcomes and all three gate statistics are retained in per_target/ and qualification_table.json.",
               "Final stdout/stderr and administrative progress are outside the immutable scientific manifest; seal these separately after process exit.", ""]
    scoped(Q / "qualification_summary.md").write_text("\n".join(report))
    # Logs can still receive the supervisor's terminal event; never hash them prematurely.
    paths = [p for p in Q.rglob("*") if p.is_file() and p.suffix in (".json", ".pt", ".md")
             and p.name not in ("final_manifest.json", "VERIFIED.json")]
    manifest = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(Q / "final_manifest.json", dict(hashes=manifest, excluded_live_logs=True), exclusive=True)
    check_hashes(manifest)
    write(Q / "VERIFIED.json", dict(status="QUALIFICATION_VERIFIED", full_priority35a_complete=False,
          table_sha256=sha(Q / "qualification_table.json"), report_sha256=sha(Q / "qualification_summary.md"),
          manifest_sha256=sha(Q / "final_manifest.json"), requires_post_exit_audit=True), exclusive=True)
    return audit_doc
