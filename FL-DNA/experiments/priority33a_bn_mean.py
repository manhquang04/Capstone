"""P33a A2 CPU-only fixed-checkpoint qualification and paired BN-channel tests."""
import argparse
import copy
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"
import numpy as np
import torch
from scipy.stats import binomtest
from experiments.priority33a_audit import OUT, AMENDMENT, write, append, sha, now
from experiments import priority32_multidataset as p
from experiments.priority24_t1_bn_valid_rq1 import recover_mean, mse_std, BN_KEY, V1, V2
from dna_encoder.transform_defense import transform_update_array
from dna_encoder import transform_defense_v2 as v2
from privacy.seed_manager import derive_seed
from experiments.dp_update_accounting import alpha_grid, epsilon_from_rdp

torch.set_num_threads(1)
METHODS = ("dna_v1_conservative", "dna_v2_0p95")
DATASETS = ("ieee_cis", "baf")
SIGMAS = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, .001, .003]


def checkpoint_status(index, status):
    path = OUT / "checklist.json"
    doc = json.loads(path.read_text())
    doc[str(index)]["status"] = status
    write(path, doc)


def progress(stage, dataset, method, done, total, failed=0):
    doc = dict(stage=stage, dataset=dataset, method=method, done=done, total=total, failed=failed,
        started_at=None, last_update=now(), eta_minutes=None)
    write(OUT / "progress.json", doc)
    append(OUT / "progress.log", doc)


def strict_state(state, context):
    issues = {}
    for key, value in state.items():
        if not torch.isfinite(value).all() or (key.endswith("running_var") and (value < 0).any()):
            issues[key] = dict(nonfinite=int((~torch.isfinite(value)).sum()),
                negative=int((value < 0).sum()) if key.endswith("running_var") else 0)
    if issues:
        raise ValueError(json.dumps(dict(gate="BN_OR_FINITE", context=context, issues=issues)))


def training_job(job):
    dataset, method, seed = job["dataset"], job["method"], job["seed"]
    folder = OUT / "A2" / dataset / "training" / job["id"]
    folder.mkdir(parents=True, exist_ok=True)
    config = p.job_config(dataset, "dna_v1_conservative" if method == "dp" else method, seed,
        json.loads((p.OUT / "execution_freeze.json").read_text())["chosen"][dataset]["training"], quality=seed != 321000)
    original_factory, original_average, original_prob = p.FraudMLP, p.fed_avg, p.probabilities
    model_ref = []
    rounds = []
    norms = []
    def forward_guard(module, inputs, outputs):
        assert outputs.device.type == "cpu", "new training must be CPU"
        strict_state(module.state_dict(), "actual_forward")
        if not torch.isfinite(outputs).all():
            raise ValueError("nonfinite actual forward logits")
    def factory(dim):
        model = original_factory(dim)
        model.register_forward_hook(forward_guard)
        model_ref.append(model)
        return model
    def average(states, counts):
        for i, state in enumerate(states):
            strict_state(state, f"transmitted_client_{i}_round_{len(rounds)+1}")
        global_state = model_ref[0].state_dict()
        if method == "baseline":
            for state in states:
                squared = sum(float(torch.sum((state[k].double()-global_state[k].double())**2)) for k in state if torch.is_floating_point(state[k]))
                norms.append(float(np.sqrt(squared)))
        result = original_average(states, counts)
        strict_state(result, f"aggregate_round_{len(rounds)+1}")
        row = dict(round=len(rounds)+1, bn_min={k: float(v.min()) for k, v in result.items() if k.endswith("running_var")})
        rounds.append(row)
        append(folder / "bn_rounds.jsonl", row)
        return result
    def probabilities(model, array):
        strict_state(model.state_dict(), "final_evaluation")
        result = original_prob(model, array)
        if not np.isfinite(result).all():
            raise ValueError("nonfinite final probabilities")
        return result
    p.FraudMLP, p.fed_avg, p.probabilities, p.write = factory, average, probabilities, write
    if method == "dp":
        from privacy.dp_engine import apply_dp_to_local_state
        counter = [0]
        def dp_state(local, global_state, unused_config):
            r, client = divmod(counter[0], 3)
            counter[0] += 1
            state, *diag = apply_dp_to_local_state(local, global_state, job["clip"], job["sigma"],
                noise_generator=torch.Generator().manual_seed(derive_seed(seed, "priority33a-utility-dp", r+1, client)))
            return state, diag
        p.dna_transform_state = dp_state
    error = None
    started = time.time()
    try:
        p.train_job(config, folder / "native_metrics.json")
        torch.save({k: v.detach().clone() for k, v in model_ref[0].state_dict().items()}, folder / "final_state.pt")
        doc = json.loads((folder / "native_metrics.json").read_text())
        reproduction = None
        if seed == 321000:
            stored = json.loads((p.OUT / "rq2_jobs" / dataset / "baseline/seed_321000.json").read_text())
            differences = {f"{split}.{key}": abs(doc[split][key] - stored[split][key])
                for split in ("validation", "test") for key in ("f1", "auc_roc", "pr_auc", "rows", "fraud_rate")}
            differences["threshold"] = abs(doc["threshold"]-stored["threshold"])
            reproduction = dict(differences=differences, passed=max(differences.values()) <= 1e-8)
            if not reproduction["passed"]:
                raise ValueError("Fixed CPU baseline did not reproduce stored P32; fail closed")
        result = dict(job=job, status="COMPLETED", metrics=doc, reproduction=reproduction, update_norms=norms,
            checkpoint_sha256=sha(folder / "final_state.pt"))
    except Exception:
        error = traceback.format_exc()
        result = dict(job=job, status="GATE_FAILED", exception=error, rounds_completed=len(rounds))
    result.update(elapsed_seconds=time.time()-started, torch_threads=torch.get_num_threads(), device="cpu", finished_at=now())
    write(folder / "result.json", result)


def execute(job):
    folder = OUT / "A2" / job["dataset"] / "training" / job["id"]
    path = folder / "result.json"
    if path.exists():
        doc = json.loads(path.read_text())
        assert doc["job"] == job
        if doc["status"] == "COMPLETED":
            assert sha(folder / "final_state.pt") == doc["checkpoint_sha256"]
        append(OUT / "runs.jsonl", dict(at=now(), event="A2_validated_skip", job=job))
        return doc
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "bn_rounds.jsonl").exists():
        raise RuntimeError("Partial job requires explicit interruption disclosure; no silent overwrite")
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--training-job", json.dumps(job)]
    append(OUT / "runs.jsonl", dict(at=now(), event="A2_job_started", job=job, command=command))
    with (folder / "stdout.log").open("a") as so, (folder / "stderr.log").open("a") as se:
        proc = subprocess.run(command, cwd=ROOT, stdout=so, stderr=se)
    if not path.exists():
        write(path, dict(job=job, status="GATE_FAILED", infrastructure_error=True, returncode=proc.returncode))
    doc = json.loads(path.read_text())
    append(OUT / "runs.jsonl", dict(at=now(), event="A2_job_finished", job=job, status=doc["status"], sha256=sha(path)))
    docs = [json.loads(f.read_text()) for f in (OUT / "A2" / job["dataset"] / "training").glob("*/result.json")]
    docs = [d for d in docs if d["job"]["method"] == job["method"] and d["job"].get("sigma") == job.get("sigma")
        and (d["job"]["seed"] == 321000) == (job["seed"] == 321000)]
    progress("A2_training", job["dataset"], job["method"], len(docs), 1 if job["seed"] == 321000 else 16,
        sum(d["status"] != "COMPLETED" for d in docs))
    return doc


def load_model(dataset):
    checkpoint = OUT / "A2" / dataset / "training/baseline_321000/final_state.pt"
    prepared = p.OUT / "prepared" / dataset
    with torch.random.fork_rng(devices=[]):
        model = p.FraudMLP(np.load(prepared / "train_x.npy", mmap_mode="r").shape[1])
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    strict_state(model.state_dict(), "fixed_checkpoint")
    return model


def targets(dataset):
    path = OUT / "A2" / dataset / "targets.json"
    source = np.load(p.OUT / "prepared" / dataset / "test_source_ids.npy")
    rng = np.random.default_rng(330320 if dataset == "ieee_cis" else 330321)
    exclusions, provenance = set(), []
    if dataset == "ieee_cis":
        # Earlier IEEE provenance uses TransactionID, exactly P32's source-ID domain.
        for manifest in (ROOT / "artifacts/priority5_ieee_cis").glob("**/target_manifest.json"):
            old = json.loads(manifest.read_text())
            ids = [int(i) for group in old["groups"] for i in group["transaction_ids"]]
            exclusions.update(ids)
            provenance.append(dict(path=str(manifest.relative_to(ROOT)), sha256=sha(manifest), source_ids=ids))
        # Read actual bundles: their container key is 'targets', not 'groups'.
        # Earlier firewall code did not consistently parse that container.
        for bundle in sorted((ROOT / "artifacts").glob("**/ieee_priority6_bundle.pt")):
            old = torch.load(bundle, map_location="cpu")
            assert old["targets"] and all("transaction_ids" in group for group in old["targets"])
            ids = [int(i) for group in old["targets"] for i in group["transaction_ids"]]
            exclusions.update(ids)
            provenance.append(dict(path=str(bundle.relative_to(ROOT)), sha256=sha(bundle), source_ids=ids))
    available = np.flatnonzero(~np.isin(source, list(exclusions)))
    indices = available[rng.permutation(len(available))[:284]].reshape(71, 4)
    groups = {"n8": indices[:8].tolist(), "n24": indices[8:32].tolist(), "n39": indices[32:].tolist()}
    doc = dict(dataset=dataset, groups=groups, source_ids={k: [[int(source[i]) for i in g] for g in v] for k, v in groups.items()},
        test_source_sha256=sha(p.OUT / "prepared" / dataset / "test_source_ids.npy"), earlier_provenance=provenance)
    flattened = [i for gs in doc["source_ids"].values() for g in gs for i in g]
    assert len(flattened) == len(set(flattened)) == 284
    assert not set(flattened).intersection(exclusions)
    if path.exists():
        assert json.loads(path.read_text()) == doc
    else:
        write(path, doc)
    return doc


def captures(dataset, stage, model, target_doc):
    path = OUT / "A2" / dataset / stage / "captures.pt"
    if path.exists():
        return torch.load(path, map_location="cpu")
    prepared = p.OUT / "prepared" / dataset
    x, y = np.load(prepared / "test_x.npy", mmap_mode="r"), np.load(prepared / "test_y.npy")
    criterion = p.BinaryFocalLoss(alpha=.95, gamma=2.)
    result = []
    for i, group in enumerate(target_doc["groups"][stage]):
        local = copy.deepcopy(model).train()
        torch.manual_seed(derive_seed(330322, dataset, stage, i))
        inputs = torch.from_numpy(np.array(x[group], copy=True))
        labels = torch.from_numpy(np.array(y[group], copy=True)[:, None])
        optimizer = torch.optim.Adam(local.parameters(), lr=.001)
        optimizer.zero_grad()
        logits = local(inputs)
        strict_state(local.state_dict(), "single_local_step")
        assert torch.isfinite(logits).all()
        loss = criterion(logits, labels)
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
        strict_state(local.state_dict(), "single_step_post_optimizer")
        result.append(dict(target=i, source_ids=target_doc["source_ids"][stage][i], raw=local.state_dict()[BN_KEY]-model.state_dict()[BN_KEY],
            true_mean=np.asarray(inputs.numpy(), dtype=np.float64).mean(axis=0)))
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, path)
    return result


def sign(a, b):
    diff = np.asarray(a)-np.asarray(b)
    wins, losses = int((diff > 0).sum()), int((diff < 0).sum())
    return dict(wins=wins, losses=losses, ties=int((diff == 0).sum()),
        p=float(binomtest(wins, wins+losses, .5, alternative="greater").pvalue) if wins+losses else 1.)


def population(dataset):
    x = np.load(p.OUT / "prepared" / dataset / "train_x.npy", mmap_mode="r")
    mean, std = np.asarray(x, dtype=np.float64).mean(axis=0), np.asarray(x, dtype=np.float64).std(axis=0)
    std[std < 1e-12] = 1
    return mean, std


def qualify(dataset, stage, model, cap, prior, std):
    recovered = [recover_mean(model, c["raw"]) for c in cap]
    rows = [dict(target=c["target"], source_ids=c["source_ids"], recovered_mse=mse_std(recovered[i], c["true_mean"], std),
        prior_mse=mse_std(prior, c["true_mean"], std), decoy_mse=mse_std(recovered[(i+1)%len(cap)], c["true_mean"], std)) for i, c in enumerate(cap)]
    tests = {name: sign([r[name+"_mse"] for r in rows], [r["recovered_mse"] for r in rows]) for name in ("prior", "decoy")}
    doc = dict(stage=stage, rows=rows, tests=tests, passed=all(t["p"] < .05 for t in tests.values()))
    write(OUT / "A2" / dataset / stage / "qualification.json", doc)
    append(OUT / "progress.log", dict(at=now(), event="qualification_finished", dataset=dataset, stage=stage, passed=doc["passed"], tests=tests))
    progress("A2_qualification_"+stage, dataset, "unprotected", len(cap), len(cap), int(not doc["passed"]))
    return doc


def projection(metadata):
    n, k = metadata.padded_size, metadata.sketch_size
    # Construct R solely from known key/metadata, never the original update.
    eye = np.eye(metadata.original_size, n)
    signs = v2._signs(n, metadata.seed)
    return np.stack([np.sqrt(n/k)*v2._fwht_normalized(row*signs)[list(metadata.sampled_indices)] for row in eye], axis=1)


def payload(model, raw, method, target):
    if method not in METHODS:
        raise ValueError("unknown defense; no fallback")
    index = list(model.state_dict()).index(BN_KEY)
    if method == METHODS[0]:
        q, metadata = transform_update_array(raw.numpy(), V1, tensor_index=index)
        return dict(kind="v1", q=torch.from_numpy(q), metadata=metadata)
    q, metadata = v2.transform_update_array_v2(raw.numpy(), V2, tensor_index=index,
        quantization_seed=derive_seed(V2.seed, "priority24-v2", index))
    return dict(kind="v2", q=q, metadata=metadata)


def recover_payload(model, transmitted):
    if transmitted["kind"] == "v1":
        return recover_mean(model, transmitted["q"])
    if transmitted["kind"] != "v2":
        raise ValueError("unknown payload; no fallback")
    r = projection(transmitted["metadata"])
    linear, bn = model.network[0], model.network[1]
    m = float(bn.momentum)
    w = linear.weight.detach().numpy().astype(np.float64)
    offset = m*(linear.bias.detach().numpy().astype(np.float64)-bn.running_mean.detach().numpy().astype(np.float64))
    return np.linalg.lstsq(r@(m*w), np.asarray(transmitted["q"], dtype=np.float64)-r@offset, rcond=None)[0]


def distortion(dataset, model, cap):
    clip = 1.01*max(float(torch.linalg.vector_norm(c["raw"])) for c in cap)
    result = {}
    for method in METHODS:
        differences, norms = [], []
        for c in cap:
            t = payload(model, c["raw"], method, c["target"])
            if method == METHODS[0]:
                decoded = t["q"].numpy()
            else:
                decoded, _ = v2.reconstruct_update_array_v2(t["q"], t["metadata"])
            differences.append(float(np.linalg.norm(decoded-c["raw"].numpy())))
            rng = torch.Generator().manual_seed(derive_seed(330330, dataset, method, c["target"]))
            norms.append(float(torch.linalg.vector_norm(torch.randn(c["raw"].shape, generator=rng, dtype=torch.float64))))
        sigma = float(np.median(differences)/(clip*np.median(norms)))
        result[method] = dict(clip=clip, sigma=sigma, median_distortion=float(np.median(differences)), delta=1e-5,
            accounting=epsilon_from_rdp(sigma, 1., 1e-5, 1, alpha_grid()) if sigma > 0 else None)
    write(OUT / "A2" / dataset / "distortion_calibration.json", result)
    return result


def confirm(dataset, model, cap, std, calibration, utilities):
    for method in METHODS:
        for c in cap:
            path = OUT / "A2" / dataset / "confirmatory" / method / f"target_{c['target']:02d}" / "result.json"
            if path.exists():
                doc = json.loads(path.read_text())
                assert doc["source_ids"] == c["source_ids"] and doc["method"] == method and doc["target"] == c["target"]
                assert np.isfinite(doc["defended_mse"])
                assert sha(path.parent / "transmitted_payload.pt") == doc["transmitted_payload_sha256"]
                for comparator in doc["comparators"].values():
                    if comparator["status"] == "VALID":
                        assert np.isfinite(comparator["dp_mse"])
                continue
            transmitted = payload(model, c["raw"], method, c["target"])
            # Persist exactly the receipt handed to recovery, independently of ground truth.
            defended_mse = mse_std(recover_payload(model, transmitted), c["true_mean"], std)
            rows = {}
            for comparator, cal in (("distortion", calibration[method]), ("utility", utilities[method])):
                if cal.get("status", "VALID") != "VALID":
                    rows[comparator] = dict(status="NOT_ASSESSABLE", reason=cal["reason"])
                    continue
                raw = c["raw"].double()
                clipped = raw*min(1., cal["clip"]/max(float(raw.norm()), 1e-12))
                gen = torch.Generator().manual_seed(derive_seed(330330, dataset, method, c["target"]))
                q = (clipped+torch.randn(raw.shape, generator=gen, dtype=torch.float64)*cal["clip"]*cal["sigma"]).to(c["raw"].dtype)
                rows[comparator] = dict(status="VALID", dp_mse=mse_std(recover_mean(model, q), c["true_mean"], std), clip=cal["clip"], sigma=cal["sigma"])
            assert np.isfinite(defended_mse)
            path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(transmitted, path.parent / "transmitted_payload.pt")
            write(path, dict(dataset=dataset, target=c["target"], source_ids=c["source_ids"], method=method, defended_mse=defended_mse,
                comparators=rows, transmitted_payload_sha256=sha(path.parent / "transmitted_payload.pt")))
            append(OUT / "progress.log", dict(at=now(), event="confirmatory_target_finished", dataset=dataset, method=method, target=c["target"]))
            progress("A2_confirmatory", dataset, method, c["target"]+1, 39)


def utility(dataset):
    # First test the required transforms. Invalid training cannot define a utility target.
    docs = {method: [execute(dict(dataset=dataset, method=method, seed=seed, id=f"{method}_{seed}")) for seed in range(330100, 330116)] for method in METHODS}
    result = {}
    for method in METHODS:
        bad = [d["job"]["seed"] for d in docs[method] if d["status"] != "COMPLETED"]
        if bad:
            result[method] = dict(status="NOT_ASSESSABLE", reason="Required full-state CPU transform violates BN/finite training gate", failed_seeds=bad)
    if len(result) == 2:
        write(OUT / "A2" / dataset / "utility_calibration.json", result)
        return result
    baselines = [execute(dict(dataset=dataset, method="baseline", seed=seed, id=f"baseline_{seed}")) for seed in range(330100, 330116)]
    if any(d["status"] != "COMPLETED" for d in baselines):
        for method in METHODS:
            result.setdefault(method, dict(status="NOT_ASSESSABLE", reason="Required baseline CPU training gate failed"))
    else:
        clip = 1.01*float(np.percentile([n for d in baselines[:2] for n in d["update_norms"]], 95))
        base_f1 = np.asarray([d["metrics"]["validation"]["f1"] for d in baselines])
        grid = []
        for sigma in SIGMAS+[.01, .03]:
            points = [execute(dict(dataset=dataset, method="dp", seed=seed, sigma=sigma, clip=clip, id=f"dp_{sigma}_{seed}")) for seed in range(330100, 330116)]
            valid = all(d["status"] == "COMPLETED" for d in points)
            grid.append(dict(sigma=sigma, valid=valid, delta=float(np.mean([d["metrics"]["validation"]["f1"] for d in points])-base_f1.mean()) if valid else None))
            if sigma == .003:
                # Extension only if the initial grid has not bracketed every valid transform.
                need_extension = False
                for method in METHODS:
                    if method in result:
                        continue
                    tolerance = float(np.mean([d["metrics"]["validation"]["f1"] for d in docs[method]])-base_f1.mean()-.005)
                    eligible = [i for i, g in enumerate(grid) if g["valid"] and g["delta"] >= tolerance]
                    if not eligible or not any(g["valid"] and g["delta"] < tolerance for g in grid[max(eligible)+1:]):
                        need_extension = True
                if not need_extension:
                    break
        for method in METHODS:
            if method in result:
                continue
            tolerance = float(np.mean([d["metrics"]["validation"]["f1"] for d in docs[method]])-base_f1.mean()-.005)
            eligible = [i for i, g in enumerate(grid) if g["valid"] and g["delta"] >= tolerance]
            selected = max(eligible) if eligible else None
            if selected is not None and any(g["valid"] and g["delta"] < tolerance for g in grid[selected+1:]):
                result[method] = dict(status="VALID", sigma=grid[selected]["sigma"], clip=clip, tolerance=tolerance, grid=grid,
                    accounting=epsilon_from_rdp(grid[selected]["sigma"], 1., 1e-5, 1, alpha_grid()))
            else:
                result[method] = dict(status="NOT_ASSESSABLE", reason="No valid bracket within frozen grid/one extension", grid=grid, tolerance=tolerance)
    write(OUT / "A2" / dataset / "utility_calibration.json", result)
    return result


def supervise():
    lock = OUT / "A2_supervisor.json"
    if lock.exists():
        try:
            os.kill(json.loads(lock.read_text())["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("A2 supervisor already active; no duplicate")
    write(lock, dict(pid=os.getpid(), started_at=now()))
    # Wait for A1 without competing with its four process budget.
    while not (OUT / "A1_attempt2_COMPLETE.json").exists():
        lock = OUT / "A1_attempt2_supervisor.json"
        if lock.exists():
            try:
                os.kill(json.loads(lock.read_text())["pid"], 0)
            except ProcessLookupError:
                raise RuntimeError("A1 stopped unexpectedly; inspect before A2")
        time.sleep(30)
    completed_a1 = json.loads((OUT / "A1_attempt2_COMPLETE.json").read_text())
    if completed_a1["failed"]:
        raise RuntimeError("A1 replay errors require inspection before next stage")
    checkpoint_status(1, "completed")
    checkpoint_status(2, "in_progress")
    dependencies = ("experiments/priority24_t1_bn_valid_rq1.py", "experiments/priority32_multidataset.py", "experiments/fraud_fl_common.py",
        "models/fraud_mlp.py", "data/load_creditcard.py", "dna_encoder/transform_defense.py", "dna_encoder/transform_defense_v2.py",
        "privacy/dp_engine.py", "privacy/seed_manager.py", "experiments/dp_update_accounting.py", "tests/test_priority33a.py")
    freeze = dict(amendment_sha256=sha(AMENDMENT), runner_sha256=sha(__file__), p32_runner_sha256=sha(p.__file__),
        dependency_sha256={name: sha(ROOT / name) for name in dependencies},
        prepared={dataset: {f.name: sha(f) for f in (p.OUT / "prepared" / dataset).glob("*") if f.is_file()} for dataset in DATASETS})
    path = OUT / "A2_execution_freeze.json"
    if path.exists():
        assert json.loads(path.read_text()) == freeze
    else:
        write(path, freeze)
    summary = {}
    for dataset in DATASETS:
        baseline = execute(dict(dataset=dataset, method="baseline", seed=321000, id="baseline_321000"))
        if baseline["status"] != "COMPLETED":
            summary[dataset] = dict(status="NOT_ASSESSABLE", reason="Fixed checkpoint replay/BN gate failed")
            continue
        model, target_doc = load_model(dataset), targets(dataset)
        checkpoint_status(3, "in_progress")
        prior, std = population(dataset)
        for stage in ("n8", "n24"):
            cap = captures(dataset, stage, model, target_doc)
            gate = qualify(dataset, stage, model, cap, prior, std)
            if not gate["passed"]:
                summary[dataset] = dict(status="NOT_ASSESSABLE", reason=f"Unprotected batch-mean qualification {stage} failed", gate=gate["tests"])
                break
        else:
            checkpoint_status(4, "in_progress")
            calibration = distortion(dataset, model, cap)
            utilities = utility(dataset)
            checkpoint_status(5, "in_progress")
            confirm(dataset, model, captures(dataset, "n39", model, target_doc), std, calibration, utilities)
            summary[dataset] = dict(status="CONFIRMATORY_COMPLETE", utility=utilities)
    write(OUT / "A2_COMPLETE.json", dict(datasets=summary, finished_at=now()))
    for index in (2, 3, 4, 5):
        # A failed dataset gate is not marked completed/passing.
        checkpoint_status(index, "resolved_with_NOT_ASSESSABLE" if any(d["status"] == "NOT_ASSESSABLE" for d in summary.values()) else "completed")
    checklist = json.loads((OUT / "checklist.json").read_text())
    checklist["dataset_outcomes"] = summary
    write(OUT / "checklist.json", checklist)
    checkpoint_status(6, "in_progress")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--training-job")
    args = parser.parse_args()
    supervise() if args.supervise else training_job(json.loads(args.training_job))
