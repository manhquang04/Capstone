"""P33a A1 exact native DP replay, read-only BN instrumentation, isolated outputs."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/priority33a"
AMENDMENT = ROOT / "protocols/amendments/2026-10-02_priority33a_bn_audit_mean_recovery.md"
CORE = ["round", "train_loss", "f1_score", "auc_roc", "pr_auc", "accuracy", "precision", "recall", "optimal_threshold", "tn", "fp", "fn", "tp"]


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(OUT.resolve())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def append(path, value):
    with Path(path).open("a") as f:
        f.write(json.dumps(value, allow_nan=False) + "\n")


def jobs():
    result = []
    for priority, seeds, branch in (
        ("P25", range(2501101, 2501104), "priority25_rq1_extension/utility_grid_20260929"),
        ("P25b", range(2501201, 2501204), "priority25b_bracketed_utility/utility_grid_initial_20260929"),
    ):
        for sigma, tag in ((1e-5, "1em05"), (3e-5, "3em05")):
            for seed in seeds:
                result.append(dict(priority=priority, seed=seed, sigma=sigma, variant="full_state_single_clip", clip=259.0841131896973,
                    stored=f"artifacts/{branch}/seed_{seed}/dp_sigma_{tag}/metrics.json"))
    clips = json.loads((ROOT / "protocols/config/priority27_c2_utility_grid.json").read_text())["clip_specs"]
    for variant in ("full_state_single_clip", "per_tensor_clip", "fedbn_trainable_only"):
        sigma, tag = (1e-6, "1em06") if variant == "fedbn_trainable_only" else (3e-5, "3em05")
        for seed in range(270201, 270204):
            result.append(dict(priority="P27", seed=seed, sigma=sigma, variant=variant, clip=clips[variant],
                stored=f"artifacts/priority27_gaps_harness/c2_utility_grid_initial_20260929/seed_{seed}/{variant}__sigma_{tag}/metrics.json"))
    assert len(result) == 21
    return result


def folder_for(job):
    return OUT / "A1_attempt2" / job["priority"] / job["variant"] / f"sigma_{job['sigma']}_seed_{job['seed']}"


def run(job):
    folder = folder_for(job)
    folder.mkdir(parents=True, exist_ok=True)
    os.environ.update({"FL_RUN_SEED": str(job["seed"]), "MAX_ROWS": "500000", "NUM_ROUNDS": "50", "LOCAL_EPOCHS": "1",
        "FL_NUM_CLIENTS": "3", "LOSS_TYPE": "focal", "FOCAL_ALPHA": ".95", "FOCAL_GAMMA": "2.0", "DATALOADER_NUM_WORKERS": "4",
        "DP_OUTPUT_PATH": str(folder / "metrics.json"), "DP_CLIP_NORM": str(job["clip"]), "DP_NOISE_MULTIPLIER": str(job["sigma"]),
        "PRIORITY27_DP_OUTPUT_PATH": str(folder / "metrics.json"), "PRIORITY27_DP_VARIANT": job["variant"],
        "PRIORITY27_DP_NOISE_MULTIPLIER": str(job["sigma"]), "PRIORITY27_DP_CLIP_SPEC_JSON": json.dumps(job["clip"]),
        "PYTHONDONTWRITEBYTECODE": "1"})
    for key in ("QUICK", "DP_NOISE_PRESET"):
        os.environ.pop(key, None)
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[key] = "1"
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from experiments import fraud_fl_common as common
    from models.fraud_mlp import FraudMLP
    torch.set_num_threads(1)
    if job["priority"] == "P27":
        from experiments import run_fraud_fl_dp_priority27 as native
    else:
        from experiments import run_fraud_fl_dp as native
    assert str(native.DEVICE) == "mps", "Exact historical device required; no CPU fallback"
    rounds, evaluations = [], []
    state = None
    actual = None
    started = time.time()

    def variance(s):
        rows = {}
        for k, v in s.items():
            if k.endswith("running_var"):
                a = v.detach().cpu()
                finite = torch.isfinite(a)
                rows[k] = dict(min=float(a.min()) if bool(finite.all()) else None, negative=int((a < 0).sum()), nonfinite=int((~finite).sum()))
        return rows

    def aggregate(states, counts):
        nonlocal state
        rng = torch.get_rng_state().clone()
        result = common.fed_avg(states, counts)
        row = dict(round=len(rounds) + 1, aggregate=variance(result), transmitted_clients=[variance(s) for s in states])
        rounds.append(row)
        append(folder / "bn_rounds.jsonl", row)
        if len(rounds) == 50:
            state = {k: v.detach().cpu().clone() for k, v in result.items()}
            torch.save(state, folder / "final_state.pt")
        assert torch.equal(rng, torch.get_rng_state()), "BN hook changed RNG"
        return result

    def evaluation(model, loader, threshold=None):
        nonlocal actual
        rec = dict(round=len(rounds), split="validation" if threshold is None else "test", rows=0, nonfinite_logits=0)
        probabilities = []
        def hook(module, inputs, outputs):
            rng = torch.get_rng_state().clone()
            rec["rows"] += len(outputs)
            rec["nonfinite_logits"] += int((~torch.isfinite(outputs)).sum().cpu())
            probabilities.append(torch.sigmoid(outputs.detach()).cpu().numpy().reshape(-1).copy())
            assert torch.equal(rng, torch.get_rng_state())
        handle = model.register_forward_hook(hook)
        try:
            metric = common.evaluate_model(model, loader, threshold)
            rec["metrics"] = metric
            return metric
        finally:
            handle.remove()
            a = np.concatenate(probabilities) if probabilities else np.empty(0)
            rec["nonfinite_probabilities"] = int((~np.isfinite(a)).sum())
            evaluations.append(rec)
            append(folder / "evaluations.jsonl", rec)
            if len(rounds) == 50:
                np.save(folder / (rec["split"] + "_probabilities.npy"), a, allow_pickle=False)
                if threshold is not None:
                    x, y = loader.dataset.tensors
                    actual = (x.detach().cpu(), y.detach().cpu(), float(threshold))
                    torch.save(dict(x=actual[0], y=actual[1], threshold=actual[2]), folder / "final_test_inputs.pt")

    error = None
    backends = {}
    try:
        native.fed_avg = aggregate
        native.evaluate_model = evaluation
        native.main()
        assert len(rounds) == 50 and state is not None and actual is not None
        x, y, threshold = actual
        for device in ("cpu", "mps"):
            # Constructor RNG is restored. Evaluation consumes no randomness.
            with torch.random.fork_rng(devices=[]):
                model = FraudMLP(x.shape[1]).to(device)
            model.load_state_dict(state)
            model.eval()
            values, nonfinite_logits = [], 0
            with torch.no_grad():
                for start in range(0, len(x), 4096):
                    logits = model(x[start:start+4096].to(device))
                    nonfinite_logits += int((~torch.isfinite(logits)).sum().cpu())
                    values.append(torch.sigmoid(logits).cpu().numpy().reshape(-1))
            a = np.concatenate(values)
            np.save(folder / f"final_{device}_probabilities.npy", a, allow_pickle=False)
            backends[device] = dict(rows=len(a), finite_probabilities=int(np.isfinite(a).sum()), nonfinite_logits=nonfinite_logits,
                parameters_buffers_unchanged=all(torch.equal(v.detach().cpu(), state[k]) for k, v in model.state_dict().items()))
            if np.isfinite(a).all():
                from experiments.priority32_multidataset import metrics
                backends[device]["metrics"] = metrics(y.numpy().reshape(-1), a, threshold)
            assert backends[device]["parameters_buffers_unchanged"]
    except Exception:
        error = traceback.format_exc()
    reproduction = None
    if (folder / "metrics.json").exists():
        stored = json.loads((ROOT / job["stored"]).read_text())
        replay = json.loads((folder / "metrics.json").read_text())
        mismatch, errors = [], []
        for a, b in zip(stored["rounds"], replay["rounds"]):
            for k in CORE:
                if a[k] is None or b[k] is None:
                    if a[k] != b[k]:
                        mismatch.append([a["round"], k, a[k], b[k]])
                else:
                    delta = abs(float(a[k]) - float(b[k]))
                    errors.append(delta)
                    if delta > 1e-8 or (k in {"tn", "fp", "fn", "tp"} and delta):
                        mismatch.append([a["round"], k, a[k], b[k]])
        reproduction = dict(round_count_equal=len(stored["rounds"]) == len(replay["rounds"]), mismatches=mismatch,
            max_absolute_error=max(errors, default=0), config_equal=stored["config"] == replay["config"])
    negative = [r["round"] for r in rounds if any(v["negative"] for v in r["aggregate"].values())]
    client_negative = [r["round"] for r in rounds if any(v["negative"] for s in r["transmitted_clients"] for v in s.values())]
    write(folder / "result.json", dict(job=job, status="FAILED" if error else "COMPLETED", exception=error, rounds_completed=len(rounds),
        negative_rounds=negative, client_negative_rounds=client_negative, final_bn=variance(state) if state else None,
        backends=backends, reproduction=reproduction, native_device=str(native.DEVICE), torch_threads=torch.get_num_threads(),
        elapsed_seconds=time.time()-started, finished_at=now()))


def supervise():
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "A1_attempt2_supervisor.json"
    if lock.exists():
        try:
            os.kill(json.loads(lock.read_text())["pid"], 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Existing supervisor alive; no duplicate")
    batch = jobs()
    sources = [AMENDMENT, Path(__file__), ROOT / "experiments/run_fraud_fl_dp.py", ROOT / "experiments/run_fraud_fl_dp_priority27.py",
        ROOT / "experiments/fraud_fl_common.py", ROOT / "experiments/priority27_dp_variants.py", ROOT / "data/load_creditcard.py",
        ROOT / "models/fraud_mlp.py", ROOT / "privacy/dp_engine.py", ROOT / "protocols/config/priority27_c2_utility_grid.json"]
    freeze = dict(jobs=batch, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in sources},
        stored_sha256={j["stored"]: sha(ROOT/j["stored"]) for j in batch}, workers=4, torch_threads=1)
    if (OUT / "A1_attempt2_execution_freeze.json").exists():
        assert json.loads((OUT / "A1_attempt2_execution_freeze.json").read_text()) == freeze, "Frozen source/config drift"
    else:
        write(OUT / "A1_attempt2_execution_freeze.json", freeze)
    write(lock, dict(pid=os.getpid(), started_at=now()))
    started = time.time()
    done = failed = 0
    def one(job):
        folder = folder_for(job)
        path = folder / "result.json"
        if path.exists():
            doc = json.loads(path.read_text())
            assert doc["job"] == job and doc["status"] in {"COMPLETED", "FAILED"}
            if doc["status"] == "COMPLETED":
                assert doc["rounds_completed"] == 50 and set(doc["backends"]) == {"cpu", "mps"}
            append(OUT / "runs.jsonl", dict(at=now(), event="validated_skip", job=job))
            return doc
        folder.mkdir(parents=True, exist_ok=True)
        if any(folder.glob("bn_rounds.jsonl")):
            raise RuntimeError(f"Partial historical replay requires disclosed interruption before resuming: {folder}")
        command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--job", json.dumps(job)]
        append(OUT / "runs.jsonl", dict(at=now(), event="job_started", job=job, command=command))
        with (folder / "stdout.log").open("a") as so, (folder / "stderr.log").open("a") as se:
            proc = subprocess.run(command, cwd=ROOT, stdout=so, stderr=se)
        if not path.exists():
            write(path, dict(job=job, status="FAILED", returncode=proc.returncode, infrastructure_error=True))
        doc = json.loads(path.read_text())
        append(OUT / "runs.jsonl", dict(at=now(), event="job_finished", job=job, status=doc["status"], sha256=sha(path)))
        return doc
    with ThreadPoolExecutor(max_workers=4) as pool:
        for future in as_completed([pool.submit(one, j) for j in batch]):
            doc = future.result()
            done += 1
            failed += int(doc["status"] == "FAILED")
            progress = dict(stage="A1_native_DP_BN_audit", dataset="paysim", method=doc["job"]["variant"], done=done, total=21,
                failed=failed, started_at=datetime.fromtimestamp(started, timezone.utc).isoformat(), last_update=now(),
                eta_minutes=(time.time()-started)/done*(21-done)/60)
            write(OUT / "progress.json", progress)
            append(OUT / "progress.log", progress)
    write(OUT / "A1_attempt2_COMPLETE.json", dict(done=done, failed=failed, elapsed_seconds=time.time()-started, finished_at=now()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--job")
    args = parser.parse_args()
    supervise() if args.supervise else run(json.loads(args.job))
