"""Qualification ONLY. Immutable jobs, detached execution, fail-closed receipts."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import itertools
import multiprocessing as mp
import socket
import platform
import subprocess
import sys
import time
import traceback
from functools import lru_cache
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments import priority34a_local_bn as old
from experiments.priority35a_runtime import OUT, Q, PROTOCOL, PROTOCOL_SHA, sha, load, write, stage, tensor_save, check_hashes, freeze_check, now
from experiments.priority35a_history import audit
from experiments.priority35a_firewall import DATASETS, reserve, validate, digest
from experiments.priority35a_checkpoints import PREPARED, arrays_partitions, replay
from experiments.priority35a_estimator import reconstruct
from experiments.priority35a_adam import require_finite_cpu, parameter_difference
from experiments.priority35a_evaluation import reference_statistics, marginal, quality, gate

PORT = 35935


def cells():
    return [dict(dataset=d, checkpoint=r, batch=b, label_setting=l, instrument=e)
            for d, r, b, l, e in itertools.product(DATASETS, (10, 30), (4, 8, 16), ("given", "estimated"), ("tableak", "wan"))]


def cell_id(cell):
    return "{dataset}_r{checkpoint}_b{batch}_{label_setting}_{instrument}".format(**cell)


def target_id(dataset, target):
    return f"{dataset}_r{target['checkpoint']}_b{target['batch']}_{target['stage']}_{target['index']:02d}"


def preflight():
    attempt = Q / "preflight" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    sources = sorted([*(ROOT / "experiments").glob("priority35a_*.py"), *(ROOT / "tests").glob("test_priority35a*.py")])
    write(attempt / "inputs.json", dict(protocol_sha256=sha(PROTOCOL), sources={str(p.relative_to(ROOT)): sha(p) for p in sources}), exclusive=True)
    commands = [[sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_priority35a*.py", "-v"],
                [sys.executable, "-m", "py_compile", *[str(p.relative_to(ROOT)) for p in sources]], ["git", "diff", "--check"]]
    for index, command in enumerate(commands):
        env = dict(os.environ, PYTHONPYCACHEPREFIX=str(OUT / "bytecode"))
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        write(attempt / f"command_{index}.json", dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), exclusive=True)
        print(result.stdout + result.stderr, flush=True)
        if result.returncode:
            write(attempt / "FAILED.json", dict(command_index=index), exclusive=True)
            raise ValueError("qualification preflight failed; preserve logs")
    for path in sources:
        text = path.read_text()
        if not text.endswith("\n") or any(line.rstrip(" \t") != line for line in text.splitlines()):
            raise ValueError("new source whitespace failure")
    write(attempt / "PASS.json", dict(at=now(), synthetic_only=True, commands=commands), exclusive=True)
    stage("qualification_preflight_passed", qualification_preflight_attempt=str(attempt.relative_to(ROOT)), measurement_jobs_launched=0)


def prepare():
    if (Q / "execution_freeze.json").exists():
        raise ValueError("freeze exists; do not prepare again")
    if sha(PROTOCOL) != PROTOCOL_SHA:
        raise ValueError("protocol changed")
    attempts = sorted((Q / "preflight").glob("*/PASS.json"))
    if not attempts:
        raise ValueError("qualification execution requires synthetic preflight")
    check_hashes(load(attempts[-1].parent / "inputs.json")["sources"])
    stage("qualification_preparation", current_authorization="qualification only; no confirmatory", required_direction=None)
    history = audit()
    write(Q / "history.json", history, exclusive=True)
    hashes = dict(history["verified_input_hashes"])
    for relative in old.SOURCES:
        hashes[relative] = sha(ROOT / relative)
    for path in itertools.chain((ROOT / "experiments").glob("priority35a_*.py"),
                                (ROOT / "tests").glob("test_priority35a*.py"),
                                (ROOT / "protocols/amendments").glob("2026-10-10_priority35a_*.md")):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    for dataset, seed in zip(DATASETS, (351000, 351001, 351002)):
        folder = PREPARED / dataset
        prepared_audit = load(folder / "audit.json")
        for name, expected in prepared_audit["prepared_sha256"].items():
            relative = str((folder / name).relative_to(ROOT))
            if sha(ROOT / relative) != expected:
                raise ValueError("P32 prepared input drift: " + relative)
            hashes[relative] = expected
        hashes[str((folder / "audit.json").relative_to(ROOT))] = sha(folder / "audit.json")
        historical_path = old.OUT / "jobs" / dataset / "baseline/321000.json"
        historic = load(historical_path)
        check_hashes(historic["config"]["source_sha256"])
        old.valid_result(historical_path, historic["config"])
        hashes.update(historic["config"]["source_sha256"])
        hashes[str(historical_path.relative_to(ROOT))] = sha(historical_path)
        for relative, expected in historic["artifact_sha256"].items():
            hashes[str((old.OUT / relative).relative_to(ROOT))] = expected
        x, _, partitions = arrays_partitions(dataset)
        ids = np.load(folder / "train_source_ids.npy")
        if ids.shape != (len(x),) or len(np.unique(ids)) != len(ids):
            raise ValueError("invalid prepared original source IDs")
        source_partitions = [ids[indices] for indices in partitions]
        reservations = reserve(dataset, source_partitions, history["exclusions"][dataset], seed)
        write(Q / "reservations" / (dataset + ".json"), reservations, exclusive=True)
        stats = reference_statistics(x, load(folder / "preprocessing.json"))
        write(Q / "evaluator" / (dataset + ".json"), stats, private=True, exclusive=True)
    for path in Q.rglob("*.json"):
        hashes[str(path.relative_to(ROOT))] = sha(path)
    write(Q / "execution_freeze.json", dict(created_at=now(), protocol_sha256=PROTOCOL_SHA, hashes=hashes,
        cells=cells(), workers=4, cpu_only=True, torch_threads_per_process=1, budgets=dict(starts=2, iterations=4000, label_steps=64),
        scope="qualification unprotected only; n39 IDs reserved but never measured",
        software=dict(python=sys.version, torch=torch.__version__, numpy=np.__version__, platform=platform.platform())), exclusive=True)
    checklist = load(OUT / "checklist.json")
    checklist.update(sources_inputs_frozen=True, qualification_execution_ready=True)
    write(OUT / "checklist.json", checklist)
    stage("qualification_execution_frozen", execution_freeze_sha256=sha(Q / "execution_freeze.json"), cells_total=72)


def client_step(public, client_bn, features, labels, private_seed):
    require_finite_cpu(public, "client shared state")
    require_finite_cpu(client_bn, "client private BN")
    require_finite_cpu({"features": features, "labels": labels}, "private target")
    with torch.random.fork_rng(devices=[]):
        model = old.p.FraudMLP(features.shape[1]).cpu()
        bn, allowed = old.domains(model)
        if set(client_bn) != set(bn) or tuple(public) != allowed:
            raise ValueError("client checkpoint tensor firewall")
        state = model.state_dict()
        state.update(client_bn)
        model.load_state_dict(state, strict=True)
        old.broadcast(model, public, bn, allowed)
        model.train()
        old.guard(model, "target pre-step")
        torch.manual_seed(private_seed)
        optimizer = torch.optim.Adam(model.parameters(), lr=.001)
        optimizer.zero_grad()
        logits = model(features)
        loss = old.p.BinaryFocalLoss(alpha=.95, gamma=2.)(logits, labels)
        if not torch.isfinite(logits).all() or not torch.isfinite(loss):
            raise ValueError("nonfinite target forward")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("nonfinite/missing target gradient")
        optimizer.step()
        old.guard(model, "target post-step")
        after = OrderedDict((name, model.state_dict()[name].clone()) for name in allowed)
        return parameter_difference(public, after)


def valid_receipt(path):
    doc = load(path)
    if doc["status"] != "COMPLETED" or doc["freeze_sha256"] != sha(Q / "execution_freeze.json"):
        raise ValueError("completed receipt identity mismatch")
    check_hashes(doc["hashes"])
    return doc


@lru_cache(maxsize=3)
def evaluator_arrays(dataset):
    x, y, _ = arrays_partitions(dataset)
    ids = np.load(PREPARED / dataset / "train_source_ids.npy")
    order = np.argsort(ids)
    return x, y, ids, order


def target_bundle(dataset, target):
    identity = target_id(dataset, target)
    receipt = Q / "uploads" / identity / "receipt.json"
    if receipt.exists():
        return valid_receipt(receipt)
    x, y, source_ids, order = evaluator_arrays(dataset)
    # Preserve target order; never substitute row indices for original IDs.
    positions = np.searchsorted(source_ids[order], target["source_ids"])
    indices = order[positions]
    if source_ids[indices].tolist() != target["source_ids"]:
        raise ValueError("target ID lookup mismatch")
    checkpoint = load(Q / "checkpoints" / dataset / "verified.json")
    check_hashes(checkpoint["hashes"])
    snapshot = torch.load(ROOT / checkpoint["snapshots"][str(target["checkpoint"])], map_location="cpu", weights_only=False)
    features = torch.from_numpy(np.array(x[indices], copy=True))
    labels = torch.from_numpy(np.array(y[indices], copy=True)[:, None])
    private_seed = int.from_bytes(os.urandom(8), "big") % (2**63-1)
    public = snapshot["global_non_bn"]
    upload = client_step(public, snapshot["client_bn"][target["client"]], features, labels, private_seed)
    meta = load(PREPARED / dataset / "preprocessing.json")
    stats = load(Q / "evaluator" / (dataset + ".json"))
    private_path = receipt.parent / "private.pt"
    public_path = receipt.parent / "public.pt"
    tensor_save(private_path, dict(target=target, features=features, labels=labels,
        dropout_seed=private_seed, client_bn=snapshot["client_bn"][target["client"]],
        mean_mode=np.repeat(np.asarray(stats["prior"])[None, :], target["batch"], axis=0),
        empirical_marginal=marginal(x, stats, target["batch"], identity)), private=True)
    tensor_save(public_path, dict(shared=public, upload=upload, meta=meta, batch=target["batch"], ordinal=identity))
    result = dict(status="COMPLETED", freeze_sha256=sha(Q / "execution_freeze.json"), identity=identity,
                  public_path=str(public_path.relative_to(ROOT)), private_path=str(private_path.relative_to(ROOT)),
                  hashes={str(path.relative_to(ROOT)): sha(path) for path in (private_path, public_path)},
                  torch_threads=torch.get_num_threads(), no_bn_uploaded=True, no_private_dropout_to_estimator=True)
    write(receipt, result, exclusive=True)
    return result


def reconstruction_job(job):
    torch.set_num_threads(1)
    result_path = ROOT / job["result_path"]
    if result_path.exists():
        return valid_receipt(result_path)
    attempt = result_path.parent / ("attempt_" + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "job.json", job, exclusive=True)
    try:
        public = torch.load(ROOT / job["public_path"], map_location="cpu", weights_only=False)
        if sha(ROOT / job["public_path"]) != job["public_sha256"]:
            raise ValueError("public job input drift")
        # No private bundle is read by this reconstruction worker.
        result = reconstruct(public["shared"], public["upload"], public["meta"], public["batch"],
            job["cell"]["label_setting"], job["cell"]["instrument"], public["ordinal"], job["known_count"])
        if result["synthetic"] or result["torch_threads"] != 1:
            raise ValueError("scientific worker contract")
        tensor_path = attempt / "reconstruction.pt"
        tensor_save(tensor_path, result)
        receipt = dict(status="COMPLETED", freeze_sha256=job["freeze_sha256"], job=job,
            reconstruction_path=str(tensor_path.relative_to(ROOT)), hashes={str(tensor_path.relative_to(ROOT)): sha(tensor_path),
            str((attempt / "job.json").relative_to(ROOT)): sha(attempt / "job.json")}, elapsed_seconds=result["elapsed_seconds"])
        write(result_path, receipt, exclusive=True)
        return receipt
    except BaseException:
        write(attempt / "failure.json", dict(status="FAILED", traceback=traceback.format_exc(), at=now()), exclusive=True)
        raise


def stage_targets(cell, which):
    doc = load(Q / "reservations" / (cell["dataset"] + ".json"))
    return [target for target in doc["targets"] if target["checkpoint"] == cell["checkpoint"]
            and target["batch"] == cell["batch"] and target["stage"] == which]


def run_stage(selected, which):
    jobs = []
    for cell in selected:
        for target in stage_targets(cell, which):
            bundle = target_bundle(cell["dataset"], target)
            result_path = Q / "jobs" / cell_id(cell) / which / f"{target['index']:02d}.json"
            if result_path.exists():
                valid_receipt(result_path)
                continue
            private = torch.load(ROOT / bundle["private_path"], map_location="cpu", weights_only=False)
            job = dict(cell=cell, stage=which, index=target["index"], public_path=bundle["public_path"],
                public_sha256=bundle["hashes"][bundle["public_path"]],
                known_count=int(private["labels"].sum()) if cell["label_setting"] == "given" else None,
                result_path=str(result_path.relative_to(ROOT)), freeze_sha256=sha(Q / "execution_freeze.json"))
            jobs.append(job)
    stage("qualification_" + which, stage_jobs_pending=len(jobs), cells_scheduled=len(selected), stage_jobs_done=0)
    errors, done = [], 0
    # Drain the submitted pool even on failure; never cancel/replace a target.
    with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
        futures = {pool.submit(reconstruction_job, job): job for job in jobs}
        write(Q / ("worker_pool_" + which + "_" + str(time.time_ns()) + ".json"),
              dict(stage=which, worker_pids=sorted(pool._processes), supervisor_pid=os.getpid(), at=now()), exclusive=True)
        for future in as_completed(futures):
            try:
                future.result()
                done += 1
            except BaseException:
                errors.append(dict(job=futures[future], traceback=traceback.format_exc()))
            stage("qualification_" + which, stage_jobs_done=done, stage_jobs_pending=len(jobs)-done-len(errors), failed_jobs=len(errors))
    if errors:
        write(Q / ("REQUIRES_DIRECTION_" + str(time.time_ns()) + ".json"), dict(errors=errors, stage=which), exclusive=True)
        raise ValueError("qualification failures; all submitted jobs drained, requires user direction")
    return [evaluate_cell(cell, which) for cell in selected]


def evaluate_cell(cell, which):
    count = 8 if which == "n8" else 24
    stats = load(Q / "evaluator" / (cell["dataset"] + ".json"))
    bundles, guesses = [], []
    for target in stage_targets(cell, which):
        bundle = valid_receipt(Q / "uploads" / target_id(cell["dataset"], target) / "receipt.json")
        bundles.append(torch.load(ROOT / bundle["private_path"], map_location="cpu", weights_only=False))
        result = valid_receipt(Q / "jobs" / cell_id(cell) / which / f"{target['index']:02d}.json")
        recon = torch.load(ROOT / result["reconstruction_path"], map_location="cpu", weights_only=False)
        require_finite_cpu(dict(rows=recon["rows"], raw=recon["raw"], **recon["nuisance"]), "saved reconstruction")
        guesses.append(recon)
    rows = []
    for i, (bundle, recon) in enumerate(zip(bundles, guesses)):
        truth = bundle["features"].numpy()
        prior = quality(truth, bundle["mean_mode"], stats)
        prior_mse = prior["standardized_mse"]
        rows.append(dict(index=i, source_ids=bundle["target"]["source_ids"], client=bundle["target"]["client"],
            fraud_count=int(bundle["labels"].sum()), estimated_fraud_count=int(recon["labels"].sum()),
            estimator=quality(truth, recon["rows"].numpy(), stats, prior_mse), mean_mode=prior,
            empirical_marginal=quality(truth, bundle["empirical_marginal"], stats, prior_mse),
            decoy=quality(truth, guesses[(i+1) % count]["rows"].numpy(), stats, prior_mse)))
    medians = {arm: {metric: float(np.median([row[arm][metric] for row in rows]))
                    for metric in ("feature_accuracy", "standardized_mse", "bounded_score")}
               for arm in ("estimator", "mean_mode", "empirical_marginal", "decoy")}
    result = dict(cell=cell, stage=which, rows=rows, gate=gate(rows, count), medians=medians)
    path = Q / "per_target" / cell_id(cell) / (which + ".json")
    if path.exists():
        if load(path) != result:
            raise ValueError("per-target evaluation changed")
    else:
        write(path, result, exclusive=True)
    return result


def supervisor():
    torch.set_num_threads(1)
    with socket.socket() as lock:
        lock.bind(("127.0.0.1", PORT))
        try:
            freeze_check()
            stage("checkpoint_replay", supervisor_pid=os.getpid(), scientific_runs_launched=1, no_live_scientific_workers=False)
            for dataset in DATASETS:
                replay(dataset)
            checklist = load(OUT / "checklist.json")
            checklist.update(checkpoints_verified=True, no_live_scientific_workers=False)
            write(OUT / "checklist.json", checklist)
            first = run_stage(cells(), "n8")
            passed = [doc["cell"] for doc in first if doc["gate"]["passed"]]
            second = run_stage(passed, "n24") if passed else []
            rows = []
            lookup = {cell_id(doc["cell"]): doc for doc in second}
            for doc in first:
                next_stage = lookup.get(cell_id(doc["cell"]))
                rows.append(dict(cell=doc["cell"], n8=doc["gate"], n24=next_stage["gate"] if next_stage else None,
                    n24_not_run_reason=None if next_stage else "n8 failed at least one prespecified control gate",
                    qualified=bool(next_stage and next_stage["gate"]["passed"])))
            write(Q / "qualification_table.json", dict(cells=rows, cell_count=72, qualified_count=sum(r["qualified"] for r in rows)), exclusive=True)
            stage("qualification_independent_verification", n8_cells=72, n24_cells=len(passed))
            from experiments.priority35a_verify import verify
            verify()
            stage("qualification_verified_waiting_user", status="QUALIFICATION_VERIFIED_WAITING_USER", qualification_complete=True,
                  qualified_cells=sum(r["qualified"] for r in rows), confirmatory_launched=False)
            checklist = load(OUT / "checklist.json")
            checklist.update(qualification_complete=True, qualification_independent_verification_pass=True,
                             no_live_scientific_workers=False, requires_post_exit_audit=True, full_evaluation_complete=False)
            write(OUT / "checklist.json", checklist)
        except BaseException:
            write(Q / ("SUPERVISOR_FAILURE_" + str(time.time_ns()) + ".json"), dict(traceback=traceback.format_exc(), at=now()), exclusive=True)
            stage("qualification_requires_direction", status="REQUIRES_DIRECTION", required_direction="preserved failure; no automatic replay")
            raise


def launch(resume):
    freeze_check()
    for path in Q.glob("*FAILURE*.json"):
        raise ValueError("preserved failure requires user direction before replay: " + str(path))
    for path in Q.glob("REQUIRES_DIRECTION*.json"):
        raise ValueError("preserved failure requires user direction before replay: " + str(path))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    # Read-only no-live audit, case-insensitive Python process names on macOS.
    processes = subprocess.check_output(["ps", "-axo", "pid,comm,args"], text=True)
    live = [line for line in processes.splitlines() if len(line.split()) >= 2
            and "python" in Path(line.split()[1]).name.lower() and "priority35a_qualification.py" in line
            and str(os.getpid()) != line.split()[0]]
    if live:
        raise ValueError("Priority35A process alive; no duplicate launch")
    for receipt in Q.glob("worker_pool_*.json"):
        for pid in load(receipt)["worker_pids"]:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise ValueError("recorded worker PID still live; verify before any replay")
    receipt_path = Q / "launch_receipt.json"
    if receipt_path.exists() and not resume:
        raise ValueError("launch already exists; verified interruption needs --resume")
    if resume:
        if not receipt_path.exists():
            raise ValueError("nothing to resume")
        for path in (Q / "jobs").glob("*/*/*.json"):
            valid_receipt(path)
    attempt = Q / ("launch_" + str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    command = [str(ROOT / ".venv-phase1/bin/python"), "-B", str(Path(__file__).resolve()), "--supervisor"]
    with (attempt / "stdout.log").open("wb") as stdout, (attempt / "stderr.log").open("wb") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True, env=os.environ.copy())
    receipt = dict(pid=child.pid, at=now(), resume=resume, command=command,
                   freeze_sha256=sha(Q / "execution_freeze.json"), stdout=str((attempt / "stdout.log").relative_to(ROOT)),
                   stderr=str((attempt / "stderr.log").relative_to(ROOT)), no_live_audit_pass=True)
    write(attempt / "receipt.json", receipt, exclusive=True)
    if not receipt_path.exists():
        write(receipt_path, receipt, exclusive=True)
    print(receipt, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--launch", action="store_true")
    mode.add_argument("--supervisor", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if Path(sys.prefix).resolve() != (ROOT / ".venv-phase1").resolve():
        raise ValueError("use .venv-phase1")
    torch.set_num_threads(1)
    if args.preflight:
        preflight()
    elif args.prepare:
        prepare()
    elif args.launch:
        launch(args.resume)
    else:
        supervisor()
