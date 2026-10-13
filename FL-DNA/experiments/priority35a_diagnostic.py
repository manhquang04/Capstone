"""Detached post-hoc descriptive D1/D2; D3 remains frozen qualification data."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from experiments.priority35a_runtime import OUT, Q, sha, load, write, stage, tensor_save, check_hashes, freeze_check, now
from experiments.priority35a_qualification import cell_id, valid_receipt
from experiments.priority35a_diagnostic_estimator import observed_gradient, sign_cosines, reconstruct
from experiments.priority35a_evaluation import quality, gate

D = OUT / "diagnostic"
ADDENDUM = ROOT / "protocols/amendments/2026-10-10_priority35a_descriptive_oracle_bn_diagnostic.md"
PORT = 35936


def selected_cells():
    return [dict(dataset=d, checkpoint=10, batch=4, label_setting="given", instrument=e)
            for d, e in itertools.product(("baf", "ieee_cis", "paysim"), ("tableak", "wan"))]


def audit_qualification():
    freeze_check()
    check_hashes(load(Q / "final_manifest.json")["hashes"])
    table = load(Q / "qualification_table.json")
    if table["cell_count"] != 72 or table["qualified_count"] != 0 or len(table["cells"]) != 72:
        raise ValueError("expected final 0/72 qualification")
    if any(r["qualified"] or r["n8"]["passed"] or r["n24"] is not None for r in table["cells"]):
        raise ValueError("qualification decision differs")
    return table


def no_live_audit():
    result = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True, check=True)
    found = []
    for line in result.stdout.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2 or int(fields[0]) == os.getpid():
            continue
        executable = fields[1].split()[0].lower()
        if "python" in Path(executable).name and ("priority35a_qualification.py" in fields[1] or
                "priority35a_diagnostic.py" in fields[1] or "spawn_main" in fields[1] and str(ROOT / ".venv-phase1") in fields[1]):
            found.append(line.strip())
    if found:
        raise ValueError("live scientific worker: " + repr(found))
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", PORT))
    return dict(at=now(), no_live_scientific_workers=True)


def prepare():
    if (D / "freeze.json").exists():
        raise ValueError("diagnostic already frozen")
    audit_qualification()
    no_live_audit()
    attempt = D / "preflight" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    sources = sorted([*(ROOT / "experiments").glob("priority35a_*.py"), *(ROOT / "tests").glob("test_priority35a*.py")])
    commands = [[sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_priority35a*.py", "-v"],
                [sys.executable, "-m", "py_compile", *[str(p.relative_to(ROOT)) for p in sources]], ["git", "diff", "--check"]]
    for i, command in enumerate(commands):
        env = dict(os.environ, PYTHONPYCACHEPREFIX=str(OUT / "diagnostic_bytecode"))
        result = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
        write(attempt / f"command_{i}.json", dict(command=command, returncode=result.returncode,
              stdout=result.stdout, stderr=result.stderr), exclusive=True)
        print(result.stdout+result.stderr, flush=True)
        if result.returncode:
            raise ValueError("diagnostic synthetic preflight failed")
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    hashes[str(ADDENDUM.relative_to(ROOT))] = sha(ADDENDUM)
    hashes.update(load(Q / "final_manifest.json")["hashes"])
    for p in (Q / "final_manifest.json", Q / "execution_freeze.json", Q / "VERIFIED.json"):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    for p in attempt.glob("*.json"):
        hashes[str(p.relative_to(ROOT))] = sha(p)
    write(D / "freeze.json", dict(at=now(), hashes=hashes, cells=selected_cells(), conditions=["D1", "D2", "D3"],
          new_jobs=96, reused_jobs=48, starts=2, iterations=4000, workers=4, threads=1,
          addendum_sha256=sha(ADDENDUM), descriptive_only=True, qualification_unchanged=True), exclusive=True)
    stage("diagnostic_frozen", status="DIAGNOSTIC_READY", current_authorization="post-hoc D1/D2 descriptive diagnostic and final report; no confirmatory",
          diagnostic_jobs_total=96, diagnostic_jobs_done=0, diagnostic_addendum_sha256=sha(ADDENDUM),
          diagnostic_freeze_sha256=sha(D / "freeze.json"), pending=["D1/D2 diagnostic", "independent diagnostic audit", "final report/CSV/Holm/manifest verification"], required_direction=None)
    checklist = load(OUT / "checklist.json")
    checklist.update(diagnostic_addendum_frozen=True, diagnostic_tests_pass=True, diagnostic_complete=False, diagnostic_independent_verification_pass=False)
    write(OUT / "checklist.json", checklist)


def frozen():
    doc = load(D / "freeze.json")
    check_hashes(doc["hashes"])
    return doc


def result_receipt(path):
    doc = load(path)
    if doc["status"] != "COMPLETED" or doc["freeze_sha256"] != sha(D / "freeze.json"):
        raise ValueError("diagnostic receipt identity")
    check_hashes(doc["hashes"])
    return doc


def capture(dataset, index):
    identity = f"{dataset}_r10_b4_n8_{index:02d}"
    path = D / "observations" / identity / "receipt.json"
    if path.exists():
        return result_receipt(path)
    receipt = valid_receipt(Q / "uploads" / identity / "receipt.json")
    public = torch.load(ROOT / receipt["public_path"], map_location="cpu", weights_only=False)
    private = torch.load(ROOT / receipt["private_path"], map_location="cpu", weights_only=False)
    gradients = observed_gradient(public, private)
    # Explicit whitelist: oracle BN only, never features, row labels, IDs or private RNG.
    payload = dict(shared=public["shared"], D1=gradients, D2=public["upload"], bn=private["client_bn"],
                   meta=public["meta"], ordinal=identity, batch=4, known_count=int(private["labels"].sum()))
    payload_path = path.parent / "oracle_payload.pt"
    tensor_save(payload_path, payload, private=True)
    doc = dict(status="COMPLETED", freeze_sha256=sha(D / "freeze.json"), payload_path=str(payload_path.relative_to(ROOT)),
               hashes={str(payload_path.relative_to(ROOT)): sha(payload_path)}, identity=identity, cosines=sign_cosines(gradients, public["upload"]),
               torch_threads=torch.get_num_threads(), no_private_dropout_to_estimator=True)
    write(path, doc, exclusive=True)
    return doc


def worker(job):
    torch.set_num_threads(1)
    attempt = ROOT / job["result_path"]
    attempt = attempt.parent / ("attempt_"+str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "job.json", job, exclusive=True)
    try:
        if sha(ROOT / job["payload_path"]) != job["payload_sha256"]:
            raise ValueError("oracle payload hash drift")
        payload = torch.load(ROOT / job["payload_path"], map_location="cpu", weights_only=False)
        if set(payload) != {"shared", "D1", "D2", "bn", "meta", "ordinal", "batch", "known_count"}:
            raise ValueError("diagnostic knowledge whitelist")
        result = reconstruct(payload["shared"], payload[job["condition"]], payload["bn"], payload["meta"], 4,
                             job["cell"]["instrument"], payload["ordinal"], payload["known_count"], job["condition"])
        if result["synthetic"] or result["torch_threads"] != 1:
            raise ValueError("diagnostic scientific contract")
        tensor = attempt / "reconstruction.pt"
        tensor_save(tensor, result, private=True)
        doc = dict(status="COMPLETED", freeze_sha256=job["freeze_sha256"], job=job,
                   reconstruction_path=str(tensor.relative_to(ROOT)), elapsed_seconds=result["elapsed_seconds"],
                   hashes={str(p.relative_to(ROOT)): sha(p) for p in (tensor, attempt / "job.json")})
        write(ROOT / job["result_path"], doc, exclusive=True)
        return doc
    except BaseException:
        write(attempt / "failure.json", dict(at=now(), traceback=traceback.format_exc()), exclusive=True)
        raise


def evaluate():
    summaries = []
    for cell in selected_cells():
        identifier = cell_id(cell)
        stats = load(Q / "evaluator" / (cell["dataset"]+".json"))
        bundles = []
        for i in range(8):
            rec = valid_receipt(Q / "uploads" / f"{cell['dataset']}_r10_b4_n8_{i:02d}" / "receipt.json")
            bundles.append(torch.load(ROOT / rec["private_path"], map_location="cpu", weights_only=False))
        for condition in ("D1", "D2", "D3"):
            if condition == "D3":
                saved = load(Q / "per_target" / identifier / "n8.json")
                rows, medians, comparison = saved["rows"], saved["medians"], saved["gate"]
            else:
                guesses = []
                for i in range(8):
                    rec = result_receipt(D / "jobs" / condition / identifier / f"{i:02d}.json")
                    guesses.append(torch.load(ROOT / rec["reconstruction_path"], map_location="cpu", weights_only=False)["rows"].numpy())
                rows = []
                for i, bundle in enumerate(bundles):
                    truth = bundle["features"].numpy()
                    prior = quality(truth, bundle["mean_mode"], stats)
                    mse = prior["standardized_mse"]
                    rows.append(dict(index=i, source_ids=bundle["target"]["source_ids"], client=bundle["target"]["client"],
                                     estimator=quality(truth, guesses[i], stats, mse), mean_mode=prior,
                                     empirical_marginal=quality(truth, bundle["empirical_marginal"], stats, mse),
                                     decoy=quality(truth, guesses[(i+1) % 8], stats, mse)))
                medians = {arm: {metric: float(np.median([r[arm][metric] for r in rows]))
                           for metric in ("feature_accuracy", "standardized_mse", "bounded_score")}
                           for arm in ("estimator", "mean_mode", "empirical_marginal", "decoy")}
                comparison = gate(rows, 8)
            cosines = [result_receipt(D / "observations" / f"{cell['dataset']}_r10_b4_n8_{i:02d}" / "receipt.json")["cosines"] for i in range(8)]
            doc = dict(cell=cell, condition=condition, descriptive_only=True, rows=rows, medians=medians,
                       comparisons=comparison, cosines=cosines,
                       diagnostic_beats_all_controls=comparison["passed"], qualification_unchanged=True)
            path = D / "per_target" / condition / (identifier+".json")
            if path.exists():
                if load(path) != doc:
                    raise ValueError("diagnostic evaluation changed")
            else:
                write(path, doc, exclusive=True)
            summaries.append(doc)
    path = D / "summary.json"
    doc = dict(descriptive_only=True, cells=18, summaries=summaries, qualification_count=0)
    if path.exists():
        if load(path) != doc:
            raise ValueError("diagnostic summary changed")
    else:
        write(path, doc, exclusive=True)


def supervisor():
    torch.set_num_threads(1)
    with socket.socket() as lock:
        lock.bind(("127.0.0.1", PORT))
        try:
            frozen()
            stage("diagnostic_observations", status="DIAGNOSTIC_RUNNING", supervisor_pid=os.getpid(), no_live_scientific_workers=False)
            jobs = []
            for cell in selected_cells():
                for i in range(8):
                    rec = capture(cell["dataset"], i)
                    for condition in ("D1", "D2"):
                        path = D / "jobs" / condition / cell_id(cell) / f"{i:02d}.json"
                        if path.exists():
                            result_receipt(path)
                            continue
                        jobs.append(dict(cell=cell, condition=condition, index=i, result_path=str(path.relative_to(ROOT)),
                                         payload_path=rec["payload_path"], payload_sha256=rec["hashes"][rec["payload_path"]], freeze_sha256=sha(D / "freeze.json")))
            errors, done = [], 96-len(jobs)
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
                futures = {pool.submit(worker, job): job for job in jobs}
                write(D / ("worker_pool_"+str(time.time_ns())+".json"),
                      dict(at=now(), worker_pids=sorted(pool._processes), supervisor_pid=os.getpid()), exclusive=True)
                for future in as_completed(futures):
                    try:
                        future.result()
                        done += 1
                    except BaseException:
                        errors.append(dict(job=futures[future], traceback=traceback.format_exc()))
                    stage("diagnostic_reconstruction", status="DIAGNOSTIC_RUNNING", diagnostic_jobs_done=done,
                          diagnostic_jobs_pending=96-done-len(errors), failed_jobs=len(errors))
            if errors:
                write(D / ("REQUIRES_DIRECTION_"+str(time.time_ns())+".json"), dict(errors=errors), exclusive=True)
                raise ValueError("submitted pool drained; diagnostic failure requires direction")
            frozen()
            evaluate()
            stage("diagnostic_measurements_finished", status="DIAGNOSTIC_AWAITING_INDEPENDENT_AUDIT", diagnostic_jobs_done=96,
                  diagnostic_jobs_pending=0, diagnostic_measurements_complete=True, no_live_scientific_workers=False)
        except BaseException:
            write(D / ("SUPERVISOR_FAILURE_"+str(time.time_ns())+".json"), dict(at=now(), traceback=traceback.format_exc()), exclusive=True)
            stage("diagnostic_requires_direction", status="REQUIRES_DIRECTION", required_direction="preserved diagnostic failure; diagnose before any fix/replay")
            raise


def launch(resume):
    frozen()
    audit = no_live_audit()
    if (D / "launch_receipt.json").exists() and not resume:
        raise ValueError("launch already exists; no implicit restart")
    if list(D.rglob("failure.json")) or list(D.glob("*FAILURE*.json")) or list(D.glob("REQUIRES_DIRECTION*.json")):
        raise ValueError("failure needs explicit direction, never automatic replay")
    if (D / "summary.json").exists():
        raise ValueError("measurements finished; do not rerun")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--supervisor"]
    attempt = D / "launches" / str(time.time_ns())
    attempt.mkdir(parents=True, exist_ok=False)
    with (attempt / "stdout.log").open("x") as stdout, (attempt / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr, start_new_session=True)
    doc = dict(at=now(), pid=child.pid, command=command, resume=resume, no_live_audit=audit,
               stdout=str((attempt / "stdout.log").relative_to(ROOT)), stderr=str((attempt / "stderr.log").relative_to(ROOT)), freeze_sha256=sha(D / "freeze.json"))
    write(attempt / "receipt.json", doc, exclusive=True)
    if not (D / "launch_receipt.json").exists():
        write(D / "launch_receipt.json", doc, exclusive=True)
    print(doc, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supervisor", action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.launch, args.supervisor)) != 1:
        parser.error("choose one action")
    if args.prepare:
        prepare()
    elif args.launch:
        launch(args.resume)
    else:
        supervisor()
