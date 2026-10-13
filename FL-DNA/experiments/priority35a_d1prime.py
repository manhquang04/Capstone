"""Detached, frozen, attempt-preserving descriptive D1-prime execution."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from pathlib import Path
import socket
import subprocess
import sys
import time
import traceback
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments.priority35a_runtime import ROOT, OUT, Q, load, write, tensor_save, sha, check_hashes, now
from experiments.priority35a_diagnostic import selected_cells, result_receipt as old_receipt
from experiments.priority35a_qualification import valid_receipt, cell_id
from experiments.priority35a_evaluation import quality, gate
from experiments.priority35a_d1prime_estimator import capture_masks, reconstruct
from experiments.priority35a_report import checks

P = OUT / "d1prime"
AMENDMENT = ROOT / "protocols/amendments/2026-10-10_priority35a_d1prime_descriptive.md"
REPORT = ROOT / "reports/priority35a_report.md"
PORT = 35937


def progress(stage, **fields):
    doc = load(OUT / "progress.json")
    doc.update(d1prime_stage=stage, d1prime_updated=now(), d1prime_complete=False)
    doc.update(fields)
    # Original Priority35A COMPLETE status is never reversed.
    write(OUT / "progress.json", doc)
    event = dict(at=now(), diagnostic="D1-prime", stage=stage, **fields)
    with (OUT / "progress.log").open("a") as handle:
        import json
        handle.write(json.dumps(event, allow_nan=False)+"\n")
    checklist = load(OUT / "checklist.json")
    checklist.update(d1prime_stage=stage, d1prime_complete=fields.get("d1prime_complete", False))
    write(OUT / "checklist.json", checklist)
    print(event, flush=True)


def no_live():
    lines = subprocess.run(["ps", "-axo", "pid=,args="], capture_output=True, text=True, check=True).stdout.splitlines()
    found = []
    for line in lines:
        parts = line.strip().split(None, 1)
        if len(parts) != 2 or int(parts[0]) == os.getpid():
            continue
        command = parts[1]
        if "python" not in Path(command.split()[0]).name.lower():
            continue
        if any(script in command for script in ("priority35a_d1prime.py", "priority35a_qualification.py", "priority35a_diagnostic.py")):
            found.append(line)
    for receipt in P.glob("worker_pool_*.json"):
        for pid in load(receipt)["worker_pids"]:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            else:
                found.append("recorded worker PID "+str(pid))
    if found:
        raise ValueError("no duplicate while live: "+repr(found))
    with socket.socket() as lock:
        lock.bind(("127.0.0.1", PORT))
    return dict(at=now(), no_live_scientific_workers=True)


def frozen():
    document = load(P / "freeze.json")
    check_hashes(document["hashes"])
    if sha(REPORT) != document["base_report_sha256"]:
        raise ValueError("base report changed before append")
    return document


def prepare():
    if (P / "freeze.json").exists():
        raise ValueError("D1-prime already frozen")
    no_live()
    complete = load(OUT / "COMPLETE.json")
    if complete["status"] != "VERIFIED_COMPLETE" or complete["qualified_cells"] != 0:
        raise ValueError("original P35A must remain verified COMPLETE0/72")
    check_hashes(load(ROOT / complete["manifest_path"])["hashes"])
    if sha(REPORT) != complete["report_sha256"]:
        raise ValueError("original report digest")
    P.mkdir(parents=True, exist_ok=True)
    with (P / "base_report.md").open("xb") as handle:
        handle.write(REPORT.read_bytes())
    folder = P / "preflight" / str(time.time_ns())
    sources = checks(folder)
    paths = sources + [AMENDMENT, P / "base_report.md", OUT / "COMPLETE.json", ROOT / complete["manifest_path"],
                       ROOT / "external_defenses/tableak/run_inversion_attacks.py",
                       ROOT / "external_defenses/tableak/attacks/ensembling.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    hashes.update(load(ROOT / complete["manifest_path"])["hashes"])
    hashes.update({str(p.relative_to(ROOT)): sha(p) for p in folder.glob("*.json")})
    write(P / "freeze.json", dict(at=now(), hashes=hashes, amendment_sha256=sha(AMENDMENT),
          base_report_sha256=sha(REPORT), base_report_bytes=REPORT.stat().st_size,
          cells=selected_cells(), conditions=["a", "b"], jobs=96, budgets=dict(a=[2,4000], b=[30,1500]),
          workers=4, threads=1, ordered_labels_direct_user_approval=True, descriptive_only=True,
          optional_c="NOT_RUN; prospectively omitted"), exclusive=True)
    progress("FROZEN", d1prime_jobs_total=96, d1prime_jobs_done=0, d1prime_failures=0,
             d1prime_freeze_sha256=sha(P / "freeze.json"), d1prime_pending=["measurement", "independent verification", "append-only report"])


def receipt(path):
    doc = load(path)
    if doc["status"] != "COMPLETED" or doc["freeze_sha256"] != sha(P / "freeze.json"):
        raise ValueError("D1-prime receipt mismatch")
    check_hashes(doc["hashes"])
    return doc


def capture(dataset, index):
    identity = f"{dataset}_r10_b4_n8_{index:02d}"
    path = P / "observations" / identity / "receipt.json"
    if path.exists():
        return receipt(path)
    original = valid_receipt(Q / "uploads" / identity / "receipt.json")
    public = torch.load(ROOT / original["public_path"], map_location="cpu", weights_only=False)
    private = torch.load(ROOT / original["private_path"], map_location="cpu", weights_only=False)
    masks, grads = capture_masks(public, private)
    d1 = old_receipt(OUT / "diagnostic/observations" / identity / "receipt.json")
    original_payload = torch.load(ROOT / d1["payload_path"], map_location="cpu", weights_only=False)
    if any(not torch.equal(g, original_payload["D1"][k]) for k, g in grads.items()):
        raise ValueError("native gradient differs from original D1")
    payload = dict(shared=public["shared"], gradient=grads, bn=private["client_bn"], masks=masks,
                   labels=private["labels"], meta=public["meta"], ordinal=identity)
    target = dict(identity=identity, source_ids=private["target"]["source_ids"], client=private["target"]["client"],
                  original_receipt=str((Q / "uploads" / identity / "receipt.json").relative_to(ROOT)))
    tensor = path.parent / "privileged_payload.pt"
    tensor_save(tensor, payload, private=True)
    write(path, dict(status="COMPLETED", freeze_sha256=sha(P / "freeze.json"), identity=identity,
          payload_path=str(tensor.relative_to(ROOT)), hashes={str(tensor.relative_to(ROOT)):sha(tensor)},
          target=target, native_mask_logits_gradient_bit_exact=True, torch_threads=1), exclusive=True)
    return receipt(path)


def worker(job):
    torch.set_num_threads(1)
    attempt = ROOT / job["result_path"]
    attempt = attempt.parent / ("attempt_"+str(time.time_ns()))
    attempt.mkdir(parents=True, exist_ok=False)
    write(attempt / "job.json", job, exclusive=True)
    try:
        check_hashes({job["payload_path"]:job["payload_sha256"]})
        payload = torch.load(ROOT / job["payload_path"], map_location="cpu", weights_only=False)
        result = reconstruct(payload, job["cell"]["instrument"], job["condition"])
        if result["synthetic"] or result["torch_threads"] != 1:
            raise ValueError("scientific worker contract")
        tensor = attempt / "reconstruction.pt"
        tensor_save(tensor, result, private=True)
        doc = dict(status="COMPLETED", freeze_sha256=job["freeze_sha256"], job=job,
                   reconstruction_path=str(tensor.relative_to(ROOT)), elapsed_seconds=result["elapsed_seconds"],
                   hashes={str(p.relative_to(ROOT)):sha(p) for p in (tensor, attempt / "job.json")})
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
            r = valid_receipt(Q / "uploads" / f"{cell['dataset']}_r10_b4_n8_{i:02d}" / "receipt.json")
            bundles.append(torch.load(ROOT / r["private_path"], map_location="cpu", weights_only=False))
        for condition in ("a", "b"):
            guesses = []
            for i in range(8):
                r = receipt(P / "jobs" / condition / identifier / f"{i:02d}.json")
                guesses.append(torch.load(ROOT / r["reconstruction_path"], map_location="cpu", weights_only=False)["rows"].numpy())
            rows = []
            for i, bundle in enumerate(bundles):
                truth = bundle["features"].numpy()
                prior = quality(truth, bundle["mean_mode"], stats)
                mse = prior["standardized_mse"]
                rows.append(dict(index=i, source_ids=bundle["target"]["source_ids"], estimator=quality(truth, guesses[i], stats, mse),
                                 mean_mode=prior, empirical_marginal=quality(truth, bundle["empirical_marginal"], stats, mse),
                                 decoy=quality(truth, guesses[(i+1)%8], stats, mse)))
            medians = {arm:{metric:float(np.median([r[arm][metric] for r in rows]))
                       for metric in ("feature_accuracy", "standardized_mse", "bounded_score")}
                       for arm in ("estimator", "mean_mode", "empirical_marginal", "decoy")}
            comparison = gate(rows, 8)
            doc = dict(cell=cell, condition=condition, rows=rows, medians=medians, comparisons=comparison,
                       descriptive_only=True, diagnostic_beats_all_controls=comparison["passed"])
            write(P / "per_target" / condition / (identifier+".json"), doc, exclusive=True)
            summaries.append(doc)
    write(P / "summary.json", dict(summaries=summaries, cells=12, paired_batches_per_cell=8,
                                   original_qualification_unchanged=True), exclusive=True)


def supervisor():
    torch.set_num_threads(1)
    with socket.socket() as lock:
        lock.bind(("127.0.0.1", PORT))
        try:
            frozen()
            progress("CAPTURING_NATIVE_MASKS", d1prime_supervisor_pid=os.getpid(), d1prime_no_live_workers=False)
            jobs = []
            targets = []
            for cell in selected_cells():
                for i in range(8):
                    r = capture(cell["dataset"], i)
                    if cell["instrument"] == "tableak":
                        targets.append(r["target"])
                    for condition in ("a", "b"):
                        path = P / "jobs" / condition / cell_id(cell) / f"{i:02d}.json"
                        if path.exists():
                            receipt(path)
                            continue
                        jobs.append(dict(cell=cell, condition=condition, index=i, result_path=str(path.relative_to(ROOT)),
                                         payload_path=r["payload_path"], payload_sha256=r["hashes"][r["payload_path"]],
                                         freeze_sha256=sha(P / "freeze.json")))
            if not (P / "target_manifest.json").exists():
                write(P / "target_manifest.json", dict(targets=targets, distinct_batches=24, reused_original_D1=True), exclusive=True)
            errors, done = [], 96-len(jobs)
            with ProcessPoolExecutor(max_workers=4, mp_context=mp.get_context("spawn")) as pool:
                futures = {pool.submit(worker, job):job for job in jobs}
                write(P / ("worker_pool_"+str(time.time_ns())+".json"),
                      dict(at=now(), worker_pids=sorted(pool._processes), supervisor_pid=os.getpid()), exclusive=True)
                for future in as_completed(futures):
                    try:
                        future.result()
                        done += 1
                    except BaseException:
                        errors.append(dict(job=futures[future], traceback=traceback.format_exc()))
                    progress("RECONSTRUCTING", d1prime_jobs_done=done, d1prime_jobs_pending=96-done-len(errors), d1prime_failures=len(errors))
            if errors:
                write(P / ("REQUIRES_DIRECTION_"+str(time.time_ns())+".json"), dict(errors=errors), exclusive=True)
                raise ValueError("all submitted jobs drained; no scientific replay")
            frozen()
            evaluate()
            progress("MEASUREMENTS_FINISHED", d1prime_jobs_done=96, d1prime_jobs_pending=0,
                     d1prime_pending=["independent verification and report append"])
        except BaseException:
            write(P / ("SUPERVISOR_FAILURE_"+str(time.time_ns())+".json"), dict(at=now(), traceback=traceback.format_exc()), exclusive=True)
            progress("REQUIRES_DIRECTION", d1prime_required_direction="preserved failure; diagnose read-only before fix/replay")
            raise


def launch(resume):
    frozen()
    audit = no_live()
    if (P / "launch_receipt.json").exists() and not resume:
        raise ValueError("no implicit restart")
    if list(P.rglob("failure.json")) or list(P.glob("*FAILURE*.json")) or list(P.glob("REQUIRES_DIRECTION*.json")):
        raise ValueError("failure requires user direction")
    if (P / "summary.json").exists():
        raise ValueError("measurements already finished")
    command = [sys.executable, "-B", str(Path(__file__).resolve()), "--supervisor"]
    folder = P / "launches" / str(time.time_ns())
    folder.mkdir(parents=True, exist_ok=False)
    with (folder / "stdout.log").open("x") as stdout, (folder / "stderr.log").open("x") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, env=os.environ.copy(), stdout=stdout, stderr=stderr, start_new_session=True)
    doc = dict(at=now(), pid=child.pid, command=command, resume=resume, no_live_audit=audit,
               stdout=str((folder / "stdout.log").relative_to(ROOT)), stderr=str((folder / "stderr.log").relative_to(ROOT)),
               freeze_sha256=sha(P / "freeze.json"))
    write(folder / "receipt.json", doc, exclusive=True)
    if not (P / "launch_receipt.json").exists():
        write(P / "launch_receipt.json", doc, exclusive=True)
    progress("LAUNCHED", d1prime_supervisor_pid=child.pid, d1prime_pending=["measurement", "independent verification", "append-only report"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    for flag in ("prepare", "launch", "supervisor", "finish", "resume"):
        parser.add_argument("--"+flag, action="store_true")
    args = parser.parse_args()
    if sum((args.prepare, args.launch, args.supervisor, args.finish)) != 1:
        parser.error("choose exactly one action")
    if args.prepare:
        prepare()
    elif args.launch:
        launch(args.resume)
    elif args.supervisor:
        supervisor()
    else:
        from experiments.priority35a_d1prime_verify import finish
        finish()
