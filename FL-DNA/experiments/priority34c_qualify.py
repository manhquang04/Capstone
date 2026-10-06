"""Detached source-frozen P34C qualification only; no confirmatory target access."""
from __future__ import annotations
import argparse
import json
import os
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.priority34c_core import OUT, PREPARED, native, np, torch, sha, write
from experiments.priority34c_core import PreparedDataset, prepare_adapters, ratio_model, gradient, ratio_decode
from experiments.priority34c_firewall import reserve, validate, CELLS
from privacy.seed_manager import derive_seed

DATASETS = ("paysim", "ieee_cis", "baf")
FREEZE = OUT / "qualification_freeze.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, doc):
    with path.open("a") as handle:
        handle.write(json.dumps(doc, sort_keys=True, allow_nan=False) + "\n")


def live_processes():
    if os.name == "nt":
        command = ["powershell", "-NoProfile", "-Command", "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json"]
        rows = json.loads(subprocess.check_output(command, text=True))
        return [row for row in rows if row["ProcessId"] != os.getpid() and
                "priority34c_qualify.py" in (row.get("CommandLine") or "") and
                any(flag in row["CommandLine"] for flag in ("--supervise", "--job"))]
    rows = subprocess.check_output(["ps", "-axo", "pid,command"], text=True).splitlines()
    return [row for row in rows if "priority34c_qualify.py" in row and
            "python" in row.lower() and int(row.strip().split(None, 1)[0]) != os.getpid() and
            any(flag in row for flag in ("--supervise", "--job"))]


def verify_freeze():
    doc = json.loads(FREEZE.read_text())
    for section in ("sources", "inputs"):
        for name, digest in doc[section].items():
            if sha(ROOT / name) != digest:
                raise ValueError("frozen " + section + " drift: " + name)
    return doc


def prepare():
    if live_processes():
        raise RuntimeError("P34C workers alive; refuse preparation")
    if FREEZE.exists():
        return verify_freeze()
    prerequisite = OUT.parent / "priority34b_local_dp/COMPLETE.json"
    complete = json.loads(prerequisite.read_text())
    if complete["status"] != "PASS" or sha(ROOT / complete["report"]) != complete["report_sha256"]:
        raise ValueError("local DP utility prerequisite not verified")
    inputs = {str(prerequisite.relative_to(ROOT)): sha(prerequisite), complete["report"]: complete["report_sha256"]}
    for dataset in DATASETS:
        history_path = OUT / "history" / (dataset + ".json")
        history = json.loads(history_path.read_text())
        for name, digest in history.get("additional_inputs", {}).items():
            if sha(ROOT / name) != digest:
                raise ValueError("historical reconstruction input drift")
            inputs[name] = digest
        for row in history["provenance"]:
            if sha(ROOT / row["path"]) != row["sha256"]:
                raise ValueError("historical input drift")
            inputs[row["path"]] = row["sha256"]
        prepare_adapters(dataset)
        source = np.load(PREPARED / dataset / "train_source_ids.npy")
        manifest = reserve(dataset, source, history["exclusions"])
        manifest_path = OUT / dataset / "targets.json"
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("target reservation drift")
        if not manifest_path.exists():
            write(manifest_path, manifest)
        validate(manifest, source, history["exclusions"])
        for path in list((PREPARED / dataset).glob("*")) + [history_path, manifest_path]:
            if path.is_file():
                inputs[str(path.relative_to(ROOT))] = sha(path)
        for cell, _ in CELLS:
            path = OUT / dataset / cell / "adapter.json"
            inputs[str(path.relative_to(ROOT))] = sha(path)
    sources = {}
    for folder in ("dna_encoder", "privacy", "external_defenses/tableak"):
        for path in (ROOT / folder).rglob("*.py"):
            sources[str(path.relative_to(ROOT))] = sha(path)
    paths = list((ROOT / "experiments").glob("priority34c*.py"))
    paths += list((ROOT / "tests").glob("test_priority34c*.py"))
    paths += list((ROOT / "protocols/amendments").glob("*priority34c*.md"))
    paths += [ROOT / name for name in ("experiments/priority33b_core.py", "experiments/priority29/tabular_native_positive_control.py",
              "experiments/priority30_native_defenses/run_audit.py", "experiments/priority30_native_defenses/native_adapters.py", "models/fraud_mlp.py")]
    for path in paths:
        sources[str(path.relative_to(ROOT))] = sha(path)
    doc = dict(at=now(), stage="qualification_only", sources=sources, inputs=inputs,
               cells=9, n8=8, n24=24, workers=4, device="cpu", intra_threads=1, inter_threads=1,
               confirmatory_authorized=False, private_noise_keys_needed=False)
    write(FREEZE, doc)
    return verify_freeze()


def folder_for(config):
    return OUT / config["dataset"] / config["cell"] / config["stage"] / ("target_%03d" % config["index"])


def valid_result(path, config):
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc["status"] != "COMPLETED" or doc["config"] != config or doc["freeze_sha256"] != sha(FREEZE):
        return False
    for name, digest in doc["files"].items():
        if sha(path.parent / doc["attempt"] / name) != digest:
            raise ValueError("result artifact drift")
    return True


def job(config):
    verify_freeze()
    if config["stage"] not in ("n8", "n24"):
        raise ValueError("qualification-only target access")
    if config["stage"] == "n24":
        gate = json.loads((OUT / "gates_n8.json").read_text())
        if not gate[config["dataset"] + "/" + config["cell"]]["passed"]:
            raise ValueError("n24 blocked by n8 gate")
    folder = folder_for(config)
    attempt = "attempt_" + str(time.time_ns())
    destination = folder / attempt
    destination.mkdir(parents=True)
    started = time.time()
    try:
        data = PreparedDataset(config["dataset"], config["cell"])
        manifest = json.loads((OUT / config["dataset"] / "targets.json").read_text())
        ids = manifest["source_ids"][config["cell"]][config["stage"]][config["index"]]
        source = np.load(PREPARED / config["dataset"] / "train_source_ids.npy")
        mapping = {int(value): i for i, value in enumerate(source)}
        indices = [mapping[value] for value in ids]
        truth, labels = data.Xtrain[indices].clone(), data.ytrain[indices].clone()
        ratio = config["cell"] == "ratio_batch1"
        net = ratio_model(data.num_features) if ratio else native.make_model(data.num_features)
        model_input = data.de_standardize(truth) if ratio else truth
        raw, bn_receipt = gradient(net, model_input, labels, ratio=ratio)
        torch.save(dict(payload=raw, labels=labels, public_model=net.state_dict(), bn=bn_receipt), destination / "server_receipt.pt")
        torch.save(dict(truth=truth, labels=labels, source_ids=ids, indices=indices), destination / "scoring_truth.pt")
        seed = derive_seed(340400, "p34c-attack", config["dataset"], config["cell"], config["stage"], config["index"])
        if ratio:
            recovered, solver_receipt = ratio_decode(list(raw), list(raw.values()))
            recovered = (recovered-data.mean)/data.std
            ensemble, losses = [], []
        else:
            recovered, ensemble, losses = native.native_recover(net, data, list(raw.values()), labels, seed)
            solver_receipt = None
        score = native.measure(data, truth, recovered)
        control_seed = derive_seed(340500, "p34c-control", config["dataset"], config["cell"], config["stage"], config["index"])
        controls, mean_mode, empirical = native.controls(data, truth, control_seed)
        torch.save(dict(reconstruction=recovered, ensemble=ensemble, objectives=losses,
                        mean_mode=mean_mode, empirical_single=empirical), destination / "reconstruction.pt")
        result = dict(status="COMPLETED", config=config, freeze_sha256=sha(FREEZE), attempt=attempt,
                      accuracy=score, baselines=controls, bn=bn_receipt, solver=solver_receipt,
                      source_ids=ids, attack_seed=seed, control_seed=control_seed,
                      device="cpu", intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(),
                      finished_at=now(), elapsed_seconds=time.time()-started,
                      files={p.name: sha(p) for p in destination.glob("*.pt")})
        write(folder / "result.json", result)
    except Exception:
        failure = dict(status="FAILED", config=config, attempt=attempt, at=now(), error=traceback.format_exc())
        write(destination / "failure.json", failure)
        raise


def execute(config):
    folder = folder_for(config)
    if valid_result(folder / "result.json", config):
        return json.loads((folder / "result.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--job", json.dumps(config, sort_keys=True)]
    append(OUT / "runs.jsonl", dict(at=now(), event="qualification_started", config=config, command=command))
    with (folder / "stdout.log").open("a") as stdout, (folder / "stderr.log").open("a") as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if process.returncode != 0 or not valid_result(folder / "result.json", config):
        raise RuntimeError("qualification job failed: " + json.dumps(config))
    return json.loads((folder / "result.json").read_text())


def gates(rows):
    result = {}
    for dataset in DATASETS:
        for cell, _ in CELLS:
            selected = [row for row in rows if row["config"]["dataset"] == dataset and row["config"]["cell"] == cell]
            if not selected:
                continue
            tests = {}
            for name in ("mean_mode", "empirical_mean"):
                diffs = [row["accuracy"]["accuracy_percent"]-row["baselines"][name] for row in selected]
                wins, losses = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
                tests[name] = dict(wins=wins, losses=losses, ties=len(diffs)-wins-losses, p=native.exact_p(wins, losses))
            result[dataset + "/" + cell] = dict(n=len(selected), tests=tests,
                      passed=all(test["p"] < .05 for test in tests.values()))
    return result


def supervise():
    owner = socket.socket()
    owner.bind(("127.0.0.1", 43463))
    owner.listen(1)
    verify_freeze()
    write(OUT / "qualification_supervisor.lock.json", dict(pid=os.getpid(), started_at=now(), port=43463))
    for stage, n in (("n8", 8), ("n24", 24)):
        eligible = None if stage == "n8" else json.loads((OUT / "gates_n8.json").read_text())
        configs = [dict(dataset=d, cell=c, stage=stage, index=i)
                   for d in DATASETS for c, _ in CELLS
                   if eligible is None or eligible[d + "/" + c]["passed"] for i in range(n)]
        rows, failures = [], []
        started = time.time()
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = {pool.submit(execute, config): config for config in configs}
            for future in as_completed(futures):
                config = futures[future]
                try:
                    rows.append(future.result())
                except Exception:
                    failures.append(dict(config=config, error=traceback.format_exc()))
                done = len(rows)+len(failures)
                progress = dict(part="qualification_" + stage, dataset=config["dataset"], defense="unprotected",
                          comparator="mean_mode_and_empirical_marginal", done=done, total=len(configs), failed=len(failures),
                          last_update=now(), eta_minutes=(time.time()-started)/done*(len(configs)-done)/60)
                write(OUT / "progress.json", progress)
                append(OUT / "progress.log", progress)
                append(OUT / "runs.jsonl", dict(event="qualification_finished", at=now(), config=config, failed=config in [f["config"] for f in failures]))
        if failures:
            write(OUT / "REQUIRES_DIRECTION_QUALIFICATION.json", dict(at=now(), failures=failures, stage=stage, pool_drained=True))
            return
        write(OUT / ("gates_" + stage + ".json"), gates(rows))
    write(OUT / "QUALIFICATION_COMPLETE.json", dict(at=now(), status="STAGE_COMPLETE_ONLY", freeze_sha256=sha(FREEZE), confirmatory_started=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--job")
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        if live_processes():
            raise RuntimeError("owned supervisor/workers alive; no duplicate")
        if (OUT / "REQUIRES_DIRECTION_QUALIFICATION.json").exists():
            raise RuntimeError("scientific failure requires explicit direction; no automatic resume")
        if list(OUT.glob("qualification_launch_*.json")) and not args.resume:
            raise RuntimeError("prior launch exists; infrastructure review and explicit resume required")
        verify_freeze()
        with (OUT / "qualification_stdout.log").open("a") as stdout, (OUT / "qualification_stderr.log").open("a") as stderr:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--supervise"],
                    cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=os.name != "nt")
        write(OUT / ("qualification_launch_" + str(time.time_ns()) + ".json"), dict(at=now(), pid=process.pid, freeze_sha256=sha(FREEZE)))
    elif args.supervise:
        supervise()
    elif args.job:
        job(json.loads(args.job))
    else:
        parser.error("choose one execution mode")


if __name__ == "__main__":
    main()
