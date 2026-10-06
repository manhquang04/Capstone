"""P34C defender-only developmental matching; never attacks n39 records."""
from __future__ import annotations
import argparse
import json
import os
import secrets
import socket
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments.priority34c_core import OUT, PREPARED, native, np, torch, sha, write, PreparedDataset
from experiments.priority34c_core import ratio_model, gradient, defend, reconstruct_update_array_v2
from experiments.priority34c_qualify import verify_freeze as verify_qualification, valid_result, now, append, live_processes as qualification_processes

FREEZE = OUT / "development_freeze.json"
PRIVATE = OUT / "development_private_noise.json"
METHODS = ("dna_v1_conservative", "dna_v2_0p95")


def configs():
    gates = json.loads((OUT / "gates_n24.json").read_text())
    return [dict(dataset=key.split("/")[0], cell=key.split("/")[1], index=i, stage="development24")
            for key, gate in sorted(gates.items()) if gate["passed"] for i in range(24)]


def key_for(config, method):
    return "/".join((config["dataset"], config["cell"], method, str(config["index"])))


def own_processes():
    if os.name == "nt":
        rows = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json"], text=True))
        return [row for row in rows if row["ProcessId"] != os.getpid() and "priority34c_development.py" in (row.get("CommandLine") or "")
                and any(flag in row["CommandLine"] for flag in ("--supervise", "--job"))]
    rows = subprocess.check_output(["ps", "-axo", "pid,command"], text=True).splitlines()
    return [row for row in rows if "priority34c_development.py" in row and "python" in row.lower()
            and int(row.strip().split(None, 1)[0]) != os.getpid() and any(flag in row for flag in ("--supervise", "--job"))]


def verify():
    verify_qualification()
    doc = json.loads(FREEZE.read_text())
    for section in ("sources", "inputs"):
        for name, digest in doc[section].items():
            if sha(ROOT / name) != digest:
                raise ValueError("development frozen input drift: " + name)
    return doc


def prepare():
    if qualification_processes() or own_processes():
        raise RuntimeError("P34C workload alive; no overlapping stage")
    if (OUT / "REQUIRES_DIRECTION_QUALIFICATION.json").exists():
        raise RuntimeError("qualification scientific failure unresolved")
    if not (OUT / "QUALIFICATION_COMPLETE.json").exists():
        raise RuntimeError("qualification incomplete")
    verify_qualification()
    if FREEZE.exists():
        return verify()
    inputs = {str((OUT / name).relative_to(ROOT)): sha(OUT / name)
              for name in ("qualification_freeze.json", "QUALIFICATION_COMPLETE.json", "gates_n8.json", "gates_n24.json")}
    n24 = json.loads((OUT / "gates_n24.json").read_text())
    for cell, gate in n24.items():
        if not gate["passed"]:
            continue
        dataset, instrument = cell.split("/")
        for stage, n in (("n8", 8), ("n24", 24)):
            for i in range(n):
                config = dict(dataset=dataset, cell=instrument, stage=stage, index=i)
                path = OUT / dataset / instrument / stage / ("target_%03d" % i) / "result.json"
                if not valid_result(path, config):
                    raise ValueError("qualification receipt incomplete")
                inputs[str(path.relative_to(ROOT))] = sha(path)
    batch = configs()
    if not PRIVATE.exists():
        write(PRIVATE, {key_for(config, method): secrets.randbits(63) for config in batch for method in METHODS})
        PRIVATE.chmod(0o600)
    inputs[str(PRIVATE.relative_to(ROOT))] = sha(PRIVATE)
    sources = {}
    for path in [Path(__file__).resolve(), ROOT / "protocols/amendments/2026-10-04_priority34c_development_execution.md",
                 ROOT / "tests/test_priority34c_development.py"]:
        sources[str(path.relative_to(ROOT))] = sha(path)
    doc = dict(at=now(), scope="development_only_no_confirmatory", sources=sources, inputs=inputs,
               jobs=batch, workers=4, device="cpu", threads=1, methods=list(METHODS))
    write(FREEZE, doc)
    return verify()


def folder_for(config):
    return OUT / config["dataset"] / config["cell"] / "development24" / ("target_%03d" % config["index"])


def valid(path, config):
    if not path.exists():
        return False
    row = json.loads(path.read_text())
    if row["status"] != "COMPLETED" or row["config"] != config or row["freeze_sha256"] != sha(FREEZE):
        return False
    for name, digest in row["files"].items():
        if sha(path.parent / row["attempt"] / name) != digest:
            raise ValueError("development artifact drift")
    return True


def job(config):
    verify()
    if config not in configs() or config["stage"] != "development24":
        raise ValueError("unauthorized development access")
    folder = folder_for(config)
    attempt = "attempt_" + str(time.time_ns())
    target = folder / attempt
    target.mkdir(parents=True)
    try:
        data = PreparedDataset(config["dataset"], config["cell"])
        manifest = json.loads((OUT / config["dataset"] / "targets.json").read_text())
        ids = manifest["source_ids"][config["cell"]]["development24"][config["index"]]
        source = np.load(PREPARED / config["dataset"] / "train_source_ids.npy")
        mapping = {int(value): i for i, value in enumerate(source)}
        indices = [mapping[value] for value in ids]
        x, labels = data.Xtrain[indices].clone(), data.ytrain[indices].clone()
        ratio = config["cell"] == "ratio_batch1"
        net = ratio_model(data.num_features) if ratio else native.make_model(data.num_features)
        raw, bn = gradient(net, data.de_standardize(x) if ratio else x, labels, ratio=ratio)
        raw_values = list(raw.values())
        norm = float(torch.sqrt(sum(g.double().square().sum() for g in raw_values)))
        secret = json.loads(PRIVATE.read_text())
        measurements = {}
        for method in METHODS:
            payload, _, meta = defend(raw, method)
            if meta is None:
                decoded = payload
            else:
                decoded = [torch.from_numpy(reconstruct_update_array_v2(q.numpy(), m)[0]) for q, m in zip(payload, meta)]
            distortion = float(torch.sqrt(sum((g.double()-q.double()).square().sum() for g, q in zip(raw_values, decoded))))
            gen = torch.Generator(device="cpu").manual_seed(secret[key_for(config, method)])
            gaussian = float(torch.sqrt(sum(torch.randn(g.shape, dtype=torch.float64, generator=gen).square().sum() for g in raw_values)))
            native.finite([distortion, gaussian, norm], "development norms")
            measurements[method] = dict(dna_l2_distortion=distortion, unit_gaussian_norm=gaussian)
        torch.save(dict(raw_gradient=raw, public_model=net.state_dict(), source_ids=ids), target / "defender_calibration_private.pt")
        row = dict(status="COMPLETED", config=config, attempt=attempt, freeze_sha256=sha(FREEZE),
                   source_ids=ids, raw_norm=norm, measurements=measurements, bn=bn,
                   files={"defender_calibration_private.pt": sha(target / "defender_calibration_private.pt")},
                   device="cpu", intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(), at=now())
        write(folder / "result.json", row)
    except Exception:
        write(target / "failure.json", dict(config=config, at=now(), error=traceback.format_exc()))
        raise


def execute(config):
    folder = folder_for(config)
    path = folder / "result.json"
    if valid(path, config):
        return json.loads(path.read_text())
    folder.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--job", json.dumps(config, sort_keys=True)]
    with (folder / "stdout.log").open("a") as stdout, (folder / "stderr.log").open("a") as stderr:
        process = subprocess.run(command, cwd=ROOT, stdout=stdout, stderr=stderr)
    if process.returncode or not valid(path, config):
        raise RuntimeError("development job failed: " + json.dumps(config))
    return json.loads(path.read_text())


def calibration(rows):
    results = {}
    for cell in sorted({row["config"]["dataset"] + "/" + row["config"]["cell"] for row in rows}):
        selected = sorted([row for row in rows if row["config"]["dataset"] + "/" + row["config"]["cell"] == cell], key=lambda row: row["config"]["index"])
        if len(selected) != 24:
            raise ValueError("missing development observations")
        clip = 1.01 * max(row["raw_norm"] for row in selected)
        methods = {}
        for method in METHODS:
            distortions = np.asarray([row["measurements"][method]["dna_l2_distortion"] for row in selected])
            gaussian = np.asarray([row["measurements"][method]["unit_gaussian_norm"] for row in selected])
            if clip <= 0 or np.median(gaussian) <= 0:
                raise ValueError("invalid developmental signal")
            sigma = float(np.median(distortions) / (clip*np.median(gaussian)))
            errors = np.divide(np.abs(clip*sigma*gaussian-distortions), distortions,
                               out=np.zeros_like(distortions), where=distortions > 0)
            passed = sigma > 0 and float(np.median(errors)) <= .05
            methods[method] = dict(status="MATCHED" if passed else "NOT_ASSESSABLE", clip=clip,
                      sigma=sigma, noise_sd=clip*sigma, median_relative_error=float(np.median(errors)),
                      defense_distortions=distortions.tolist(), unit_gaussian_norms=gaussian.tolist(),
                      matching_rule="P33B median per-record relative error<=.05", n=24)
        results[cell] = methods
    return results


def supervise():
    owner = socket.socket()
    owner.bind(("127.0.0.1", 43464))
    owner.listen(1)
    doc = verify()
    write(OUT / "development_supervisor.lock.json", dict(pid=os.getpid(), started_at=now(), port=43464))
    rows, failures = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute, config): config for config in doc["jobs"]}
        for future in as_completed(futures):
            config = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                failures.append(dict(config=config, error=traceback.format_exc()))
            done = len(rows)+len(failures)
            progress = dict(part="development24", dataset=config["dataset"], defense="v1_and_v2",
                        comparator="client_distortion_DP_calibration", done=done, total=len(doc["jobs"]), failed=len(failures),
                        last_update=now(), eta_minutes=(time.time()-started)/done*(len(doc["jobs"])-done)/60)
            write(OUT / "progress.json", progress)
            append(OUT / "progress.log", progress)
            append(OUT / "runs.jsonl", dict(at=now(), event="development_finished", config=config))
    if failures:
        write(OUT / "REQUIRES_DIRECTION_DEVELOPMENT.json", dict(at=now(), failures=failures, pool_drained=True))
        return
    write(OUT / "distortion_calibration.json", calibration(rows))
    write(OUT / "DEVELOPMENT_COMPLETE.json", dict(at=now(), stage_only=True, jobs=len(rows), freeze_sha256=sha(FREEZE)))


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
        if own_processes() or qualification_processes():
            raise RuntimeError("P34C workload alive; no duplicate/overlap")
        if (OUT / "REQUIRES_DIRECTION_DEVELOPMENT.json").exists():
            raise RuntimeError("scientific failure requires direction")
        if list(OUT.glob("development_launch_*.json")) and not args.resume:
            raise RuntimeError("prior launch: verified infrastructure interruption and explicit resume required")
        verify()
        with (OUT / "development_stdout.log").open("a") as stdout, (OUT / "development_stderr.log").open("a") as stderr:
            process = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--supervise"], cwd=ROOT,
                     stdout=stdout, stderr=stderr, start_new_session=os.name != "nt")
        write(OUT / ("development_launch_" + str(time.time_ns()) + ".json"), dict(pid=process.pid, at=now(), freeze_sha256=sha(FREEZE)))
    elif args.supervise:
        supervise()
    elif args.job:
        job(json.loads(args.job))
    else:
        parser.error("choose one mode")


if __name__ == "__main__":
    main()
