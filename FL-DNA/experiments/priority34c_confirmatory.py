"""P34C source-frozen individual-gradient confirmatory measurements."""
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
from experiments.priority34c_core import OUT, PREPARED, native, np, torch, sha, write
from experiments.priority34c_core import PreparedDataset, ratio_model, gradient, defend, ratio_decode
from experiments.priority34c_qualify import now, append, live_processes, verify_freeze
from experiments import priority34c_development as development
from experiments.priority33b_comparators import debias_blocks
from privacy.seed_manager import derive_seed

FREEZE = OUT / "confirmatory_freeze.json"
PRIVATE = OUT / "confirmatory_private_noise.json"
LOCAL_SD = .08031273039270019
DNA = ("dna_v1_conservative", "dna_v2_0p95")


def job_key(config):
    return "/".join(str(config[k]) for k in ("dataset", "cell", "arm", "index"))


def configs():
    gates = json.loads((OUT / "gates_n24.json").read_text())
    calibration = json.loads((OUT / "distortion_calibration.json").read_text())
    result = []
    for cell, gate in sorted(gates.items()):
        if not gate["passed"]:
            continue
        arms = ["unprotected", *DNA, "dp_local_eps10"]
        arms += ["dp_distortion_"+method for method in DNA if calibration[cell][method]["status"] == "MATCHED"]
        dataset, instrument = cell.split("/")
        result += [dict(dataset=dataset, cell=instrument, stage="n39", arm=arm, index=i)
                   for i in range(39) for arm in arms]
    return result


def own_processes():
    if os.name == "nt":
        rows = json.loads(subprocess.check_output(["powershell", "-NoProfile", "-Command",
              "Get-CimInstance Win32_Process | Select-Object ProcessId,CommandLine | ConvertTo-Json"], text=True))
        return [row for row in rows if row["ProcessId"] != os.getpid() and "priority34c_confirmatory.py" in (row.get("CommandLine") or "")
                and any(flag in row["CommandLine"] for flag in ("--supervise", "--job"))]
    rows = subprocess.check_output(["ps", "-axo", "pid,command"], text=True).splitlines()
    return [row for row in rows if "priority34c_confirmatory.py" in row and "python" in row.lower()
            and int(row.strip().split(None, 1)[0]) != os.getpid() and any(flag in row for flag in ("--supervise", "--job"))]


def verify():
    development.verify()
    doc = json.loads(FREEZE.read_text())
    for section in ("sources", "inputs"):
        for name, digest in doc[section].items():
            if sha(ROOT / name) != digest:
                raise ValueError("confirmatory frozen drift: " + name)
    return doc


def prepare():
    if live_processes() or development.own_processes() or own_processes():
        raise RuntimeError("P34C workload active; refuse overlapping stage")
    for stage in ("QUALIFICATION", "DEVELOPMENT", "CONFIRMATORY"):
        if (OUT / ("REQUIRES_DIRECTION_"+stage+".json")).exists():
            raise RuntimeError("scientific failure unresolved: " + stage)
    if not (OUT / "DEVELOPMENT_COMPLETE.json").exists():
        raise RuntimeError("development incomplete")
    development.verify()
    if FREEZE.exists():
        return verify()
    jobs = configs()
    if not PRIVATE.exists():
        write(PRIVATE, {job_key(c): secrets.randbits(63) for c in jobs if c["arm"].startswith("dp_")})
        PRIVATE.chmod(0o600)
    inputs = {str((OUT / name).relative_to(ROOT)): sha(OUT / name) for name in
              ("development_freeze.json", "DEVELOPMENT_COMPLETE.json", "distortion_calibration.json", PRIVATE.name)}
    for c in development.configs():
        path = development.folder_for(c) / "result.json"
        if not development.valid(path, c):
            raise ValueError("missing development receipt")
        inputs[str(path.relative_to(ROOT))] = sha(path)
    paths = [Path(__file__).resolve(), ROOT / "experiments/priority34c_statistics.py",
             ROOT / "experiments/analyze_priority34c.py",
             ROOT / "experiments/priority33b_comparators.py",
             ROOT / "protocols/amendments/2026-10-04_priority34c_confirmatory_execution.md",
             ROOT / "protocols/amendments/2026-10-04_priority34c_report_audit.md",
             ROOT / "protocols/amendments/2026-10-04_priority34c_accounting_audit_order.md",
             ROOT / "protocols/amendments/2026-10-04_priority34c_confirmatory_dependency_seal.md"]
    paths += [ROOT / ("tests/test_priority34c_"+name+".py") for name in ("confirmatory", "statistics", "analysis")]
    for module in list(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix == ".py" and path.is_relative_to(ROOT) and not path.relative_to(ROOT).parts[0].startswith(".venv"):
            paths.append(path)
    write(FREEZE, dict(at=now(), sources={str(p.relative_to(ROOT)): sha(p) for p in paths},
          inputs=inputs, jobs=jobs, workers=4, device="cpu", threads=1,
          local_clip=.01, local_noise_sd=LOCAL_SD, fixed_family=72))
    return verify()


def folder_for(c):
    return OUT / c["dataset"] / c["cell"] / "n39" / ("target_%03d" % c["index"]) / c["arm"]


def valid(path, c):
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc["status"] != "COMPLETED" or doc["config"] != c or doc["freeze_sha256"] != sha(FREEZE):
        return False
    for name, digest in doc["files"].items():
        if sha(path.parent / doc["attempt"] / name) != digest:
            raise ValueError("confirmatory artifact drift")
    return True


def ratio_objective(net, record, labels, names, observed):
    candidate, _ = gradient(net, record, labels, ratio=True)
    if list(candidate) != names:
        raise ValueError("gradient domain mismatch")
    a = torch.cat([g.double().reshape(-1) for g in candidate.values()])
    b = torch.cat([g.double().reshape(-1) for g in observed])
    denominator = torch.linalg.vector_norm(a)*torch.linalg.vector_norm(b)
    if denominator <= 0:
        raise ValueError("undefined observable cosine objective")
    value = float(1.-torch.dot(a, b)/denominator)
    native.finite(value, "ratio observable objective")
    return value


def local_payload(raw, seed):
    """P34B local global-clip query, including its serialization safety factor."""
    norm = float(torch.sqrt(sum(g.double().square().sum() for g in raw.values())))
    factor = min(1., .01/norm) if norm > 0 else 1.
    generator = torch.Generator().manual_seed(seed)
    result = []
    for g in raw.values():
        clipped = (g.double()*factor*(1-4*np.finfo(np.float32).eps)).to(g)
        noisy = (clipped.double()+torch.randn(g.shape, dtype=torch.float64, generator=generator)*LOCAL_SD).to(g)
        native.finite(noisy, "local DP individual upload")
        result.append(noisy)
    return result


def recover(net, data, names, payload, labels, seed, arm, plans, metadata, ratio):
    variants = [("plain", payload)]
    if arm == "dna_v1_conservative":
        variants.append(("structure_debias", debias_blocks(payload)))
    candidates = []
    for name, observed in variants:
        if ratio:
            prepared, solver = ratio_decode(names, observed, metadata)
            # Sketch LS has its own observable residual; no raw gradient is used.
            objective = (sum(row["residual"] for row in solver["least_squares"])
                         if metadata is not None else ratio_objective(net, prepared, labels, names, observed))
            rec, ensemble, losses = (prepared-data.mean)/data.std, [], []
        else:
            rec, ensemble, losses = native.native_recover(net, data, observed, labels, seed,
                  mode="v2_sketch" if plans is not None else "plain", plans=plans)
            objective, solver = float(min(losses)), None
        native.finite(rec, "recovered record")
        native.finite(objective, "selection objective")
        candidates.append(dict(name=name, objective=objective, reconstruction=rec,
                               ensemble=ensemble, objectives=losses, solver=solver))
    selected = min(range(len(candidates)), key=lambda i: candidates[i]["objective"])
    return candidates, selected


def job(c):
    doc = verify()
    if c not in doc["jobs"] or c["stage"] != "n39":
        raise ValueError("unauthorized confirmatory target access")
    destination = folder_for(c) / ("attempt_"+str(time.time_ns()))
    destination.mkdir(parents=True)
    started = time.time()
    try:
        data = PreparedDataset(c["dataset"], c["cell"])
        manifest = json.loads((OUT / c["dataset"] / "targets.json").read_text())
        ids = manifest["source_ids"][c["cell"]]["n39"][c["index"]]
        source = np.load(PREPARED / c["dataset"] / "train_source_ids.npy")
        mapping = {int(value): i for i, value in enumerate(source)}
        truth, labels = data.Xtrain[[mapping[v] for v in ids]].clone(), data.ytrain[[mapping[v] for v in ids]].clone()
        ratio = c["cell"] == "ratio_batch1"
        net = ratio_model(data.num_features) if ratio else native.make_model(data.num_features)
        raw, bn = gradient(net, data.de_standardize(truth) if ratio else truth, labels, ratio=ratio)
        kwargs, dp_contract = {}, None
        if c["arm"].startswith("dp_"):
            if c["arm"] == "dp_local_eps10":
                clip, sd = .01, LOCAL_SD
                dp_contract = dict(unit="per-client update-level not record DP", rounds=50, epsilon=10, delta=1e-5)
            else:
                method = c["arm"].removeprefix("dp_distortion_")
                calibration = json.loads((OUT / "distortion_calibration.json").read_text())[c["dataset"]+"/"+c["cell"]][method]
                if calibration["status"] != "MATCHED":
                    raise ValueError("distortion gate failed")
                clip, sd = calibration["clip"], calibration["noise_sd"]
            kwargs = dict(noise_seed=json.loads(PRIVATE.read_text())[job_key(c)], clip=clip, noise_sd=sd)
            dp_contract = dict(dp_contract or {}, clip=clip, noise_sd=sd, client_side_before_upload=True)
        if c["arm"] == "dp_local_eps10":
            payload, plans, metadata = local_payload(raw, kwargs["noise_seed"]), None, None
        else:
            payload, plans, metadata = defend(raw, c["arm"], **kwargs)
        # Receipt contains transmitted information only. No raw/noise seed is serialized.
        public_bn = {key: value for key, value in bn.items() if key != "loss"}
        torch.save(dict(payload=payload, payload_names=list(raw), labels=labels, public_model=net.state_dict(),
                        plans=plans, metadata=metadata, bn=public_bn, dp=dp_contract), destination / "server_receipt.pt")
        seed = derive_seed(340400, "p34c-attack", c["dataset"], c["cell"], "n39", c["index"])
        candidates, selected = recover(net, data, list(raw), payload, labels, seed, c["arm"], plans, metadata, ratio)
        score = native.measure(data, truth, candidates[selected]["reconstruction"])
        torch.save(dict(truth=truth, labels=labels, source_ids=ids), destination / "scoring_truth.pt")
        torch.save(dict(candidates=candidates, selected=selected), destination / "reconstruction.pt")
        write(destination.parent / "result.json", dict(status="COMPLETED", config=c,
              freeze_sha256=sha(FREEZE), attempt=destination.name, accuracy=score, source_ids=ids,
              selected_variant=candidates[selected]["name"], attack_seed=seed, bn=bn, dp=dp_contract,
              device="cpu", intra_threads=torch.get_num_threads(), inter_threads=torch.get_num_interop_threads(),
              at=now(), elapsed_seconds=time.time()-started, files={p.name: sha(p) for p in destination.glob("*.pt")}))
    except Exception:
        write(destination / "failure.json", dict(config=c, at=now(), error=traceback.format_exc()))
        raise


def execute(c):
    folder = folder_for(c)
    if valid(folder / "result.json", c):
        return json.loads((folder / "result.json").read_text())
    folder.mkdir(parents=True, exist_ok=True)
    append(OUT / "runs.jsonl", dict(at=now(), event="confirmatory_started", config=c))
    with (folder / "stdout.log").open("a") as stdout, (folder / "stderr.log").open("a") as stderr:
        p = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--job", json.dumps(c)],
                           cwd=ROOT, stdout=stdout, stderr=stderr)
    if p.returncode or not valid(folder / "result.json", c):
        raise RuntimeError("confirmatory job failed: "+json.dumps(c))
    return json.loads((folder / "result.json").read_text())


def supervise():
    owner = socket.socket()
    owner.bind(("127.0.0.1", 43465))
    owner.listen(1)
    doc = verify()
    write(OUT / "confirmatory_supervisor.lock.json", dict(pid=os.getpid(), at=now(), port=43465))
    rows, failures, started = [], [], time.time()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(execute, c): c for c in doc["jobs"]}
        for future in as_completed(futures):
            c = futures[future]
            try:
                rows.append(future.result())
            except Exception:
                failures.append(dict(config=c, error=traceback.format_exc()))
            done = len(rows)+len(failures)
            progress = dict(part="confirmatory_n39", dataset=c["dataset"], defense=c["arm"], comparator="individual_gradient",
                  done=done, total=len(doc["jobs"]), failed=len(failures), last_update=now(),
                  eta_minutes=(time.time()-started)/done*(len(doc["jobs"])-done)/60)
            write(OUT / "progress.json", progress)
            append(OUT / "progress.log", progress)
            append(OUT / "runs.jsonl", dict(at=now(), event="confirmatory_finished", config=c))
    if failures:
        write(OUT / "REQUIRES_DIRECTION_CONFIRMATORY.json", dict(at=now(), pool_drained=True, failures=failures))
    else:
        write(OUT / "CONFIRMATORY_COMPLETE.json", dict(at=now(), stage_only=True, jobs=len(rows), freeze_sha256=sha(FREEZE)))


def main():
    p = argparse.ArgumentParser()
    for flag in ("prepare", "launch", "resume", "supervise"):
        p.add_argument("--"+flag, action="store_true")
    p.add_argument("--job")
    args = p.parse_args()
    if args.prepare:
        prepare()
    elif args.launch:
        if own_processes() or live_processes() or development.own_processes():
            raise RuntimeError("P34C workload active")
        if (OUT / "REQUIRES_DIRECTION_CONFIRMATORY.json").exists():
            raise RuntimeError("scientific failure requires direction")
        if list(OUT.glob("confirmatory_launch_*.json")) and not args.resume:
            raise RuntimeError("prior launch requires verified infrastructure interruption/resume")
        verify()
        with (OUT / "confirmatory_stdout.log").open("a") as stdout, (OUT / "confirmatory_stderr.log").open("a") as stderr:
            proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()), "--supervise"],
                  cwd=ROOT, stdout=stdout, stderr=stderr, start_new_session=True)
        write(OUT / ("confirmatory_launch_"+str(time.time_ns())+".json"), dict(pid=proc.pid, at=now(), freeze_sha256=sha(FREEZE)))
    elif args.supervise:
        supervise()
    elif args.job:
        job(json.loads(args.job))
    else:
        p.error("choose mode")


if __name__ == "__main__":
    main()
