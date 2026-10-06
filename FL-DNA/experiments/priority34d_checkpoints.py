"""Only P34D Part3 baseline checkpoint training; no recovery target access."""
import argparse
import json
import os
import socket
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority33a_bn_mean as b
from experiments import priority33a_audit as audit
torch = b.torch
torch.set_num_threads(1)
if torch.get_num_interop_threads() != 1:
    torch.set_num_interop_threads(1)
assert torch.get_num_interop_threads() == torch.get_num_threads() == 1
OUT = ROOT / "artifacts/priority34d"
AMENDMENT = ROOT / "protocols/amendments/2026-10-04_priority34d_checkpoint_robustness.md"
SEEDS = (342000, 342001, 342002)
FREEZE = OUT / "checkpoint_execution_freeze.json"


def write(path, value):
    path = Path(path)
    if not path.resolve().is_relative_to(OUT.resolve()):
        raise ValueError("P34D write firewall")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")
    temp.replace(path)


def append(path, value):
    if not Path(path).resolve().is_relative_to(OUT.resolve()):
        raise ValueError("P34D append firewall")
    with Path(path).open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False)+"\n")


def folder(seed):
    if seed not in SEEDS:
        raise ValueError("seed not registered")
    return OUT / "A2/baf/training" / ("baseline_"+str(seed))


def live():
    rows = subprocess.check_output(["ps", "-axo", "pid,command"], text=True).splitlines()
    return [r for r in rows if "priority34d_checkpoints.py" in r
            and ("--supervise" in r or "--job" in r)
            and int(r.split()[0]) != os.getpid()]


def verify():
    doc = json.loads(FREEZE.read_text())
    for name, digest in {**doc["sources"], **doc["inputs"]}.items():
        if audit.sha(ROOT / name) != digest:
            raise ValueError("checkpoint freeze mismatch: "+name)
    if doc["seeds"] != list(SEEDS):
        raise ValueError("changed seed schedule")
    return doc


def prepare():
    assert not live()
    assert not FREEZE.exists(), "existing freeze: use verify, never overwrite"
    paths = {Path(__file__).resolve(), AMENDMENT, ROOT / "tests/test_priority34d_checkpoints.py",
             ROOT / "protocols/amendments/2026-10-04_priority34d_direction_and_preflight.md"}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path:
            path = Path(path).resolve()
            if path.is_relative_to(ROOT) and ".venv" not in str(path) and path.suffix == ".py":
                paths.add(path)
    sources = {str(p.relative_to(ROOT)): audit.sha(p) for p in sorted(paths)}
    inputs = list((b.p.OUT / "prepared/baf").glob("*"))
    inputs += [b.p.OUT / "execution_freeze.json", ROOT / "artifacts/priority34c/COMPLETE.json",
               ROOT / "reports/priority34c_report.md"]
    write(FREEZE, dict(status="FROZEN", at=audit.now(), seeds=list(SEEDS),
                      sources=sources, inputs={str(p.relative_to(ROOT)):audit.sha(p)
                                               for p in inputs if p.is_file()}))
    print(json.dumps({"freeze_sha256":audit.sha(FREEZE), "sources":len(sources),
                      "inputs":len(json.loads(FREEZE.read_text())["inputs"])}))


def valid(seed):
    path = folder(seed) / "checkpoint_validated.json"
    if not path.exists():
        return False
    doc = json.loads(path.read_text())
    if doc["seed"] != seed or doc["freeze_sha256"] != audit.sha(FREEZE) or doc["status"] != "PASS":
        raise ValueError("invalid existing checkpoint receipt")
    for name, digest in doc["outputs"].items():
        if audit.sha(folder(seed) / name) != digest:
            raise ValueError("checkpoint output hash mismatch")
    return True


def job(seed):
    verify()
    if valid(seed):
        return
    if (folder(seed)/"result.json").exists() or (folder(seed)/"bn_rounds.jsonl").exists():
        raise RuntimeError("preserved partial attempt; direction required before retry")
    # Reuse P33A's strict actual-forward, local/upload/aggregate/evaluation guards.
    # Namespaces only change; P32 prepared data and original source files do not.
    b.OUT, b.AMENDMENT = OUT, AMENDMENT
    audit.OUT, audit.AMENDMENT = OUT, AMENDMENT
    b.write, b.append = write, append
    original_config = b.p.job_config
    def final_config(*args, **kwargs):
        kwargs["quality"] = False  # compute descriptive test endpoint; never select
        return original_config(*args, **kwargs)
    b.p.job_config = final_config
    b.training_job(dict(dataset="baf", method="baseline", seed=seed, id="baseline_"+str(seed)))
    result = json.loads((folder(seed)/"result.json").read_text())
    if result["status"] != "COMPLETED":
        raise RuntimeError("checkpoint strict training gate failed; preserve result")
    state = torch.load(folder(seed)/"final_state.pt", map_location="cpu", weights_only=False)
    b.strict_state(state, "P34D_final_checkpoint")
    rounds = [json.loads(line) for line in (folder(seed)/"bn_rounds.jsonl").read_text().splitlines()]
    assert len(rounds) == 50 and all(min(r["bn_min"].values()) >= 0 for r in rounds)
    outputs = {p.name:audit.sha(p) for p in folder(seed).iterdir()
               if p.is_file() and p.name not in ("stdout.log", "stderr.log")}
    write(folder(seed)/"checkpoint_validated.json", dict(status="PASS", seed=seed,
          at=audit.now(), freeze_sha256=audit.sha(FREEZE), outputs=outputs,
          rounds=50, device="cpu", intra_threads=torch.get_num_threads(),
          inter_threads=torch.get_num_interop_threads()))


def execute(seed):
    if valid(seed):
        return dict(seed=seed, status="PASS", skipped=True)
    folder(seed).mkdir(parents=True, exist_ok=True)
    append(OUT/"runs.jsonl", dict(at=audit.now(), event="checkpoint_started", seed=seed))
    with (folder(seed)/"stdout.log").open("a") as so, (folder(seed)/"stderr.log").open("a") as se:
        proc = subprocess.run([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                               "--job", str(seed)], cwd=ROOT, stdout=so, stderr=se)
    status = "PASS" if proc.returncode == 0 and valid(seed) else "FAILED"
    append(OUT/"runs.jsonl", dict(at=audit.now(), event="checkpoint_finished", seed=seed,
                                 status=status, returncode=proc.returncode))
    return dict(seed=seed, status=status, returncode=proc.returncode)


def supervise():
    guard = socket.socket()
    guard.bind(("127.0.0.1", 43466)); guard.listen(1)
    if live():
        raise RuntimeError("checkpoint supervisor/workers active")
    verify()
    write(OUT/"checkpoint_supervisor.lock.json", dict(pid=os.getpid(), at=audit.now()))
    rows = []
    checklist = json.loads((OUT/"checklist.json").read_text())
    checklist["part3_checkpoint_preflight_freeze"]["status"] = "completed"
    checklist["part3_checkpoints"]["status"] = "in_progress"
    write(OUT/"checklist.json", checklist)
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(execute, seed):seed for seed in SEEDS}
        for future in as_completed(futures):
            try:
                row = future.result()
            except Exception as error:
                row = dict(seed=futures[future], status="FAILED", error=repr(error))
            rows.append(row)
            progress = dict(part="part3_checkpoints", dataset="baf", defense="unprotected",
                comparator="checkpoint_training", done=len(rows), total=3,
                failed=sum(r["status"] != "PASS" for r in rows), last_update=audit.now(), eta_minutes=None)
            write(OUT/"progress.json", progress); append(OUT/"progress.log", progress)
    failed = [r for r in rows if r["status"] != "PASS"]
    if failed:
        write(OUT/"REQUIRES_DIRECTION_CHECKPOINTS.json", dict(failed=failed, pool_drained=True))
        checklist["part3_checkpoints"]["status"] = "requires_direction"
    else:
        write(OUT/"CHECKPOINTS_COMPLETE.json", dict(stage_only=True, jobs=3, at=audit.now(),
                                                   freeze_sha256=audit.sha(FREEZE)))
        checklist["part3_checkpoints"]["status"] = "completed"
    write(OUT/"checklist.json", checklist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--supervise", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--job", type=int)
    args = parser.parse_args()
    if args.prepare:
        prepare()
    elif args.job is not None:
        job(args.job)
    elif args.supervise:
        supervise()
    elif args.launch:
        assert not live(), "active workers; no duplicate"
        assert not (OUT/"REQUIRES_DIRECTION_CHECKPOINTS.json").exists()
        assert not (OUT/"CHECKPOINTS_COMPLETE.json").exists()
        if list(OUT.glob("checkpoint_launch_*.json")) and not args.resume:
            raise RuntimeError("previous launch: inspect then explicit infrastructure resume only")
        verify()
        with (OUT/"checkpoint_stdout.log").open("a") as so, (OUT/"checkpoint_stderr.log").open("a") as se:
            proc = subprocess.Popen([sys.executable, "-B", "-u", str(Path(__file__).resolve()),
                                     "--supervise"], cwd=ROOT, stdout=so, stderr=se, start_new_session=True)
        receipt = dict(pid=proc.pid, at=audit.now(), freeze_sha256=audit.sha(FREEZE), resume=args.resume)
        write(OUT/("checkpoint_launch_"+str(proc.pid)+".json"), receipt)
        print(json.dumps(receipt))
    else:
        parser.error("choose prepare/launch/supervise/job")
