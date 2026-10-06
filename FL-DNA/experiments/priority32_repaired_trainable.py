"""Separate authorized P32 utility variant; original runner/artifacts untouched."""
import argparse
import json
import os
import random
import shutil
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments import priority32_multidataset as p

ORIGINAL = p.OUT
OUT = ORIGINAL / "repair_trainable_raw_bn_20261002"
AMENDMENT = p.ROOT / "protocols/amendments/2026-10-02_priority32_bn_domain_repair_authorized.md"
TRAINABLE = frozenset(name for name, _ in p.FraudMLP(1).named_parameters())
V1, V2 = p.dna_transform_state, p.dna_transform_v2_state


def raw_buffers(transform, state, *args):
    defended, stats = transform(state, *args)
    for name, value in state.items():
        if name not in TRAINABLE:
            defended[name] = value.detach().clone()
    return defended, stats


def v1(state, *args):
    return raw_buffers(V1, state, *args)


def v2(state, *args):
    return raw_buffers(V2, state, *args)


def worker(config, path):
    p.dna_transform_state, p.dna_transform_v2_state = v1, v2
    try:
        return p.train_job(config, path)
    finally:
        p.dna_transform_state, p.dna_transform_v2_state = V1, V2


def record(event, **fields):
    doc = {"at": p.now(), "event": event, **fields}
    with (OUT / "runs.jsonl").open("a") as handle:
        handle.write(json.dumps(doc) + "\n")
    print(json.dumps(doc), flush=True)


def progress(done, failed, start, config=None, stage="4_repaired_RQ2"):
    config = config or {}
    doc = {"stage": stage, "dataset": config.get("dataset", "all"),
           "method": config.get("method", "all"), "done": done, "total": 126,
           "failed": failed, "started_at": start[1], "last_update": p.now(),
           "eta_minutes": (time.time()-start[0]) / done * (126-done) / 60 if done else None}
    p.write(OUT / "progress.json", doc)
    with (OUT / "progress.log").open("a") as handle:
        handle.write(json.dumps(doc) + "\n")
    # Mutable top-level telemetry only; original scientific outputs unchanged.
    p.write(ORIGINAL / "progress.json", doc)
    checklist = {str(i): {"title": title, "status": status, "last_update": p.now()}
                 for i, (title, status) in enumerate([
                     ("Data audit reused and validated", "completed"),
                     ("Frozen preprocessing reused", "completed"),
                     ("Original validation gates reused", "completed"),
                     ("Authorized repair amendment and execution freeze", "completed"),
                     ("126 repaired transform jobs plus 63 paired baselines",
                      "completed" if done == 126 and not failed else "in_progress"),
                     ("RQ2 analysis and independent recomputation", "pending"),
                     ("RQ3 Bundle B and descriptive DP cost", "pending"),
                     ("Report, hashes and final checks", "pending")])}
    p.write(OUT / "checklist.json", checklist)


def supervise():
    OUT.mkdir(exist_ok=True)
    lock = OUT / "supervisor.lock.json"
    if lock.exists():
        prior = json.loads(lock.read_text())
        try:
            os.kill(prior["pid"], 0)
        except ProcessLookupError:
            lock.rename(OUT / ("supervisor.lock.previous."+str(time.time_ns())+".json"))
        else:
            raise RuntimeError("live repaired supervisor; duplicate prohibited")
    p.write(lock, {"pid": os.getpid(), "at": p.now()})
    freeze = json.loads((ORIGINAL / "execution_freeze.json").read_text())
    baseline_hashes = {}
    jobs = []
    for dataset in p.DATASETS:
        p.prepare_dataset(dataset)  # validates hashes and skips; no regeneration
        audit_copy = OUT / "prepared" / dataset / "audit.json"
        audit_copy.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ORIGINAL / "prepared" / dataset / "audit.json", audit_copy)
        for seed in p.SEEDS:
            source = ORIGINAL / "rq2_jobs" / dataset / "baseline" / f"seed_{seed}.json"
            cfg = p.job_config(dataset, "baseline", seed, freeze["chosen"][dataset]["training"])
            assert p.valid_result(source, cfg)
            target = OUT / "rq2_jobs" / dataset / "baseline" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                assert p.sha(target) == p.sha(source)
            else:
                shutil.copyfile(source, target)
            baseline_hashes[str(source.relative_to(p.ROOT))] = p.sha(source)
            for method in p.METHODS[1:]:
                cfg = p.job_config(dataset, method, seed, freeze["chosen"][dataset]["training"])
                cfg.update({"protocol_variant": "trainable_only_transform_raw_bn",
                            "repair_amendment_sha256": p.sha(AMENDMENT),
                            "repair_code_sha256": p.sha(__file__)})
                jobs.append((cfg, OUT / "rq2_jobs" / dataset / method / f"seed_{seed}.json"))
    plan = {"status": "FROZEN/AUTHORIZED", "amendment_sha256": p.sha(AMENDMENT),
            "chosen": freeze["chosen"],
            "code_sha256": p.sha(__file__), "baseline_hashes": baseline_hashes,
            "jobs": [{"config": cfg, "path": str(path), "config_sha256": p.canonical_sha(cfg)} for cfg,path in jobs],
            "original_execution_freeze_sha256": p.sha(ORIGINAL / "execution_freeze.json")}
    manifest = OUT / "execution_freeze.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == plan
    else:
        p.write(manifest, plan)
    shutil.copyfile(ORIGINAL / "data_audit.json", OUT / "data_audit.json")
    random.Random(320032).shuffle(jobs)
    start = (time.time(), p.now())
    done, failed = 0, []
    record("supervisor_started_or_resumed", command=sys.argv, pid=os.getpid())
    progress(done, 0, start)
    with ProcessPoolExecutor(max_workers=4, mp_context=__import__("multiprocessing").get_context("spawn")) as pool:
        futures = {}
        for cfg, path in jobs:
            if p.valid_result(path, cfg):
                done += 1
                record("validated_skip", path=str(path))
                progress(done, len(failed), start, cfg)
            else:
                record("job_launched", path=str(path), config_sha256=p.canonical_sha(cfg))
                futures[pool.submit(worker, cfg, str(path))] = (cfg, path)
        for future in as_completed(futures):
            cfg, path = futures[future]
            try:
                future.result()
                assert p.valid_result(path, cfg)
                done += 1
                record("job_completed", path=str(path), sha256=p.sha(path))
            except Exception:
                failed.append(str(path))
                record("job_failed", path=str(path), traceback=traceback.format_exc())
            progress(done, len(failed), start, cfg)
    if failed:
        p.write(OUT / "REQUIRES_DIRECTION.json", {"failed_paths": failed, "completed": done})
        record("stopped_after_drain", failed_paths=failed)
        return
    assert done == 126
    progress(done, 0, start, stage="RQ2_REPAIRED_JOBS_COMPLETE_PENDING_ANALYSIS_COST")
    record("repaired_training_completed", elapsed_seconds=time.time()-start[0])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--supervise", action="store_true", required=True)
    parser.parse_args()
    supervise()
