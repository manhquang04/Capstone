"""Run the single frozen paired RQ2 confirmatory batch."""

from __future__ import annotations

import argparse, hashlib, json, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [1984441809,653660776,650446433,1419599061,1306102865,1754157487,1247758568,1723661792,1703506450,848721957,1402156061,1726548684,1764332517,1278184566,420568183,1819638191,232043868,1937725506,1347707636,98089969,1723546374]
METHODS = ["baseline", "dna_lossless", "dna_transform", "dp_0.00025"]
CONTRACT = ROOT / "artifacts/rq2/confirmatory_freeze_20260913/per_seed_contract.json"
CONTRACT_SHA256 = "2015b6062245661109bb84c0c3c24ba589e1990747c9ff93582902337eed643c"

def sha256(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def script_for(method):
    return {"baseline": ("experiments/run_fraud_fl_baseline.py","BASELINE_OUTPUT_PATH"),
            "dna_lossless": ("experiments/run_fraud_fl_dna.py","DNA_OUTPUT_PATH"),
            "dna_transform": ("experiments/run_fraud_fl_dna_transform.py","DNA_TRANSFORM_OUTPUT_PATH")}.get(method,
            ("experiments/run_fraud_fl_dp.py","DP_OUTPUT_PATH"))

def run_job(job, python, output):
    seed, method = int(job["seed"]), job["method"]
    folder = output / f"seed_{seed}" / method; folder.mkdir(parents=True, exist_ok=False)
    result = folder / "metrics.json"; script, outvar = script_for(method)
    env = os.environ.copy(); env.update({"PYTHONPATH":str(ROOT),"FL_RUN_SEED":str(seed),"MAX_ROWS":"500000",
        "NUM_ROUNDS":"50","LOCAL_EPOCHS":"1","FL_NUM_CLIENTS":"3","LOSS_TYPE":"focal","FOCAL_ALPHA":"0.95",
        "FOCAL_GAMMA":"2.0","OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1",
        "VECLIB_MAXIMUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1",outvar:str(result)})
    if method == "dna_transform": env.update({"DNA_TRANSFORM_BLOCK_SIZE":"256","DNA_TRANSFORM_MIX":"0.08","DNA_TRANSFORM_KEEP":"0.88","DNA_TRANSFORM_SHRINK":"0.45"})
    if method.startswith("dp_"): env.update({"DP_CLIP_NORM":"100.0","DP_NOISE_MULTIPLIER":"0.00025"})
    cmd=[python,script]; started=time.perf_counter()
    with (folder/"stdout.log").open("w") as so, (folder/"stderr.log").open("w") as se:
        process=subprocess.run(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
    row={**job,"command":cmd,"returncode":process.returncode,"seconds":time.perf_counter()-started,
         "status":"SUCCESS" if process.returncode==0 and result.exists() else "FAILED"}
    (folder/"job_record.json").write_text(json.dumps(row,indent=2)+"\n")
    return row

def main():
    p=argparse.ArgumentParser(); p.add_argument("--workers",type=int,default=8); p.add_argument("--python",default=str(ROOT/".venv-phase1/bin/python")); p.add_argument("--output-dir",type=Path,default=ROOT/"artifacts/rq2/confirmatory_run_20260913"); a=p.parse_args()
    if not 1 <= a.workers <= 9: raise ValueError("workers must be in [1, 9]")
    if "confirmatory_execution_authorized: true" not in (ROOT/"protocols/config/rq2_confirmatory.yaml").read_text(): raise RuntimeError("RQ2 execution not authorized")
    if sha256(CONTRACT) != CONTRACT_SHA256: raise RuntimeError("per-seed contract checksum mismatch")
    contract=json.loads(CONTRACT.read_text()); contract_seeds=[r["seed"] for r in contract["records"]]
    if contract_seeds != SEEDS: raise RuntimeError("frozen seed order/contract mismatch")
    output=a.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    jobs=[]
    for wave in range(len(METHODS)):
        for i,seed in enumerate(SEEDS): jobs.append({"seed":seed,"method":METHODS[(i+wave)%len(METHODS)],"wave":wave})
    manifest={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"scope":"single frozen RQ2 confirmatory execution","seeds":SEEDS,"methods":METHODS,"workers":a.workers,"jobs":jobs,"contract_sha256":CONTRACT_SHA256}
    (output/"execution_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        fs=[ex.submit(run_job,j,a.python,output) for j in jobs]
        for f in as_completed(fs):
            row=f.result(); rows.append(row); print(json.dumps({k:row[k] for k in ("seed","method","status","seconds")}),flush=True)
    summary={"created_at":datetime.now(timezone.utc).isoformat(),"jobs":len(rows),"success":sum(r["status"]=="SUCCESS" for r in rows),"failed":sum(r["status"]=="FAILED" for r in rows),"records":sorted(rows,key=lambda r:(r["seed"],r["method"]))}
    (output/"execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({k:summary[k] for k in ("jobs","success","failed")},indent=2)); raise SystemExit(0 if summary["failed"]==0 else 1)

if __name__ == "__main__": main()
