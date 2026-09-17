"""Execute the frozen development attack gates for strong update-DP points."""
from __future__ import annotations
import argparse, json, os, shutil, subprocess, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"
MULTIPLIERS={"epsilon_100":0.09866372299196495,"epsilon_50":0.15890264833373352,"epsilon_10":0.5678974412358136,"epsilon_1":4.900556183310786}

def job_run(job, python, out, target):
    folder=out/job["branch"]/f"group_{job['group']}_run";folder.mkdir(parents=True,exist_ok=False)
    for name in ("baseline_gate.json","frozen.json"): shutil.copy2(SOURCE/name,folder/name)
    shutil.copy2(target,folder/target.name)
    if job["branch"]=="raw":
        cmd=[python,"experiments/run_phase4_harddiff_reparam_for_misselected.py",str(folder),"--target-file",target.name,"--groups",str(job["group"]),"--restarts","8","--init-mode","standard","--nonnegative-lambda","0.001"]
    else:
        cmd=[python,"experiments/run_phase4_simple_defense_attack.py",str(folder),"--target-file",target.name,"--defense","clipping_noise_mc","--groups",str(job["group"]),"--restarts","8","--iterations","600","--attack-lr","0.1","--init-mode","standard","--nonnegative-lambda","0.001","--clip-norm","100","--noise-multiplier",str(MULTIPLIERS[job["branch"]]),"--mc-noise-samples","100","--defense-seed","314159265"]
    env=os.environ.copy();env.update({"PYTHONPATH":str(ROOT),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    t=time.perf_counter()
    with (folder/"stdout.log").open("w") as so,(folder/"stderr.log").open("w") as se: p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
    r={**job,"command":cmd,"returncode":p.returncode,"seconds":time.perf_counter()-t,"status":"SUCCESS" if p.returncode==0 else "FAILED"};(folder/"job_record.json").write_text(json.dumps(r,indent=2)+"\n");return r

def main():
 p=argparse.ArgumentParser();p.add_argument("--target",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--workers",type=int,default=9);p.add_argument("--python",default=str(ROOT/".venv-phase1/bin/python"));a=p.parse_args()
 if not 1<=a.workers<=9: raise ValueError("workers must be in 1..9")
 target=a.target.resolve();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
 jobs=[{"branch":"raw","group":g} for g in range(8)]+[{"branch":b,"group":g} for b in MULTIPLIERS for g in range(8)]
 (out/"manifest.json").write_text(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"scope":"development-only strong update-DP attack gates", "amendment":"protocols/amendments/2026-09-16_strong_update_dp_exploratory_pareto.md", "target":str(target),"multipliers":MULTIPLIERS,"jobs":jobs},indent=2)+"\n")
 records=[]
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  fs=[ex.submit(job_run,j,a.python,out,target) for j in jobs]
  for f in as_completed(fs):
   x=f.result();records.append(x);print(json.dumps({k:x[k] for k in ("branch","group","status","seconds")}),flush=True)
 summary={"created_at":datetime.now(timezone.utc).isoformat(),"jobs":len(records),"success":sum(x["status"]=="SUCCESS" for x in records),"failed":sum(x["status"]=="FAILED" for x in records),"records":sorted(records,key=lambda x:(x["branch"],x["group"]))};(out/"execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n");
 if summary["failed"]:raise SystemExit(1)
if __name__=="__main__":main()
