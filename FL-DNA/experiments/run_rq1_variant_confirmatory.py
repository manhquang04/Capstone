"""Execute one frozen RQ1 DNA-variant confirmatory family."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_RUN = ROOT / "artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _command(branch: str, run: Path, group: int, python: str, config: dict, target: Path) -> list[str]:
    attack=config["attack"]
    if branch == "raw":
        return [python,"experiments/run_phase4_harddiff_reparam_for_misselected.py",str(run),"--target-file",target.name,"--groups",str(group),"--restarts",str(attack["restarts"]),"--init-mode","standard","--nonnegative-lambda",str(attack["nonnegative_lambda"])]
    if branch == "dna":
        dna=config["dna_transform"]
        return [python,"experiments/run_phase4_dna_level1_forward_attack.py",str(run),"--target-file",target.name,"--groups",str(group),"--candidates",str(attack["dna_candidates"]),"--restarts",str(attack["restarts"]),"--iterations",str(attack["iterations"]),"--attack-lr",str(attack["learning_rate"]),"--init-mode","standard","--nonnegative-lambda",str(attack["nonnegative_lambda"]),"--block-size",str(dna["block_size"]),"--mix-ratio",str(dna["mix_ratio"]),"--keep-ratio",str(dna["keep_ratio"]),"--shrink-factor",str(dna["shrink_factor"]),"--dna-run-seed",str(dna["dna_run_seed"])]
    if branch == "dp":
        dp=config["dp_distortion_matched"]
        return [python,"experiments/run_phase4_simple_defense_attack.py",str(run),"--target-file",target.name,"--defense","clipping_noise_mc","--groups",str(group),"--restarts",str(attack["restarts"]),"--iterations",str(attack["iterations"]),"--attack-lr",str(attack["learning_rate"]),"--init-mode","standard","--nonnegative-lambda",str(attack["nonnegative_lambda"]),"--clip-norm",str(dp["clip_norm"]),"--noise-multiplier",str(dp["noise_multiplier"]),"--mc-noise-samples",str(dp["mc_noise_samples"]),"--defense-seed",str(dp["defense_seed"])]
    raise ValueError(branch)


def _job(job: dict, python: str, output: Path, config: dict, target: Path) -> dict:
    branch=job["branch"]; group=job["group"]; run=output/branch/f"group_{group}_run"
    run.mkdir(parents=True,exist_ok=False)
    shutil.copy2(SOURCE_RUN/"baseline_gate.json",run/"baseline_gate.json"); shutil.copy2(SOURCE_RUN/"frozen.json",run/"frozen.json"); shutil.copy2(target,run/target.name)
    command=_command(branch,run,group,python,config,target)
    env=os.environ.copy(); env.update({"PYTHONPATH":str(ROOT),"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1","VECLIB_MAXIMUM_THREADS":"1","NUMEXPR_NUM_THREADS":"1"})
    started=time.perf_counter()
    with (run/"stdout.log").open("w") as stdout,(run/"stderr.log").open("w") as stderr:
        process=subprocess.run(command,cwd=ROOT,env=env,stdout=stdout,stderr=stderr,check=False)
    row={**job,"command":command,"returncode":process.returncode,"seconds":time.perf_counter()-started,"status":"SUCCESS" if process.returncode==0 else "FAILED"}
    (run/"job_record.json").write_text(json.dumps(row,indent=2)+"\n"); return row


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); parser.add_argument("--workers",type=int,default=9); parser.add_argument("--python",default=str(ROOT/".venv-phase1/bin/python")); args=parser.parse_args()
    if not 1<=args.workers<=9: raise ValueError("workers must be in [1,9]")
    config=json.loads(args.config.read_text()); target=(ROOT/config["target"]["path"]).resolve(); overlap=(ROOT/config["target"]["overlap_matrix"]).resolve()
    if config.get("confirmatory_attack_authorized") is not True: raise RuntimeError("variant attack not authorized")
    if _sha(target)!=config["target"]["sha256"]: raise RuntimeError("target hash mismatch")
    disjoint=json.loads(overlap.read_text())
    if disjoint["disjointness_gate"]!="PASS" or disjoint["max_overlap"]!=0: raise RuntimeError("target disjointness gate failed")
    for path,expected in config["implementation_hashes"].items():
        if _sha(ROOT/path)!=expected: raise RuntimeError(f"implementation hash mismatch: {path}")
    output=args.output_dir.resolve(); output.mkdir(parents=True,exist_ok=False)
    jobs=[{"branch":branch,"group":group} for branch in ("raw","dna","dp") for group in range(config["target"]["groups"])]
    (output/"execution_manifest.json").write_text(json.dumps({"created_at":datetime.now(timezone.utc).isoformat(),"config_sha256":_sha(args.config),"target_sha256":_sha(target),"jobs":jobs,"workers":args.workers},indent=2)+"\n")
    rows=[]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures=[executor.submit(_job,job,args.python,output,config,target) for job in jobs]
        for future in as_completed(futures):
            row=future.result(); rows.append(row); print(json.dumps({key:row[key] for key in ("branch","group","status","seconds")}),flush=True)
    summary={"jobs":len(rows),"success":sum(r["status"]=="SUCCESS" for r in rows),"failed":sum(r["status"]=="FAILED" for r in rows),"records":sorted(rows,key=lambda r:(r["branch"],r["group"]))}
    (output/"execution_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps({key:summary[key] for key in ("jobs","success","failed")},indent=2)); raise SystemExit(0 if summary["failed"]==0 else 1)


if __name__=="__main__": main()

