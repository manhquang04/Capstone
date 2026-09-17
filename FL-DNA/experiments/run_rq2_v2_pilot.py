"""Execute the pre-frozen paired RQ2 DNA Transform v2 pilot."""
from __future__ import annotations
import argparse,json,os,subprocess,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SEEDS=[733003850,1065351580,880535557,48511685,1771003338,672591598,589150035,848504097]
def run(job,python,out):
 seed,method=job['seed'],job['method'];d=out/f'seed_{seed}'/method;d.mkdir(parents=True,exist_ok=False);metric=d/'metrics.json'
 if method=='baseline': script,outvar='experiments/run_fraud_fl_baseline.py','BASELINE_OUTPUT_PATH'
 else: script,outvar='experiments/run_fraud_fl_dna_transform_v2.py','DNA_TRANSFORM_V2_OUTPUT_PATH'
 env=os.environ.copy();env.update({'PYTHONPATH':str(ROOT),'FL_RUN_SEED':str(seed),'MAX_ROWS':'500000','NUM_ROUNDS':'50','LOCAL_EPOCHS':'1','FL_NUM_CLIENTS':'3','LOSS_TYPE':'focal','FOCAL_ALPHA':'0.95','FOCAL_GAMMA':'2.0','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','VECLIB_MAXIMUM_THREADS':'1','NUMEXPR_NUM_THREADS':'1',outvar:str(metric)})
 if method=='dna_transform_v2':env.update({'DNA_TRANSFORM_V2_COMPRESSION_RATIO':'0.95','DNA_TRANSFORM_V2_QUANTIZATION_ETA':'0.01','DNA_TRANSFORM_V2_BASE_SEED':'20260916'})
 cmd=[python,script];t=time.perf_counter()
 with (d/'stdout.log').open('w') as so,(d/'stderr.log').open('w') as se:p=subprocess.run(cmd,cwd=ROOT,env=env,stdout=so,stderr=se,check=False)
 r={**job,'command':cmd,'returncode':p.returncode,'seconds':time.perf_counter()-t,'status':'SUCCESS' if p.returncode==0 and metric.exists() else 'FAILED'};(d/'job_record.json').write_text(json.dumps(r,indent=2)+'\n');return r
def main():
 p=argparse.ArgumentParser();p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=8);p.add_argument('--python',default=str(ROOT/'.venv-phase1/bin/python'));a=p.parse_args()
 if not 1<=a.workers<=9:raise ValueError('workers must be 1..9')
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);jobs=[{'seed':s,'method':m}for s in SEEDS for m in ('baseline','dna_transform_v2')]
 (out/'manifest.json').write_text(json.dumps({'created_at':datetime.now(timezone.utc).isoformat(),'scope':'pre-frozen RQ2 v2 pilot','amendment':'protocols/amendments/2026-09-16_rq2_v2_pilot_and_confirmatory.md','seeds':SEEDS,'jobs':jobs},indent=2)+'\n')
 rows=[]
 with ThreadPoolExecutor(max_workers=a.workers)as ex:
  fs=[ex.submit(run,j,a.python,out)for j in jobs]
  for f in as_completed(fs):
   r=f.result();rows.append(r);print(json.dumps({k:r[k]for k in('seed','method','status','seconds')}),flush=True)
 s={'created_at':datetime.now(timezone.utc).isoformat(),'jobs':len(rows),'success':sum(r['status']=='SUCCESS'for r in rows),'failed':sum(r['status']=='FAILED'for r in rows),'records':sorted(rows,key=lambda x:(x['seed'],x['method']))};(out/'execution_summary.json').write_text(json.dumps(s,indent=2)+'\n')
 if s['failed']:raise SystemExit(1)
if __name__=='__main__':main()
