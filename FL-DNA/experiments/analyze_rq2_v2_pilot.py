"""Analyze pre-frozen v2 pilot and derive conservative required RQ2 sample size."""
from __future__ import annotations
import argparse,csv,json,math,warnings
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import chi2,nct,t
ROOT=Path(__file__).resolve().parents[1]
MARGINS={'f1':.02,'auc':.005}
def upper_var(x):
 v=float(np.var(x,ddof=1));return 0. if v==0 else float((len(x)-1)*v/chi2.ppf(.05,len(x)-1))
def need(sigma,margin):
 if sigma==0:return 2
 for n in range(2,10001):
  with warnings.catch_warnings():
   warnings.simplefilter('ignore',RuntimeWarning)
   if nct.sf(t.ppf(.95,n-1),n-1,margin*math.sqrt(n)/sigma)>=.8:return n
 raise RuntimeError('n too large')
def main():
 p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();by={}
 for f in a.run_dir.glob('seed_*/**/metrics.json'):
  seed=int(f.parts[-3].split('_')[1]); method=f.parent.name;d=json.loads(f.read_text());z=d['rounds'][-1]
  if int(d['config']['seed'])!=seed or int(z['round'])!=50:raise ValueError(f'contract error {f}')
  by.setdefault(seed,{})[method]={'f1':float(z['f1_score']),'auc':float(z['auc_roc'])}
 if len(by)!=8 or any(set(x)!={'baseline','dna_transform_v2'} for x in by.values()):raise RuntimeError('incomplete paired v2 pilot')
 rows=[];power=[]
 for endpoint in MARGINS:
  d=np.array([by[s]['dna_transform_v2'][endpoint]-by[s]['baseline'][endpoint] for s in sorted(by)])
  sd=float(np.std(d,ddof=1));uv=upper_var(d);n=need(math.sqrt(uv),MARGINS[endpoint]);half=float(t.ppf(.975,7)*sd/math.sqrt(8))
  power.append({'endpoint':endpoint,'n_pilot':8,'mean_delta':float(d.mean()),'sd':sd,'ci95':[float(d.mean()-half),float(d.mean()+half)],'upper_95pct_variance':uv,'margin':MARGINS[endpoint],'required_n':n})
 for s in sorted(by):rows.append({'seed':s,'baseline_f1':by[s]['baseline']['f1'],'v2_f1':by[s]['dna_transform_v2']['f1'],'delta_f1':by[s]['dna_transform_v2']['f1']-by[s]['baseline']['f1'],'baseline_auc':by[s]['baseline']['auc'],'v2_auc':by[s]['dna_transform_v2']['auc'],'delta_auc':by[s]['dna_transform_v2']['auc']-by[s]['baseline']['auc']})
 req=max(r['required_n'] for r in power);o=a.output_dir.resolve();o.mkdir(parents=True,exist_ok=False)
 with (o/'per_seed.csv').open('w',newline='')as h:w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 out={'created_at':datetime.now(timezone.utc).isoformat(),'scope':'RQ2-v2 pre-frozen pilot only','seeds':sorted(by),'power_method':'one-sided paired non-inferiority noncentral-t; 95% chi-square upper variance bound','alpha':.05,'power':.8,'endpoint_rule':'both F1 and AUC endpoints required','rows':power,'required_confirmatory_seed_count':req,'maximum_feasible_seed_count':30,'within_cap':req<=30}
 (o/'pilot_summary.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
