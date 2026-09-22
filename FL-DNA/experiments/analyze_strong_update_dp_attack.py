"""Aggregate gates and selected full-feature MSE for strong update-DP sweep."""
from __future__ import annotations
import argparse,csv,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,torch
ROOT=Path(__file__).resolve().parents[1]
BRANCHES=("raw","epsilon_100","epsilon_50","epsilon_10","epsilon_1")
def tail(w,n):return sum(math.comb(n,k) for k in range(w,n+1))/2**n
def gate(vals):
 w=sum(v<0 for v in vals);t=sum(v==0 for v in vals);l=sum(v>0 for v in vals);n=w+l
 return {"n":len(vals),"wins":w,"losses":l,"ties":t,"non_tied_n":n,"mean_difference":float(np.mean(vals)),"median_difference":float(np.median(vals)),"one_sided_sign_p":tail(w,n) if n else 1.0,"gate":bool(np.mean(vals)<0 and np.median(vals)<0 and n and tail(w,n)<.05)}
def choose(ptfiles):
 rows=[]
 for f in ptfiles:
  x=torch.load(f,map_location="cpu",weights_only=False); rows.append((float(x["best_objective"]),x))
 return min(rows,key=lambda z:z[0])[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False);allrows=[];branches={}
 manifest=json.loads((a.run_dir/"manifest.json").read_text()) if (a.run_dir/"manifest.json").exists() else {}
 groups=int(manifest.get("groups",8))
 for branch in BRANCHES:
  summaries=[]
  for group in range(groups):
   run=a.run_dir/branch/f"group_{group}_run"
   report=next(run.rglob("*report.json")); data=json.loads(report.read_text()); s=data["group_summary"][0]
   if branch=="raw": prior=s["objective_minus_prior_fraud_mse"];zero=s["objective_minus_zero_fraud_mse"]; pts=list(run.rglob("baseline.pt"));method="baseline"
   else: prior=s["objective_minus_prior_fraud_mse"];zero=s["objective_minus_zero_fraud_mse"];pts=list(run.rglob("baseline.pt"));method="baseline"
   best=choose(pts);original=best["original"].float();aligned=best["aligned"].float();fmse=float(torch.mean((original-aligned).square()))
   row={"branch":branch,"group":group,"selected_method":method,"feature_mse":fmse,"fraud_mse":float(s["objective_best_fraud_mse"]),"prior_difference_fraud_mse":float(prior),"zero_difference_fraud_mse":float(zero)};summaries.append(row);allrows.append(row)
  pg=gate([r["prior_difference_fraud_mse"] for r in summaries]);zg=gate([r["zero_difference_fraud_mse"] for r in summaries]);branches[branch]={"prior_gate":pg,"zero_gate":zg,"branch_gate_pass":bool(pg["gate"] and zg["gate"]),"feature_mse":{"mean":float(np.mean([r["feature_mse"] for r in summaries])),"median":float(np.median([r["feature_mse"] for r in summaries]))}}
 with (out/"selected_feature_mse.csv").open("w",newline="") as h:w=csv.DictWriter(h,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 payload={"created_at":datetime.now(timezone.utc).isoformat(),"scope":"development-only strong update-DP attacker gates and feature-MSE; not confirmatory", "branches":branches,"note":"A failed branch gate means its selected feature-MSE is reported as unresolved for scientific reconstruction-resistance interpretation."};(out/"attack_summary.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
