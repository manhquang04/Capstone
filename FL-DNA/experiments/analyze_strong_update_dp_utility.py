"""Analyze frozen strong update-DP development utility sweep."""
from __future__ import annotations
import argparse, csv, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def summary(x):
    a=np.asarray(x,dtype=float); se=float(a.std(ddof=1)/np.sqrt(a.size)) if a.size>1 else 0.
    return {"n":int(a.size),"mean":float(a.mean()),"sd":float(a.std(ddof=1)) if a.size>1 else 0.,"ci95_normal": [float(a.mean()-1.96*se),float(a.mean()+1.96*se)]}

def main():
    p=argparse.ArgumentParser();p.add_argument("--run-dir",type=Path,required=True);p.add_argument("--output-dir",type=Path,required=True);a=p.parse_args()
    registry={}
    for f in a.run_dir.glob("seed_*/**/metrics.json"):
        seed=int(f.parts[-3].split("_")[1]); method=f.parent.name; d=json.loads(f.read_text()); last=d["rounds"][-1]
        registry.setdefault(seed,{})[method]={"f1":float(last["f1_score"]),"auc":float(last["auc_roc"])}
    expected={"baseline","epsilon_100","epsilon_50","epsilon_10","epsilon_1"}
    if set(registry)!={101,202,303} or any(set(x)!=expected for x in registry.values()): raise RuntimeError("incomplete paired results")
    rows=[]; analyses={}
    for method in sorted(expected-{"baseline"}):
        f1=[];auc=[]
        for seed in sorted(registry):
            b=registry[seed]["baseline"]; v=registry[seed][method];f1.append(v["f1"]-b["f1"]);auc.append(v["auc"]-b["auc"])
            rows.append({"seed":seed,"method":method,"baseline_f1":b["f1"],"dp_f1":v["f1"],"delta_f1":f1[-1],"baseline_auc":b["auc"],"dp_auc":v["auc"],"delta_auc":auc[-1]})
        analyses[method]={"delta_f1":summary(f1),"delta_auc":summary(auc)}
    out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=False)
    with (out/"paired_utility.csv").open("w",newline="") as h: w=csv.DictWriter(h,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    payload={"created_at":datetime.now(timezone.utc).isoformat(),"scope":"development-only descriptive strong update-DP utility results", "n_paired_seeds":3,"analyses":analyses,"warning":"Normal CIs with n=3 are descriptive only; no confirmatory inference."}
    (out/"utility_summary.json").write_text(json.dumps(payload,indent=2)+"\n");print(json.dumps(payload,indent=2))
if __name__=="__main__":main()
