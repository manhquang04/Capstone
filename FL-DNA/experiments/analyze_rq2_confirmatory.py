"""Analyze the single frozen paired RQ2 confirmatory batch."""
from __future__ import annotations
import argparse,csv,json,math
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from scipy.stats import t
ROOT=Path(__file__).resolve().parents[1]; METHODS=["dna_transform","dna_lossless","dp_0.00025"]

def stats(values,margin=None):
    x=np.asarray(values,float); n=len(x); mean=float(x.mean()); sd=float(x.std(ddof=1)); half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)); low,high=mean-half,mean+half
    out={"n":n,"mean":mean,"sd":sd,"median":float(np.median(x)),"ci95":[low,high]}
    if margin is not None: out.update({"noninferiority_margin":margin,"noninferiority_pass":low>=-margin})
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/rq2/confirmatory_run_20260913"); p.add_argument("--output-dir",type=Path,default=ROOT/"results/rq2/confirmatory_20260913"); a=p.parse_args()
    values={}; rows=[]
    for path in sorted(a.run_dir.glob("seed_*/**/metrics.json")):
        seed=int(path.parts[-3].split("_")[1]); method=path.parent.name; d=json.loads(path.read_text()); final=d["rounds"][-1]
        if int(d["config"]["seed"])!=seed or int(final["round"])!=50: raise ValueError(f"seed/round mismatch {path}")
        values.setdefault(seed,{})[method]={"f1":float(final["f1_score"]),"auc":float(final["auc_roc"])}
    expected={"baseline","dna_lossless","dna_transform","dp_0.00025"}
    if len(values)!=21 or any(set(v)!=expected for v in values.values()): raise RuntimeError("incomplete paired registry")
    for seed in sorted(values):
        b=values[seed]["baseline"]
        for method,v in values[seed].items(): rows.append({"seed":seed,"method":method,"f1":v["f1"],"auc":v["auc"],"delta_f1":v["f1"]-b["f1"],"delta_auc":v["auc"]-b["auc"]})
    analyses={}
    for method in METHODS:
        rs=[r for r in rows if r["method"]==method]; analyses[method]={"f1":stats([r["delta_f1"] for r in rs],.02 if method=="dna_transform" else None),"auc":stats([r["delta_auc"] for r in rs],.005 if method=="dna_transform" else None)}
    tr=analyses["dna_transform"]; f1pass=tr["f1"]["noninferiority_pass"]; aucpass=tr["auc"]["noninferiority_pass"]
    conclusion=("DNA Transform establishes non-inferiority for both F1 and AUC-ROC at the approved margins" if f1pass and aucpass else ("DNA Transform không thiết lập được non-inferiority F1 ở margin đã duyệt" if not f1pass else "DNA Transform establishes F1 non-inferiority but not AUC-ROC non-inferiority at the approved margin"))
    summary={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"scope":"single frozen RQ2 confirmatory execution","n_seeds":21,"primary_hypothesis":"DNA_TRANSFORM_MINUS_BASELINE","primary_alpha":.05,"endpoint_rule":"both_f1_and_auc_must_pass","analyses":analyses,"primary_conclusion":conclusion,"lossless_role":"secondary technical/sanity check; not adjusted jointly with primary","dp_role":"secondary/contextual distortion-matched comparator"}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    with (a.output_dir/"rq2_per_seed.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    (a.output_dir/"rq2_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
