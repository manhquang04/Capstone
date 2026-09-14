"""Analyze the single frozen RQ1 confirmatory attack batch."""
from __future__ import annotations
import argparse, csv, json, math
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
from scipy.stats import beta, binomtest, t
from attacks.inversion_metrics import reconstruction_metrics

ROOT=Path(__file__).resolve().parents[1]; THRESHOLD=0.0390625

def best(paths, method):
    rows=[]
    for p in paths:
        a=torch.load(p,map_location="cpu",weights_only=False)
        if a.get("method")==method: rows.append((float(a["best_objective"]),str(p),p,a))
    if not rows: raise RuntimeError(f"no {method} artifact")
    return min(rows,key=lambda x:(x[0],x[1]))[2:]

def metrics(a):
    return reconstruction_metrics(a["original"].detach().cpu().numpy(),a.get("aligned",a["reconstruction"]).detach().cpu().numpy())

def ci(values, confidence=.95):
    x=np.asarray(values,float); mean=float(x.mean())
    if len(x)<2: return [mean,mean]
    h=float(t.ppf((1+confidence)/2,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x)))
    return [mean-h,mean+h]

def gate(values):
    x=np.asarray(values,float); wins=int((x<0).sum()); p=float(binomtest(wins,len(x),.5,alternative="greater").pvalue)
    return {"n":len(x),"wins":wins,"mean_difference":float(x.mean()),"median_difference":float(np.median(x)),"one_sided_sign_p":p,"pass":bool(x.mean()<0 and np.median(x)<0 and p<.05)}

def main():
    p=argparse.ArgumentParser(); p.add_argument("--run-dir",type=Path,default=ROOT/"artifacts/rq1/confirmatory_run_20260913"); p.add_argument("--output-dir",type=Path,default=ROOT/"results/rq1/confirmatory_20260913"); a=p.parse_args()
    rows=[]; control={b:{"prior":[],"zero":[]} for b in ("raw","dna","dp")}
    for gid in range(39):
        selected={}
        for branch in ("raw","dna","dp"):
            root=a.run_dir/branch/f"group_{gid}_run"
            bp,ba=best(list(root.glob("**/baseline.pt")),"baseline"); zp,za=best(list(root.glob("**/zero_update.pt")),"zero_update")
            report_path=next(root.glob("**/*_report.json")); report=json.loads(report_path.read_text()); gs=report["group_summary"][0]
            control[branch]["prior"].append(float(gs["objective_minus_prior_fraud_mse"])); control[branch]["zero"].append(float(gs["objective_minus_zero_fraud_mse"]))
            m=metrics(ba); selected[branch]=(ba,m)
            rows.append({"group_id":gid,"branch":branch,"selected_artifact":str(bp),"selection_objective":float(ba["best_objective"]),**m})
        ids=[list(map(int,selected[b][0]["source_ids"])) for b in ("raw","dna","dp")]
        if not ids[0]==ids[1]==ids[2]: raise AssertionError(f"source mismatch group {gid}")
    gates={b:{c:gate(control[b][c]) for c in ("prior","zero")} for b in control}
    for b in gates: gates[b]["pass"]=gates[b]["prior"]["pass"] and gates[b]["zero"]["pass"]
    by={(r["group_id"],r["branch"]):r for r in rows}; paired=[]
    for gid in range(39):
        r,d,pd=by[(gid,"raw")],by[(gid,"dna")],by[(gid,"dp")]; D=d["feature_mse"]-pd["feature_mse"]
        paired.append({"group_id":gid,"mse_raw":r["feature_mse"],"mse_dna":d["feature_mse"],"mse_dp":pd["feature_mse"],"delta_dna_raw":d["feature_mse"]-r["feature_mse"],"delta_dp_raw":pd["feature_mse"]-r["feature_mse"],"D_dna_minus_dp":D,"tie":abs(D)<=THRESHOLD,"psnr_raw":r["psnr"],"psnr_dna":d["psnr"],"psnr_dp":pd["psnr"],"ssim_raw":r["ssim"],"ssim_dna":d["ssim"],"ssim_dp":pd["ssim"]})
    non=[x for x in paired if not x["tie"]]; wins=sum(x["D_dna_minus_dp"]>THRESHOLD for x in non); losses=len(non)-wins
    pval=float(binomtest(wins,len(non),.5,alternative="greater").pvalue) if non else 1.0
    win_ci=[0.0 if wins==0 else float(beta.ppf(.025,wins,len(non)-wins+1)),1.0 if wins==len(non) else float(beta.ppf(.975,wins+1,len(non)-wins))]
    primary_valid=gates["dna"]["pass"] and gates["dp"]["pass"]
    summary={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"scope":"single frozen RQ1 confirmatory execution","n_targets":39,"tie_threshold_mse":THRESHOLD,"branch_gates":gates,"primary":{"contrast":"DNA_TRANSFORM_MINUS_DP_DISTORTION_MATCHED","valid":primary_valid,"wins":wins,"losses":losses,"ties":39-len(non),"non_tied_n":len(non),"win_probability":wins/len(non) if non else None,"win_probability_exact_ci95":win_ci,"one_sided_exact_sign_p":pval,"reject_h0":bool(primary_valid and pval<.05),"mean_mse_difference":float(np.mean([x["D_dna_minus_dp"] for x in paired])),"mean_mse_difference_ci95":ci([x["D_dna_minus_dp"] for x in paired]),"conclusion":"DNA shows significantly greater reconstruction resistance than distortion-matched clipping/noise" if primary_valid and pval<.05 else ("No significant DNA advantage over distortion-matched clipping/noise under the frozen test" if primary_valid else "Primary contrast withheld because a required branch gate failed")},"secondary_descriptive":{"dna_minus_raw_mse_ci95":ci([x["delta_dna_raw"] for x in paired]),"dp_minus_raw_mse_ci95":ci([x["delta_dp_raw"] for x in paired]),"dna_minus_dp_psnr_ci95":ci([x["psnr_dna"]-x["psnr_dp"] for x in paired]),"dna_minus_dp_ssim_ci95":ci([x["ssim_dna"]-x["ssim_dp"] for x in paired])}}
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,data in (("rq1_selected_metrics.csv",rows),("rq1_paired_metrics.csv",paired)):
        with (a.output_dir/name).open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    (a.output_dir/"rq1_summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))

if __name__=="__main__": main()
