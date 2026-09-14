"""Analyze one frozen medium/stronger RQ1 confirmatory family."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta, binomtest, t

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from attacks.inversion_metrics import reconstruction_metrics


def _best(paths: list[Path], method: str):
    rows=[]
    for path in paths:
        artifact=torch.load(path,map_location="cpu",weights_only=False)
        if artifact.get("method")==method: rows.append((float(artifact["best_objective"]),str(path),path,artifact))
    if not rows: raise RuntimeError(f"no {method} artifact")
    return min(rows,key=lambda row:(row[0],row[1]))[2:]


def _metrics(artifact: dict) -> dict:
    return reconstruction_metrics(artifact["original"].detach().cpu().numpy(),artifact.get("aligned",artifact["reconstruction"]).detach().cpu().numpy())


def _ci(values: list[float]) -> list[float]:
    x=np.asarray(values,float); mean=float(x.mean()); half=float(t.ppf(.975,len(x)-1)*x.std(ddof=1)/math.sqrt(len(x))) if len(x)>1 else 0.0
    return [mean-half,mean+half]


def _gate(values: list[float]) -> dict:
    x=np.asarray(values,float); wins=int((x<0).sum()); p=float(binomtest(wins,len(x),.5,alternative="greater").pvalue)
    return {"n":len(x),"wins":wins,"mean_difference":float(x.mean()),"median_difference":float(np.median(x)),"one_sided_sign_p":p,"pass":bool(x.mean()<0 and np.median(x)<0 and p<.05)}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",type=Path,required=True); parser.add_argument("--run-dir",type=Path,required=True); parser.add_argument("--output-dir",type=Path,required=True); args=parser.parse_args()
    config=json.loads(args.config.read_text()); groups=config["target"]["groups"]; threshold=config["statistics"]["tie_threshold_mse"]
    rows=[]; control={branch:{"prior":[],"zero":[]} for branch in ("raw","dna","dp")}
    for group in range(groups):
        selected={}
        for branch in ("raw","dna","dp"):
            root=args.run_dir/branch/f"group_{group}_run"; artifact_path,artifact=_best(list(root.glob("**/baseline.pt")),"baseline"); _,zero=_best(list(root.glob("**/zero_update.pt")),"zero_update")
            report=json.loads(next(root.glob("**/*_report.json")).read_text()); summary=report["group_summary"][0]
            control[branch]["prior"].append(float(summary["objective_minus_prior_fraud_mse"])); control[branch]["zero"].append(float(summary["objective_minus_zero_fraud_mse"]))
            metric=_metrics(artifact); selected[branch]=(artifact,metric); rows.append({"group_id":group,"branch":branch,"selected_artifact":str(artifact_path),"selection_objective":float(artifact["best_objective"]),**metric})
        ids=[list(map(int,selected[branch][0]["source_ids"])) for branch in ("raw","dna","dp")]
        if not ids[0]==ids[1]==ids[2]: raise AssertionError(f"source mismatch {group}")
    gates={branch:{control_name:_gate(control[branch][control_name]) for control_name in ("prior","zero")} for branch in control}
    for branch in gates: gates[branch]["pass"]=gates[branch]["prior"]["pass"] and gates[branch]["zero"]["pass"]
    lookup={(row["group_id"],row["branch"]):row for row in rows}; paired=[]
    for group in range(groups):
        raw,dna,dp=(lookup[(group,branch)] for branch in ("raw","dna","dp")); difference=dna["feature_mse"]-dp["feature_mse"]
        paired.append({"group_id":group,"mse_raw":raw["feature_mse"],"mse_dna":dna["feature_mse"],"mse_dp":dp["feature_mse"],"delta_dna_raw":dna["feature_mse"]-raw["feature_mse"],"delta_dp_raw":dp["feature_mse"]-raw["feature_mse"],"D_dna_minus_dp":difference,"tie":abs(difference)<=threshold,"psnr_raw":raw["psnr"],"psnr_dna":dna["psnr"],"psnr_dp":dp["psnr"],"ssim_raw":raw["ssim"],"ssim_dna":dna["ssim"],"ssim_dp":dp["ssim"]})
    non_tied=[row for row in paired if not row["tie"]]; wins=sum(row["D_dna_minus_dp"]>threshold for row in non_tied); losses=len(non_tied)-wins; p=float(binomtest(wins,len(non_tied),.5,alternative="greater").pvalue) if non_tied else 1.0
    win_ci=[0.0 if wins==0 else float(beta.ppf(.025,wins,losses+1)),1.0 if losses==0 else float(beta.ppf(.975,wins+1,losses))]
    valid=gates["dna"]["pass"] and gates["dp"]["pass"]
    summary={"variant":config["variant"],"n_targets":groups,"branch_gates":gates,"primary":{"valid":valid,"wins":wins,"losses":losses,"ties":groups-len(non_tied),"non_tied_n":len(non_tied),"win_probability":wins/len(non_tied) if non_tied else None,"win_probability_exact_ci95":win_ci,"one_sided_exact_sign_p":p,"reject_h0":bool(valid and p<.05),"mean_mse_difference":float(np.mean([row["D_dna_minus_dp"] for row in paired])),"mean_mse_difference_ci95":_ci([row["D_dna_minus_dp"] for row in paired]),"conclusion":"DNA shows significantly greater reconstruction resistance than distortion-matched clipping/noise" if valid and p<.05 else ("No significant DNA advantage over distortion-matched clipping/noise under the frozen test" if valid else "Primary contrast withheld because a required branch gate failed")},"secondary":{"dna_minus_raw_mse_ci95":_ci([row["delta_dna_raw"] for row in paired]),"dp_minus_raw_mse_ci95":_ci([row["delta_dp_raw"] for row in paired]),"dna_minus_dp_psnr_ci95":_ci([row["psnr_dna"]-row["psnr_dp"] for row in paired]),"dna_minus_dp_ssim_ci95":_ci([row["ssim_dna"]-row["ssim_dp"] for row in paired])}}
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for name,data in (("selected_metrics.csv",rows),("paired_metrics.csv",paired)):
        with (args.output_dir/name).open("w",newline="") as handle: writer=csv.DictWriter(handle,fieldnames=list(data[0])); writer.writeheader(); writer.writerows(data)
    (args.output_dir/"summary.json").write_text(json.dumps(summary,indent=2)+"\n"); print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
