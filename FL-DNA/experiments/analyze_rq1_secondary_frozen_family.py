"""Apply the predeclared Holm rule to the frozen RQ1 secondary family.

This script only implements the secondary family already specified in the
approved protocol. It does not alter the primary analysis or experiment.
"""
from __future__ import annotations
import csv,json
from datetime import datetime,timezone
from pathlib import Path
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]

def test(name,values,direction):
    values=[float(v) for v in values if float(v)!=0.0]
    wins=sum(v>0 for v in values) if direction=="greater" else sum(v<0 for v in values)
    return {"contrast":name,"direction":direction,"wins":wins,"n":len(values),"unadjusted_p":float(binomtest(wins,len(values),.5,alternative="greater").pvalue)}

def main():
    rows=list(csv.DictReader(open(ROOT/"results/rq1/confirmatory_20260913/rq1_paired_metrics.csv")))
    tests=[
        test("DNA_vs_RAW_normalized_feature_MSE",[r["delta_dna_raw"] for r in rows],"greater"),
        test("DP_vs_RAW_normalized_feature_MSE",[r["delta_dp_raw"] for r in rows],"greater"),
        test("DNA_vs_DP_PSNR",[float(r["psnr_dna"])-float(r["psnr_dp"]) for r in rows],"less"),
        test("DNA_vs_DP_SSIM",[float(r["ssim_dna"])-float(r["ssim_dp"]) for r in rows],"less"),
    ]
    order=sorted(range(len(tests)),key=lambda i:tests[i]["unadjusted_p"]); running=0.0
    for rank,i in enumerate(order):
        running=max(running,(len(tests)-rank)*tests[i]["unadjusted_p"]); tests[i]["holm_adjusted_p"]=min(1.0,running); tests[i]["reject_at_0p05"]=tests[i]["holm_adjusted_p"]<.05
    report={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"family":"predeclared RQ1 secondary family","adjustment":"Holm","alpha":.05,"tests":tests,"any_rejection":any(x["reject_at_0p05"] for x in tests)}
    out=ROOT/"results/rq1/confirmatory_20260913/rq1_secondary_holm.json"; out.write_text(json.dumps(report,indent=2)+"\n"); print(json.dumps(report,indent=2))

if __name__=="__main__": main()
