"""Apply the frozen strict and relaxed-sensitivity utility matching rules."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [101, 202, 303, 404, 505, 606, 707, 808]
MULTIPLIERS = [0.00001, 0.000025, 0.00005, 0.000075]


def _final(path: Path) -> dict[str, float]:
    document = json.loads(path.read_text())
    row = document["rounds"][-1]
    if int(row["round"]) != 50:
        raise ValueError(f"incomplete run {path}")
    return {"f1": float(row["f1_score"]), "auc": float(row["auc_roc"])}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, default=ROOT / "artifacts/rq2/stage1_development_20260912")
    parser.add_argument("--extension", type=Path, default=ROOT / "artifacts/rq2/utility_extension_20260913")
    args = parser.parse_args()
    rows=[]
    dna_delta={"f1":[],"auc":[]}
    per_seed={}
    for seed in SEEDS:
        baseline=_final(args.original/f"seed_{seed}/baseline/metrics.json")
        dna=_final(args.original/f"seed_{seed}/dna_transform/metrics.json")
        per_seed[seed]=(baseline,dna)
        for endpoint in dna_delta: dna_delta[endpoint].append(dna[endpoint]-baseline[endpoint])
    dna_mean={endpoint:float(np.mean(values)) for endpoint,values in dna_delta.items()}
    for multiplier in MULTIPLIERS:
        method=f"dp_{multiplier:g}"; deltas={"f1":[],"auc":[]}
        for seed in SEEDS:
            baseline,_=per_seed[seed]; dp=_final(args.extension/f"seed_{seed}/{method}/metrics.json")
            for endpoint in deltas: deltas[endpoint].append(dp[endpoint]-baseline[endpoint])
        mean={endpoint:float(np.mean(values)) for endpoint,values in deltas.items()}
        deviations={endpoint:abs(mean[endpoint]-dna_mean[endpoint]) for endpoint in mean}
        strict=deviations["f1"]<=0.01 and deviations["auc"]<=0.0025
        relaxed=deviations["f1"]<=0.02 and deviations["auc"]<=0.005
        rows.append({"method":method,"noise_multiplier":multiplier,"mean_delta_f1":mean["f1"],"mean_delta_auc":mean["auc"],"dna_mean_delta_f1":dna_mean["f1"],"dna_mean_delta_auc":dna_mean["auc"],"absolute_f1_deviation":deviations["f1"],"absolute_auc_deviation":deviations["auc"],"strict_distance":max(deviations["f1"]/0.01,deviations["auc"]/0.0025),"strict_eligible":strict,"relaxed_distance":max(deviations["f1"]/0.02,deviations["auc"]/0.005),"relaxed_eligible":relaxed})
    strict=[row for row in rows if row["strict_eligible"]]
    relaxed=[row for row in rows if row["relaxed_eligible"]]
    if strict:
        selected=min(strict,key=lambda row:(row["strict_distance"],row["noise_multiplier"])); status="UTILITY_MATCHED_STRICT"
    elif relaxed:
        selected=min(relaxed,key=lambda row:(row["relaxed_distance"],row["noise_multiplier"])); status="UTILITY_MATCHED_RELAXED_SENSITIVITY"
    else:
        selected=None; status="NOT_FOUND"
    with (args.extension/"utility_extension_grid.csv").open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report={"scope":"development-only one-time utility extension","seeds":SEEDS,"dna_mean_delta":dna_mean,"strict_tolerances":{"f1":0.01,"auc":0.0025},"relaxed_sensitivity_tolerances":{"f1":0.02,"auc":0.005},"selection_priority":"strict_then_relaxed_sensitivity_then_NOT_FOUND","status":status,"selected_candidate":selected,"search_closed_permanently":True,"privacy_metrics_used":False,"rows":rows}
    (args.extension/"utility_extension_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__ == "__main__":
    main()
