"""Independent recomputation directly from job JSON; no primary analysis imports."""
import csv
import json
import math
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"artifacts/priority32_multidataset"


def main():
    summary = list(csv.DictReader((OUT/"rq2_analysis/combined_summary.csv").open()))
    maximum = 0.
    for row in summary:
        differences, base_values, defended_values = [], [], []
        for seed in range(321000, 321021):
            base = json.loads((OUT/"rq2_jobs"/row["dataset"]/"baseline"/f"seed_{seed}.json").read_text())["test"][row["endpoint"]]
            defense = json.loads((OUT/"rq2_jobs"/row["dataset"]/row["method"]/f"seed_{seed}.json").read_text())["test"][row["endpoint"]]
            differences.append(defense-base); base_values.append(base); defended_values.append(defense)
        # Independent summation and scipy SEM/interval, rather than primary sd/sqrt formula.
        mean = math.fsum(differences)/len(differences)
        sd = math.sqrt(math.fsum((v-mean)**2 for v in differences)/(len(differences)-1))
        if sd == 0:
            interval = (mean, mean)
        else:
            interval = stats.t.interval(.95, df=len(differences)-1, loc=mean, scale=stats.sem(differences))
        checks = {"mean_delta": mean, "sd_delta": sd, "ci95_lower": interval[0], "ci95_upper": interval[1],
                  "baseline_mean": math.fsum(base_values)/21, "defended_mean": math.fsum(defended_values)/21}
        for key, value in checks.items():
            error = abs(float(row[key])-value)
            maximum = max(maximum, error)
            assert error <= 1e-12, (row, key, value)
        if row["margin"]:
            assert row["gate"] == ("PASS" if interval[0] > -float(row["margin"]) else "FAIL")
    result = {"pass": True, "contrasts_endpoints": len(summary), "max_abs_error": maximum,
              "method": "independent math.fsum/sample variance + scipy.stats.sem/t.interval from original per-job JSON"}
    (OUT/"rq2_analysis/independent_verification.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
