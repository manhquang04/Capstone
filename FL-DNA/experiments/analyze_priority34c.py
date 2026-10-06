"""P34C independent audit, fixed-family statistics and report (no training)."""
from __future__ import annotations
import argparse
import csv
import json
import math
import py_compile
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from experiments import priority34c_confirmatory as c
from experiments import priority34c_qualify as q
from experiments import priority34c_statistics as stats
from scipy.stats import binomtest

OUT, np, torch = c.OUT, c.np, c.torch


def independent_p(wins, losses):
    return float(binomtest(wins, wins+losses, .5, alternative="greater").pvalue) if wins+losses else 1.


def independent_holm(values):
    if len(values) != 72:
        raise ValueError("fixed family must have72 tests")
    order = np.argsort(np.asarray(values), kind="stable")
    sorted_adjusted = np.minimum(1., np.maximum.accumulate(np.asarray(values)[order]*np.arange(72, 0, -1)))
    result = np.empty(72)
    result[order] = sorted_adjusted
    return result.tolist()


def independent_accounting(sd):
    sigma = sd/.02
    rho = 50/(2*sigma*sigma)
    alpha = 1+math.sqrt(math.log(1e5)/rho)
    return dict(sigma=sigma, alpha=alpha, epsilon=rho*alpha+math.log(1e5)/(alpha-1))


def reload_score(path, row, data, qualification=False):
    attempt = path.parent / row["attempt"]
    receipt = torch.load(attempt / "server_receipt.pt", map_location="cpu", weights_only=False)
    truth = torch.load(attempt / "scoring_truth.pt", map_location="cpu", weights_only=False)
    reconstruction = torch.load(attempt / "reconstruction.pt", map_location="cpu", weights_only=False)
    if qualification:
        selected = reconstruction["reconstruction"]
    else:
        index = reconstruction["selected"]
        variants = reconstruction["candidates"]
        if index != min(range(len(variants)), key=lambda i: variants[i]["objective"]):
            raise ValueError("non-observable candidate selection")
        selected = variants[index]["reconstruction"]
    metric = c.native.measure(data, truth["truth"], selected)
    if metric != row["accuracy"]:
        raise ValueError("reloaded mixed feature scores differ")
    ids = json.loads((OUT / row["config"]["dataset"] / "targets.json").read_text())["source_ids"][row["config"]["cell"]][row["config"]["stage"]][row["config"]["index"]]
    if ids != truth["source_ids"] or ids != row["source_ids"]:
        raise ValueError("reserved target ID mismatch")
    if row["device"] != "cpu" or row["intra_threads"] != 1 or row["inter_threads"] != 1 or not row["bn"]["no_bn_payload"]:
        raise ValueError("CPU/thread/noBN gate")
    names = list(receipt["payload"]) if qualification else receipt["payload_names"]
    if set(names) & set(row["bn"]["omitted_bn_parameters"]):
        raise ValueError("BN transmitted")
    if not qualification and any(name in receipt for name in ("raw_gradient", "noise_seed", "noise", "private_noise_seeds")):
        raise ValueError("attack receipt exposed private defender information")
    if not qualification and "loss" in receipt["bn"]:
        raise ValueError("raw private loss in protected attack receipt")
    return metric["accuracy_percent"]


def audit_qualification(data):
    audited = 0
    for stage, n in (("n8", 8), ("n24", 24)):
        gates = json.loads((OUT / ("gates_"+stage+".json")).read_text())
        for cell, gate in gates.items():
            dataset, instrument = cell.split("/")
            rows = []
            for index in range(n):
                config = dict(dataset=dataset, cell=instrument, stage=stage, index=index)
                path = q.folder_for(config) / "result.json"
                if not q.valid_result(path, config):
                    raise ValueError("qualification missing")
                row = json.loads(path.read_text())
                reload_score(path, row, data[cell], qualification=True)
                control_seed = row["control_seed"]
                truth = torch.load(path.parent / row["attempt"] / "scoring_truth.pt", map_location="cpu", weights_only=False)
                controls, _, _ = c.native.controls(data[cell], truth["truth"], control_seed)
                if controls != row["baselines"]:
                    raise ValueError("baseline reconstruction mismatch")
                rows.append(row)
                audited += 1
            for control, test in gate["tests"].items():
                diffs = [r["accuracy"]["accuracy_percent"]-r["baselines"][control] for r in rows]
                wins, losses = sum(v > 0 for v in diffs), sum(v < 0 for v in diffs)
                if wins != test["wins"] or losses != test["losses"] or abs(independent_p(wins, losses)-test["p"]) > 1e-14:
                    raise ValueError("independent qualification sign test mismatch")
            if gate["passed"] != all(t["p"] < .05 for t in gate["tests"].values()):
                raise ValueError("qualification gate mismatch")
    return audited


def audit_calibration():
    stored = json.loads((OUT / "distortion_calibration.json").read_text())
    for cell, methods in stored.items():
        dataset, instrument = cell.split("/")
        observations = []
        for index in range(24):
            config = dict(dataset=dataset, cell=instrument, index=index, stage="development24")
            path = c.development.folder_for(config) / "result.json"
            if not c.development.valid(path, config):
                raise ValueError("development output missing")
            row = json.loads(path.read_text())
            bundle = torch.load(path.parent / row["attempt"] / "defender_calibration_private.pt", map_location="cpu", weights_only=False)
            norm = float(torch.sqrt(sum(g.double().square().sum() for g in bundle["raw_gradient"].values())))
            if norm != row["raw_norm"]:
                raise ValueError("development norm mismatch")
            private = json.loads(c.development.PRIVATE.read_text())
            for method in c.DNA:
                raw = bundle["raw_gradient"]
                payload, _, metadata = c.defend(raw, method)
                decoded = payload if metadata is None else [torch.from_numpy(c.development.reconstruct_update_array_v2(v.numpy(), m)[0]) for v, m in zip(payload, metadata)]
                distortion = float(torch.sqrt(sum((a.double()-b.double()).square().sum() for a, b in zip(raw.values(), decoded))))
                generator = torch.Generator().manual_seed(private[c.development.key_for(config, method)])
                gaussian = float(torch.sqrt(sum(torch.randn(v.shape, dtype=torch.float64, generator=generator).square().sum() for v in raw.values())))
                measured = row["measurements"][method]
                if distortion != measured["dna_l2_distortion"] or gaussian != measured["unit_gaussian_norm"]:
                    raise ValueError("independent defender distortion/noise norm mismatch")
            observations.append(row)
        clip = 1.01*max(r["raw_norm"] for r in observations)
        for method, record in methods.items():
            distortions = np.asarray([r["measurements"][method]["dna_l2_distortion"] for r in observations])
            gaussian = np.asarray([r["measurements"][method]["unit_gaussian_norm"] for r in observations])
            sigma = float(np.median(distortions)/(clip*np.median(gaussian)))
            errors = np.abs(clip*sigma*gaussian-distortions)/np.where(distortions > 0, distortions, 1.)
            if record["clip"] != clip or record["sigma"] != sigma or record["median_relative_error"] != float(np.median(errors)):
                raise ValueError("independent developmental matching mismatch")
            if (record["status"] == "MATCHED") != (sigma > 0 and np.median(errors) <= .05):
                raise ValueError("distortion comparator gate mismatch")
    return stored


def run():
    if q.live_processes() or c.development.own_processes() or c.own_processes():
        raise RuntimeError("live P34C workload; final audit deferred")
    for stage in ("QUALIFICATION", "DEVELOPMENT", "CONFIRMATORY"):
        if (OUT / ("REQUIRES_DIRECTION_"+stage+".json")).exists():
            raise RuntimeError("scientific failure unresolved")
    if not (OUT / "CONFIRMATORY_COMPLETE.json").exists():
        raise RuntimeError("confirmatory stage incomplete")
    doc = c.verify()
    data = {d+"/"+cell: c.PreparedDataset(d, cell) for d in stats.DATASETS for cell in stats.CELLS}
    qualification_count = audit_qualification(data)
    calibration = audit_calibration()
    observations = {}
    for config in doc["jobs"]:
        path = c.folder_for(config) / "result.json"
        if not c.valid(path, config):
            raise ValueError("scheduled confirmatory output missing")
        row = json.loads(path.read_text())
        value = reload_score(path, row, data[config["dataset"]+"/"+config["cell"]])
        observations[(config["dataset"], config["cell"], config["arm"], config["index"])] = value
    gates = json.loads((OUT / "gates_n24.json").read_text())
    pairs, unavailable = {}, {}
    for d in stats.DATASETS:
        for cell in stats.CELLS:
            key = d+"/"+cell
            for arm in stats.DNA:
                for comparator in stats.COMPARATORS:
                    contrast = (d, cell, arm, comparator)
                    if key not in gates or not gates[key]["passed"]:
                        unavailable[contrast] = "unprotected qualification gate failed"
                    elif comparator == "distortion_matched" and calibration[key][arm]["status"] != "MATCHED":
                        unavailable[contrast] = "development distortion matching gate failed"
                    else:
                        dp = "dp_local_eps10" if comparator == "local_dp_eps10" else "dp_distortion_"+arm
                        pairs[contrast] = ([observations[(d, cell, arm, i)] for i in range(39)],
                                           [observations[(d, cell, dp, i)] for i in range(39)])
    tests = stats.evaluate_family(pairs, unavailable)
    with (OUT / "paired_observations.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["dataset", "cell", "defense", "comparator", "index", "dna_accuracy", "dp_accuracy", "dna_minus_dp"])
        for key, (dna_values, dp_values) in sorted(pairs.items()):
            for index, (dna, dp) in enumerate(zip(dna_values, dp_values)):
                writer.writerow([*key, index, dna, dp, dna-dp])
    for row in tests:
        if row["status"] == "ASSESSABLE":
            s = row["summary"]
            key = (row["dataset"], row["cell"], row["defense"], row["comparator"])
            dna, dp = pairs[key]
            effects = sorted(a-b for a, b in zip(dna, dp))
            if s["dna_median"] != sorted(dna)[19] or s["dp_median"] != sorted(dp)[19] or s["paired_effect_median"] != effects[19]:
                raise ValueError("independent medians mismatch")
            if s["paired_effect_order_interval"] != [effects[12], effects[26]] or s["dna_order_interval"] != [sorted(dna)[12], sorted(dna)[26]] or s["dp_order_interval"] != [sorted(dp)[12], sorted(dp)[26]]:
                raise ValueError("independent order-statistic intervals mismatch")
            wins = s["dna_lower_wins"] if row["direction"] == "dna_lower_recovery" else s["dna_higher_wins"]
            losses = s["effective_n"]-wins
            if abs(row["raw_p"]-independent_p(wins, losses)) > 1e-14:
                raise ValueError("independent confirmatory sign mismatch")
    adjusted = independent_holm([r["raw_p"] for r in tests])
    if not np.allclose(adjusted, [r["holm_p"] for r in tests], rtol=0, atol=1e-14):
        raise ValueError("independent Holm mismatch")
    accounting = independent_accounting(c.LOCAL_SD)
    epsilon = accounting["epsilon"]
    if abs(epsilon-10.) > 1e-10:
        raise ValueError("local accounting mismatch")
    c.write(OUT / "independent_statistics.json", dict(status="PASS", fixed_tests=72, tests=tests,
            qualification_jobs=qualification_count, confirmatory_jobs=len(observations), local_epsilon=epsilon))
    lines = ["# Priority34C — individual-gradient record recovery", "", "Verified complete; all scheduled outputs and fixed72 directional tests audited.", "",
        "Honest-but-curious server: client-side noise before upload, not central DP. Whole-training local epsilon10 uses per-client update-level replace-one adjacency over50 rounds, delta1e-5; NOT record DP.", "",
        "Batch1 is the easiest recovery setting. Public initial checkpoints/known labels and public population priors are assumed. Ratio uses full P32 FraudMLP/evalBN; native TabLeak uses P33B FC (IEEE60 selected features, BAF58, PaySim13). FedSGD single-gradient queries differ from multi-step Adam utility; epsilon transfer is conditional on the same clipped individual-query sensitivity, not a final-model inversion claim. Key-known v2 uses LSMR/sketch-space loss. V1 selection uses observable objectives only. Protected uploads exclude BN; private truth/noise bundles are audit-only, outside the DP release.", "",
        "Commands: `.venv-phase1/bin/python -B -u experiments/priority34c_qualify.py --launch`; development and confirmatory drivers `--prepare` then `--launch`; `experiments/analyze_priority34c.py --verify`. Resume skips only validated outputs after verified infrastructure interruption.", "", "## Qualification", "", "|Cell|n8|n24|Verdict|", "|---|---|---|---|"]
    first = json.loads((OUT / "gates_n8.json").read_text())
    for key in sorted(first):
        second = gates.get(key)
        lines.append(f"|{key}|{json.dumps(first[key]['tests'], sort_keys=True)}|{json.dumps(second['tests'], sort_keys=True) if second else 'not eligible'}|{'QUALIFIED' if second and second['passed'] else 'NOT_ASSESSABLE'}|")
    lines += ["", "## Paired confirmatory results (feature accuracy %, DNA−DP)", "", "|Dataset/cell|DNA|Comparator|DNA median|DP median|Median effect [ranks13,27]|Direction|raw p|Holm72 p|", "|---|---|---|---:|---:|---|---|---:|---:|"]
    for row in tests:
        s = row["summary"]
        values = (f"{s['dna_median']:.8g} {s['dna_order_interval']}|{s['dp_median']:.8g} {s['dp_order_interval']}|{s['paired_effect_median']:.8g} {s['paired_effect_order_interval']}" if s else "—|—|NOT_ASSESSABLE: "+row["reason"])
        lines.append(f"|{row['dataset']}/{row['cell']}|{row['defense']}|{row['comparator']}|{values}|{row['direction']}|{row['raw_p']:.10g}|{row['holm_p']:.10g}|")
    lines += ["", "## Distortion calibration", "", "```json", json.dumps(calibration, indent=2, sort_keys=True), "```", "",
        "Historical metadata preflight failures were preserved and amended before science. No outcome-based tuning, clamping, seed exclusion or central aggregate comparator. Failed qualification/matching is NOT_ASSESSABLE, not evidence of privacy. No optional secure-aggregate secondary arm was run.", "",
        "Independent reload, gates, accounting, paired sign/Holm72: PASS. SHA256 manifest includes original and additive freezes, receipts and attempt artifacts; earlier evidence is preserved."]
    report = ROOT / "reports/priority34c_report.md"
    contents = "\n".join(lines)+"\n"
    if report.exists() and report.read_text() != contents:
        raise RuntimeError("existing report differs; preserve it and write an amendment before correction")
    if not report.exists():
        report.write_text(contents)
    sources = {**q.verify_freeze()["sources"], **c.development.verify()["sources"], **doc["sources"]}
    sources[str(Path(__file__).resolve().relative_to(ROOT))] = c.sha(Path(__file__).resolve())
    for name in sources:
        if name.endswith(".py"):
            py_compile.compile(str(ROOT / name), doraise=True)
    subprocess.run(["git", "diff", "--check"], cwd=ROOT, check=True)
    if q.live_processes() or c.development.own_processes() or c.own_processes():
        raise RuntimeError("final no-live-worker audit failed")
    progress = dict(part="final_audit_pass", dataset="all", defense="all", comparator="fixed72",
                    done=len(observations), total=len(doc["jobs"]), failed=0, last_update=c.now(), eta_minutes=0.)
    c.write(OUT / "progress.json", progress)
    c.append(OUT / "progress.log", progress)
    checklist = json.loads((OUT / "checklist.json").read_text())
    for key in ("implementation_and_freeze", "qualification", "calibration_and_confirmatory", "report_and_final_audit"):
        checklist[key].update(status="completed", last_update=c.now())
    c.write(OUT / "checklist.json", checklist)
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.name not in ("sha256_manifest.csv", "COMPLETE.json")]
    frozen_inputs = {**q.verify_freeze()["inputs"], **c.development.verify()["inputs"], **doc["inputs"]}
    paths += [ROOT / name for name in sources] + [ROOT / name for name in frozen_inputs] + [report]
    with (OUT / "sha256_manifest.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "sha256"])
        for path in sorted(set(paths)):
            writer.writerow([str(path.relative_to(ROOT)), c.sha(path)])
    if q.live_processes() or c.development.own_processes() or c.own_processes():
        raise RuntimeError("final no-live-worker audit failed")
    c.write(OUT / "COMPLETE.json", dict(status="PASS", at=c.now(), report=str(report.relative_to(ROOT)),
          report_sha256=c.sha(report), manifest_sha256=c.sha(OUT / "sha256_manifest.csv"), fixed_tests=72,
          qualification_jobs=qualification_count, confirmatory_jobs=len(observations), no_live_workers=True,
          independent_audit="PASS", py_compile="PASS", git_diff_check="PASS"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true", required=True)
    parser.parse_args()
    run()
