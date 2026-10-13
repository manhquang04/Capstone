"""Final negative-result report and independently checked machine-readable evidence."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import csv
import itertools
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from scipy.stats import binomtest
from experiments.priority35a_runtime import OUT, Q, PROTOCOL, PROTOCOL_SHA, sha, load, write, stage, check_hashes, now
from experiments.priority35a_diagnostic import D, ADDENDUM, selected_cells, no_live_audit, frozen
from experiments.priority35a_qualification import cell_id, cells
from experiments.priority35a_final_audit import verify_measurements

F = OUT / "final"
METRICS = ("feature_accuracy", "standardized_mse", "bounded_score")
ARMS = ("estimator", "mean_mode", "empirical_marginal", "decoy")
CONTROLS = ARMS[1:]


def interpretation(d1, d2):
    if not d1:
        return "INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy"
    if not d2:
        return "ADAM_STEP_LIMITATION under fixed diagnostic; not causal proof"
    return "UNKNOWN_LOCAL_BN_LIMITATION under fixed diagnostic; not causal proof"


def holm_numpy(probabilities):
    p = np.array(probabilities, dtype=np.float64)
    if p.ndim != 1 or not len(p) or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid fixed-family probabilities")
    order = np.argsort(p, kind="stable")
    adjusted = np.minimum(1., np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result


def exact_summary(differences):
    v = np.array(differences, dtype=np.float64)
    if v.shape != (8,) or not np.isfinite(v).all():
        raise ValueError("incomplete/nonfinite CSV units")
    w, l = int((v > 0).sum()), int((v < 0).sum())
    return dict(n=8, wins=w, losses=l, ties=8-w-l,
                p_one_sided=float(binomtest(w, w+l, alternative="greater").pvalue) if w+l else 1.,
                median_difference=float(np.median(v)))


def csv_write(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise ValueError("never overwrite final CSV")
    with path.open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def export_csvs():
    table = load(Q / "qualification_table.json")
    diagnostics = load(D / "summary.json")["summaries"]
    rows, summaries = [], []
    for entry in table["cells"]:
        identifier = cell_id(entry["cell"])
        saved = load(Q / "per_target" / identifier / "n8.json")
        for row in saved["rows"]:
            for arm in ARMS:
                rows.append(dict(cell=identifier, stage="n8", index=row["index"], arm=arm,
                                 source_ids=" ".join(map(str, row["source_ids"])), **{k: row[arm][k] for k in METRICS}))
        for control, test in entry["n8"]["tests"].items():
            summaries.append(dict(cell=identifier, stage="n8", control=control, **test))
    csv_write(F / "qualification_per_target.csv", rows, ["cell", "stage", "index", "arm", "source_ids", *METRICS])
    csv_write(F / "qualification_tests.csv", summaries, ["cell", "stage", "control", "n", "wins", "losses", "ties", "p_one_sided", "median_difference"])
    csv_write(F / "qualification_table.csv", [dict(cell=cell_id(e["cell"]), n8_passed=False, n24="NOT_RUN", reason=e["n24_not_run_reason"], qualified=False) for e in table["cells"]],
              ["cell", "n8_passed", "n24", "reason", "qualified"])
    rows, summaries, cosine_rows = [], [], []
    for saved in diagnostics:
        identifier = cell_id(saved["cell"])
        condition = saved["condition"]
        for row in saved["rows"]:
            for arm in ARMS:
                rows.append(dict(cell=identifier, condition=condition, index=row["index"], arm=arm,
                                 source_ids=" ".join(map(str, row["source_ids"])), **{k: row[arm][k] for k in METRICS}))
        for control, test in saved["comparisons"]["tests"].items():
            summaries.append(dict(cell=identifier, condition=condition, control=control, **test))
        for i, cos in enumerate(saved["cosines"]):
            cosine_rows.append(dict(cell=identifier, condition=condition, index=i,
                                    observed_vs_sign=cos["gradient_vs_sign" if condition == "D1" else "adam_vs_sign"], **cos))
    csv_write(F / "diagnostic_per_target.csv", rows, ["cell", "condition", "index", "arm", "source_ids", *METRICS])
    csv_write(F / "diagnostic_tests.csv", summaries, ["cell", "condition", "control", "n", "wins", "losses", "ties", "p_one_sided", "median_difference"])
    csv_write(F / "diagnostic_cosines.csv", cosine_rows, ["cell", "condition", "index", "observed_vs_sign", "gradient_vs_sign", "adam_vs_sign", "descent_aligned_adam_vs_sign", "zero_gradient_norm", "zero_upload_norm"])
    reservation = []
    for cell, transform, comparator, metric, direction in itertools.product(cells(), ("v1", "v2"),
            ("none", "dp_eps1_global", "dp_eps1_per_tensor", "dp_eps3_global", "dp_eps3_per_tensor", "dp_eps10_global", "dp_eps10_per_tensor"), METRICS, ("positive", "negative")):
        reservation.append(dict(cell=cell_id(cell), transform=transform, comparator=comparator, metric=metric, direction=direction,
                                n=0, p_raw=1., p_holm=1., reject=False, status="NOT_APPLICABLE_UNQUALIFIED"))
    csv_write(F / "holm6048.csv", reservation, ["cell", "transform", "comparator", "metric", "direction", "n", "p_raw", "p_holm", "reject", "status"])
    csv_write(F / "confirmatory_per_target.csv", [], ["cell", "index", "target_id", "arm", *METRICS])
    write(F / "confirmatory_status.json", dict(status="NOT_APPLICABLE", reason="0/72 cells qualified; no measurements allowed", measured_targets=0,
          order_statistic_interval="NOT_APPLICABLE: no n39 pairs; ranks 13/27 unchanged", family=6048), exclusive=True)


def read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def csv_check():
    """Every exact test recalculated from exported per-target rows, not saved p."""
    tested = 0
    for kind, keys, expected_rows in (("qualification", ("cell", "stage"), 2304), ("diagnostic", ("cell", "condition"), 576)):
        rows = read_csv(F / (kind+"_per_target.csv"))
        if len(rows) != expected_rows:
            raise ValueError("CSV target count")
        groups = {}
        for row in rows:
            identity = tuple(row[k] for k in keys)
            index = int(row["index"])
            if row["arm"] not in ARMS or index not in range(8):
                raise ValueError("CSV unit/arm")
            metrics = {k: float(row[k]) for k in METRICS}
            if any(not math.isfinite(v) for v in metrics.values()):
                raise ValueError("nonfinite CSV")
            key = (index, row["arm"])
            group = groups.setdefault(identity, {})
            if key in group:
                raise ValueError("duplicate CSV unit")
            group[key] = (metrics, row["source_ids"])
        tests = read_csv(F / (kind+"_tests.csv"))
        if len(tests) != len(groups)*3:
            raise ValueError("CSV test completeness")
        test_keys = set()
        for test in tests:
            identity = tuple(test[k] for k in keys)
            control = test["control"]
            if (identity, control) in test_keys or control not in CONTROLS:
                raise ValueError("CSV test identity")
            test_keys.add((identity, control))
            group = groups[identity]
            if set(group) != set(itertools.product(range(8), ARMS)):
                raise ValueError("missing CSV pair")
            delta = []
            for i in range(8):
                if len({group[(i, arm)][1] for arm in ARMS}) != 1:
                    raise ValueError("unpaired CSV source IDs")
                delta.append(group[(i, "estimator")][0]["feature_accuracy"]-group[(i, control)][0]["feature_accuracy"])
            expected = exact_summary(delta)
            if any(float(test[k]) != v for k, v in expected.items()):
                raise ValueError("independent CSV sign/effect mismatch")
            tested += 1
        # Bind every CSV numeric value to independently reloaded tensor evidence.
        for identity, group in groups.items():
            identifier, which = identity
            saved = load(Q / "per_target" / identifier / "n8.json") if kind == "qualification" else load(D / "per_target" / which / (identifier+".json"))
            for i, row in enumerate(saved["rows"]):
                for arm in ARMS:
                    expected = {k: row[arm][k] for k in METRICS}
                    if group[(i, arm)] != (expected, " ".join(map(str, row["source_ids"]))):
                        raise ValueError("CSV-to-tensor evidence binding")
    family = read_csv(F / "holm6048.csv")
    expected_keys = set(itertools.product([cell_id(c) for c in cells()], ("v1", "v2"),
        ("none", "dp_eps1_global", "dp_eps1_per_tensor", "dp_eps3_global", "dp_eps3_per_tensor", "dp_eps10_global", "dp_eps10_per_tensor"), METRICS, ("positive", "negative")))
    actual_keys = [(r["cell"], r["transform"], r["comparator"], r["metric"], r["direction"]) for r in family]
    if len(family) != 6048 or len(set(actual_keys)) != 6048 or set(actual_keys) != expected_keys:
        raise ValueError("fixed Holm family reservations")
    adjusted = holm_numpy([float(r["p_raw"]) for r in family])
    if any(float(r["p_raw"]) != 1 or float(r["p_holm"]) != adjusted[i] or r["reject"] != "False" or r["status"] != "NOT_APPLICABLE_UNQUALIFIED" or r["n"] != "0" for i, r in enumerate(family)):
        raise ValueError("Holm NA cannot imply measurement")
    if read_csv(F / "confirmatory_per_target.csv"):
        raise ValueError("ineligible confirmatory rows")
    qtable = read_csv(F / "qualification_table.csv")
    if len(qtable) != 72 or [r["cell"] for r in qtable] != [cell_id(c) for c in cells()] or any(r["qualified"] != "False" or r["n8_passed"] != "False" or r["n24"] != "NOT_RUN" or not r["reason"] for r in qtable):
        raise ValueError("qualification CSV decision")
    cosrows = read_csv(F / "diagnostic_cosines.csv")
    if len(cosrows) != 144:
        raise ValueError("cosine CSV completeness")
    seen = set()
    for row in cosrows:
        key = (row["cell"], row["condition"], int(row["index"]))
        if key in seen:
            raise ValueError("duplicate cosine CSV")
        seen.add(key)
        cos = load(D / "per_target" / row["condition"] / (row["cell"]+".json"))["cosines"][int(row["index"])]
        if any(row[k] != str(v) for k, v in cos.items()):
            raise ValueError("cosine CSV binding")
        expected = cos["gradient_vs_sign" if row["condition"] == "D1" else "adam_vs_sign"]
        if float(row["observed_vs_sign"]) != expected:
            raise ValueError("observed operator cosine sign")
    return dict(status="PASS", exact_tests_recomputed=tested, qualification_tests=216, diagnostic_descriptive_tests=54,
                complete_qualification_csv_rows=2304, complete_diagnostic_csv_rows=576, paired_n=8,
                fixed_holm_reservations=6048, confirmatory_rows=0, qualification_decisions_unchanged=True)


def report_text(audit, csv_audit, tests_path):
    table = load(Q / "qualification_table.json")
    diag = load(D / "summary.json")["summaries"]
    lines = ["# Priority35A: Config-C record recovery evaluation", "", "Date: 2026-10-10. Final negative qualification result with a separately authorized post-hoc descriptive diagnostic.", "",
        "## Outcome", "", "**0/72 cells qualified.** All failed at n8; fresh n24 and n39 were NOT_RUN. No confirmatory, v1/v2 or DP measurements were made.",
        "No qualified instrument recovered records from realistic Adam uploads even without a defense. This is a result about the tested instruments and budgets, not evidence of privacy, immunity, or transform superiority over DP.", "",
        "## Design, knowledge and metrics", "",
        "Config-C is P34A FedBN: K=3, Adam lr .001, binary focal alpha .95/gamma 2; only eight non-BN trainable tensor differences upload. BAF (primary), IEEE-CIS, PaySim; seed 321000 rounds 10/30. All three round-50 replays are bit-exact against historical P34A. Client BN affine/statistics and dropout masks are unavailable in deployable Config-C. BN stays train mode; the qualification surrogate optimizes free default BN affine and replaces only dropout with identity. Labels given means unordered multiset; labels estimated enumerates every fraud count with 64 steps then two 4,000-step starts. TabLeak-style uses cosine plus .1 normalized L2; Wan update matching uses cosine only, both simulating fresh Adam exactly in float32.",
        "b=1 is NOT_APPLICABLE: training-mode BN cannot process it. A no-BN synthetic ratio test is a correctness check only. No eval-BN or oracle-BN substitution occurred in qualification.",
        "The experimental unit is one batch (b=4/8/16), not a row or coordinate. Qualification compares paired feature accuracies to mean/mode, empirical marginal and cyclic next-target decoy; exact one-sided sign p<.05 must hold for all three, at n8 and fresh n24. Ties use exact equality and are excluded only from the binomial count. No failed/missing/numerical targets were discarded. All 144 distinct original batches passed the P24–34 original-source-ID firewall.",
        "Accuracy uses native TabLeak blind Hungarian alignment, .319 training SD (ddof1) continuous tolerance and exact category match. Its absent-feature-type denominator convention is retained. Standardized MSE uses the same alignment; zero SD fallback 1 applies only to MSE, while accuracy tolerance stays zero. Bounded score is min(MSE, mean/mode MSE). Frequency-encoded categories are evaluated in encoded coordinates, not recovered original categories. Mean/mode and empirical marginal are training-derived, update-free references, not literally distribution-free.", "",
        "## Qualification table", "", "W/L/T below is estimator minus reference accuracy; p is one-sided greater. Each entry has n=8 batches. All n24 entries are NOT_RUN because n8 failed at least one locked control.", "",
        "| Cell | Mean/mode W/L/T; p | Marginal W/L/T; p | Decoy W/L/T; p | n24 | Decision |",
        "|---|---|---|---|---|---|"]
    def test_label(test):
        return f"{test['wins']}/{test['losses']}/{test['ties']}; {test['p_one_sided']:.8g}"
    for r in table["cells"]:
        t = r["n8"]["tests"]
        lines.append(f"| {cell_id(r['cell'])} | {test_label(t['mean_mode'])} | {test_label(t['empirical_marginal'])} | {test_label(t['decoy'])} | NOT_RUN | NOT_QUALIFIED |")
    lines += ["", "### Closest-to-qualifying evidence", "", "Ranked by the worst of the three gate p values, then cell ID; this ordering is descriptive, not a retest.", ""]
    for r in sorted(table["cells"], key=lambda r: (r["n8"]["worst_p"], cell_id(r["cell"])))[:5]:
        lines.append("- "+cell_id(r["cell"])+": "+"; ".join(f"{c} W/L/T {test_label(t)}, median accuracy delta {t['median_difference']:.9g}" for c, t in r["n8"]["tests"].items()))
    lines += ["", "## Post-hoc descriptive diagnostic (not qualification)", "",
        "The dated addendum was written and frozen before D1/D2. Six cells use the exact same eight n8 batches, b=4, round10, labels given and the same public initialization seeds, two starts and 4,000 iterations/start. D1 observes plain non-BN gradients with true local BN supplied; D2 observes the unchanged Adam upload with true local BN supplied. True BN affine is fixed, buffers supplied but BN remains train mode. D3 reuses qualification without new runs. Both oracle conditions remain dropout-identity surrogates and receive no private dropout seed/masks or true rows/label alignment. Original stochastic dropout is retained in the evaluator's actual gradient/Adam capture; that Adam step reproduces existing uploads bit for bit.",
        "The controls are unchanged; decoy is cyclic next-target reconstruction within the same condition and cell. 'Beats all controls' retains the original n8 three-sign-test intersection, but only as unadjusted descriptive classification outside Holm and without changing eligibility. These 24 paired batches are reused across conditions/instruments, not 144 independent batches.", "",
        "| Cell | Condition | Median accuracy | Mean/mode W/L/T; median delta | Marginal W/L/T; median delta | Decoy W/L/T; median delta | Beats all (descriptive) |",
        "|---|---|---:|---|---|---|---|"]
    for d in diag:
        t = d["comparisons"]["tests"]
        fields = [f"{t[c]['wins']}/{t[c]['losses']}/{t[c]['ties']}; {t[c]['median_difference']:.9g}" for c in CONTROLS]
        lines.append(f"| {cell_id(d['cell'])} | {d['condition']} | {d['medians']['estimator']['feature_accuracy']:.9g} | "+" | ".join(fields)+f" | {d['diagnostic_beats_all_controls']} |")
    lines += ["", "### Interpretation under the pre-run diagnostic rules", ""]
    for cell in selected_cells():
        group = {d["condition"]: d for d in diag if d["cell"] == cell}
        lines.append("- "+cell_id(cell)+": "+interpretation(group["D1"]["diagnostic_beats_all_controls"], group["D2"]["diagnostic_beats_all_controls"])+".")
    lines += ["", "If D1 fails, the fixed estimators/budgets are inadequate for this model under the declared approximations; the negative qualification result says nothing about privacy. A positive D1 and failed D2 are a pattern consistent with an Adam-step limitation; positive D1/D2 with failed D3 are consistent with unknown-BN limitations. These diagnostic attributions are conditional, not causal proof; shared dropout mismatch, small post-hoc-selected samples and optimizer sensitivity remain. No diagnostic result qualifies a cell or authorizes confirmatory measurements.", "",
        "### Upload sign-likeness", "",
        "Adam upload is theta_after minus theta_before: cosine with +sign(g) is generally negative because descent is in the opposite direction. The additional descent-aligned cosine uses -upload versus sign(g). No absolute-value substitution. D1 observed gradient uses +g. Cosines are per actual batch, not model-coordinate replication.", "",
        "| Dataset | Median gradient vs sign(g) | Median Adam upload vs sign(g) | Median -Adam upload vs sign(g) | Zero gradient/upload norms |", "|---|---:|---:|---:|---|"]
    for dataset in ("baf", "ieee_cis", "paysim"):
        cos = next(d["cosines"] for d in diag if d["cell"]["dataset"] == dataset)
        lines.append(f"| {dataset} | {np.median([c['gradient_vs_sign'] for c in cos]):.9g} | {np.median([c['adam_vs_sign'] for c in cos]):.9g} | {np.median([c['descent_aligned_adam_vs_sign'] for c in cos]):.9g} | {sum(c['zero_gradient_norm'] for c in cos)}/{sum(c['zero_upload_norm'] for c in cos)} |")
    lines += ["", "## Confirmatory table and fixed Holm family", "",
        "| Eligible cells | Measured arms/targets | Intervals ranks 13/27 | Transform-versus-DP conclusion |", "|---:|---|---|---|", "| 0 | None; empty declared CSV schema | NOT_APPLICABLE; no n39 pairs | NOT_APPLICABLE |",
        "All 6,048 original reservations remain: 72 cells × 2 transforms × 7 comparators × 3 metrics × 2 directions. Each is p=1 with NOT_APPLICABLE_UNQUALIFIED, n=0, Holm-adjusted p=1, reject=false. These values reserve the family, not measured p values or evidence of equality. No DP, v1 or v2 arm is measured. The proposed P34B comparator was joint C=.01/per-tensor, replace-one sensitivity 2C, 50 releases, delta1e-5, epsilon1/3/10; no calibrated sigma or empirical DP outcome is claimed here. That mechanism is contribution/transcript-level, not record-level DP. No transform privacy theorem follows.", "",
        "## Independent verification and provenance", "",
        f"Independent tensor audit: {audit['status']}; 576 qualification and 96 new diagnostic reconstructions, 48 reused D3, 144 original uploads. Source firewall, exact checkpoint/upload replay, known/private knowledge separation, BN finite/variance, budget and observable-selection checks PASS. Separate vectorized native TabLeak cost/SciPy Hungarian reload every metric; SciPy exact binomial tests independently agree with production integer-combinatorial tests.",
        f"CSV readback: {csv_audit['status']}; all {csv_audit['exact_tests_recomputed']} tests recomputed (216 qualification, 54 descriptive), all paired IDs/numerics and medians bound to reloaded evidence; NumPy Holm recomputed over all6048. No missing/nonfinite CSV outputs. Float64 independent dot products check float32 sign-cosines with an audit-only 2e-6 numeric tolerance; this does not affect any exact tie or gate threshold.",
        f"Main amendment SHA-256: `{PROTOCOL_SHA}`.", f"Post-hoc addendum SHA-256: `{sha(ADDENDUM)}`.",
        f"Qualification freeze SHA-256: `{sha(Q / 'execution_freeze.json')}`.", f"Diagnostic freeze SHA-256: `{sha(D / 'freeze.json')}`.",
        f"Qualification scientific manifest SHA-256: `{sha(Q / 'final_manifest.json')}` (original unchanged).", "Final manifest and COMPLETE receipt carry the final report digest; no self-referential report hash is embedded in this document.",
        "Earlier post-exit administrative seals are preserved snapshots. Their old progress/checklist hashes are not commitments to later authorized administrative updates. Bytecode caches are excluded as reproducible intermediates; scientific outputs and closed detached logs are hashed. No Latex/, external_defenses/, dataset, earlier artifact/report, commit or push was changed.", "",
        "## Exact commands", "", "Run from FL-DNA/ with .venv-phase1, CPU and torch/BLAS thread1. All launch/preflight receipts retain exact paths and stdout/stderr.", "", "```text",
        ".venv-phase1/bin/python -B experiments/priority35a_qualification.py --preflight",
        ".venv-phase1/bin/python -B experiments/priority35a_qualification.py --prepare",
        ".venv-phase1/bin/python -B experiments/priority35a_qualification.py --launch",
        ".venv-phase1/bin/python -B experiments/priority35a_postexit.py",
        ".venv-phase1/bin/python -B experiments/priority35a_diagnostic.py --prepare",
        ".venv-phase1/bin/python -B experiments/priority35a_diagnostic.py --launch",
        ".venv-phase1/bin/python -B experiments/priority35a_report.py --freeze",
        ".venv-phase1/bin/python -B experiments/priority35a_report.py --finish",
        "```", "",
        f"Final tests/py_compile/git diff --check commands, return codes and logs: `{tests_path.relative_to(ROOT)}`. Python {sys.version.split()[0]}, torch {torch.__version__}, NumPy {np.__version__}. CPU only, no GPU workloads.", "",
        "## Limitations", "",
        "This is a negative instrument qualification result, not a guarantee that records are private. The oracle-BN diagnostic is privileged and post hoc, limited to b4/r10/labels given/n8. Dropout masks remain unknown and its fixed identity approximation is not an expectation of the complete nonlinear network/update. Labels estimated is focal-loss count enumeration, not a universal sign-based label theorem. Adam arithmetic is reproduced exactly but its near-sign update and rounded weak derivatives can limit reconstruction optimization. True BN affine was fixed only in privileged D1/D2, never deployable qualification. Population-derived references, encoded-category ambiguity, native absent-type normalization, natural fraud prevalence, two-start finite budget, shared clients/checkpoints and finite-population sampling all bound interpretation. A failure to beat references is not a zero-leakage test. Conditional n24/n39 were not reached, so there is no adjusted comparison of DNA transforms versus DP and no record-level DP certification.", "",
        "Sources: [TabLeak, Vero et al. (2023)](https://proceedings.mlr.press/v202/vero23a.html), [Wan et al., optimizer-update matching (2026)](https://proceedings.mlr.press/v306/wan26c.html). These are model-specific adaptations, not reproductions of every official-paper configuration.", ""]
    return "\n".join(lines)


def checks(folder):
    sources = sorted([*(ROOT / "experiments").glob("priority35a_*.py"), *(ROOT / "tests").glob("test_priority35a*.py")])
    commands = [[sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_priority35a*.py", "-v"],
                [sys.executable, "-m", "py_compile", *[str(p.relative_to(ROOT)) for p in sources]], ["git", "diff", "--check"]]
    for i, command in enumerate(commands):
        result = subprocess.run(command, cwd=ROOT, env=dict(os.environ, PYTHONPYCACHEPREFIX=str(OUT / "final_bytecode")), capture_output=True, text=True)
        write(folder / f"command_{i}.json", dict(command=command, returncode=result.returncode, stdout=result.stdout, stderr=result.stderr), exclusive=True)
        if result.returncode:
            raise ValueError("final verification command failed; preserved logs")
    for p in sources:
        if not p.read_text().endswith("\n") or any(line.rstrip(" \t") != line for line in p.read_text().splitlines()):
            raise ValueError("new untracked source whitespace failure")
    return sources


def freeze_report():
    if (F / "report_freeze.json").exists():
        raise ValueError("report audit already frozen")
    folder = F / "preflight" / str(time.time_ns())
    sources = checks(folder)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in sources}
    hashes[str(ADDENDUM.relative_to(ROOT))] = sha(ADDENDUM)
    hashes[str(PROTOCOL.relative_to(ROOT))] = PROTOCOL_SHA
    write(F / "report_freeze.json", dict(at=now(), hashes=hashes, fixed_family=6048, diagnostic_descriptive=True,
          interpretation="locked addendum; no eligibility or Holm change", preflight=str(folder.relative_to(ROOT))), exclusive=True)
    # Report freeze may occur while scientific workers run; do not replace their progress stage.
    checklist = load(OUT / "checklist.json")
    checklist.update(final_verifier_report_sources_frozen=True)
    write(OUT / "checklist.json", checklist)


def finish():
    no_live = no_live_audit()
    if (OUT / "COMPLETE.json").exists() or (ROOT / "reports/priority35a_report.md").exists():
        raise ValueError("final output already exists; never overwrite evidence")
    check_hashes(load(F / "report_freeze.json")["hashes"])
    frozen()
    stage("final_independent_reload", status="FINAL_VERIFYING", no_live_scientific_workers=True)
    audit = verify_measurements()
    write(F / "independent_tensor_verification.json", dict(at=now(), **audit), exclusive=True)
    stage("final_csv_export_and_recomputation", status="FINAL_VERIFYING")
    export_csvs()
    csv_audit = csv_check()
    write(F / "independent_csv_verification.json", dict(at=now(), **csv_audit), exclusive=True)
    checks_path = F / "verification_commands" / str(time.time_ns())
    checks(checks_path)
    report = ROOT / "reports/priority35a_report.md"
    report.parent.mkdir(parents=True, exist_ok=True)
    # Generated artifact, not an edit to an earlier report. Exclusive and never replayed.
    with report.open("x") as handle:
        handle.write(report_text(audit, csv_audit, checks_path))
    no_live_audit()
    paths = [p for p in OUT.rglob("*") if p.is_file() and p.suffix in (".json", ".pt", ".md", ".csv", ".log")
             and p.name not in ("progress.json", "progress.log", "checklist.json", "COMPLETE.json")]
    paths += [report, ADDENDUM, PROTOCOL]
    paths += list((ROOT / "experiments").glob("priority35a_*.py")) + list((ROOT / "tests").glob("test_priority35a*.py"))
    manifest_path = F / "manifest.json"
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(manifest_path, dict(at=now(), hashes=hashes,
          excluded=["bytecode caches", "mutable progress.json/progress.log/checklist.json", "manifest self and COMPLETE receipt"],
          old_administrative_seals="preserved historical snapshots, not current mutable-file assertions"), exclusive=True)
    check_hashes(hashes)
    checklist = load(OUT / "checklist.json")
    checklist.update(complete=True, full_evaluation_complete=True, diagnostic_complete=True,
                     diagnostic_independent_verification_pass=True, confirmatory_complete_or_preregistered_not_applicable=True,
                     holm_family_verified=True, independent_csv_verification_pass=True, report_manifest_verified=True,
                     no_live_scientific_workers=True, stage="COMPLETE")
    write(OUT / "checklist.json", checklist)
    stage("COMPLETE", status="COMPLETE", complete=True, full_priority35a_complete=True,
          diagnostic_jobs_done=96, diagnostic_complete=True, tests_run=40, pending=[],
          no_live_scientific_workers=True, required_direction=None, report_path=str(report.relative_to(ROOT)))
    # stage() defaults checklist.complete false; terminal administrative correction is explicit.
    checklist = load(OUT / "checklist.json")
    checklist.update(complete=True, full_evaluation_complete=True)
    write(OUT / "checklist.json", checklist)
    write(OUT / "COMPLETE.json", dict(at=now(), status="VERIFIED_COMPLETE", report_sha256=sha(report),
          manifest_path=str(manifest_path.relative_to(ROOT)), manifest_sha256=sha(manifest_path),
          independent_tensor_pass=True, independent_csv_pass=True, fixed_holm_family=6048, qualified_cells=0,
          diagnostic_jobs=96, no_live_scientific_workers=no_live["no_live_scientific_workers"]), exclusive=True)
    terminal = [OUT / p for p in ("progress.json", "progress.log", "checklist.json", "COMPLETE.json")]
    write(F / "administrative_terminal_seal.json", dict(at=now(), hashes={str(p.relative_to(ROOT)): sha(p) for p in terminal},
          scientific_manifest_sha256=sha(manifest_path), report_sha256=sha(report)), exclusive=True)
    print(dict(status="VERIFIED_COMPLETE", report_sha256=sha(report), manifest_sha256=sha(manifest_path)), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    if args.freeze == args.finish:
        parser.error("choose --freeze or --finish")
    if args.freeze:
        freeze_report()
    else:
        finish()
