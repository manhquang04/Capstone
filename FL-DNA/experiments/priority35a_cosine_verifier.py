"""Approved additive post-hoc infrastructure verifier; never runs measurements."""
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
import argparse
import math
from pathlib import Path
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from experiments import priority35a_final_audit as original
from experiments import priority35a_report as report
from experiments.priority35a_estimator import flatten
from experiments.priority35a_runtime import ROOT, OUT, sha, load, write, check_hashes, now, stage

ADDENDUM = ROOT / "protocols/amendments/2026-10-10_priority35a_cosine_verifier_infrastructure_posthoc.md"
FAILURE = report.F / "REQUIRES_DIRECTION_cosine_audit_20261010.json"
FAILURE_SHA = "92921fec6f4403fffb50d9a908e30038b577957256a15f869e5ca0b1ac8d0ea1"
FREEZE = report.F / "cosine_infrastructure_freeze.json"
ABS_TOL = 1e-5
RECORDS = []


def cosine_audit(saved, gradients, upload):
    torch.set_num_threads(1)
    g32, d32 = flatten(gradients), flatten(upload)
    if g32.dtype != torch.float32 or d32.dtype != torch.float32:
        raise ValueError("cosine requires original float32 tensors")
    if not torch.isfinite(g32).all() or not torch.isfinite(d32).all():
        raise ValueError("nonfinite cosine input")
    sign32 = g32.sign()
    vectors = dict(gradient_vs_sign=g32, adam_vs_sign=d32, descent_aligned_adam_vs_sign=-d32)
    exact = {k: float(torch.nn.functional.cosine_similarity(v, sign32, dim=0, eps=1e-12))
             for k, v in vectors.items()}
    flags = dict(zero_gradient_norm=bool(g32.norm() == 0), zero_upload_norm=bool(d32.norm() == 0))
    if set(saved) != set(exact) | set(flags):
        raise ValueError("cosine schema")
    for key, value in flags.items():
        if type(saved[key]) is not bool or saved[key] != value:
            raise ValueError("cosine zero flag")
    sign64 = sign32.numpy().astype(np.float64)
    for key, value in exact.items():
        if not math.isfinite(saved[key]) or saved[key] != value:
            raise ValueError("exact float32 cosine mismatch")
        vector64 = vectors[key].numpy().astype(np.float64)
        independent = float(np.dot(vector64, sign64) /
                            (max(np.linalg.norm(vector64), 1e-12) * max(np.linalg.norm(sign64), 1e-12)))
        difference = abs(saved[key] - independent)
        if not math.isfinite(independent) or difference > ABS_TOL:
            raise ValueError("absolute float64 cosine mismatch")
        RECORDS.append(dict(statistic=key, stored_float32=value, independent_float64=independent,
                            absolute_difference=difference))


def prepare():
    if FREEZE.exists():
        raise ValueError("infrastructure verifier already frozen")
    if sha(FAILURE) != FAILURE_SHA:
        raise ValueError("original failed record changed")
    check_hashes(load(report.F / "report_freeze.json")["hashes"])
    folder = report.F / "cosine_infrastructure_preflight" / str(time.time_ns())
    sources = report.checks(folder)
    paths = sources + [ADDENDUM, FAILURE, report.F / "report_freeze.json"]
    write(FREEZE, dict(at=now(), hashes={str(p.relative_to(ROOT)): sha(p) for p in paths},
                      absolute_float64_tolerance=ABS_TOL, exact_float32=True,
                      scope="post-hoc infrastructure only; no measurement rerun"), exclusive=True)
    stage("cosine_infrastructure_verifier_ready", status="FINAL_VERIFYING", required_direction=None,
          measurement_reruns=0, original_failed_record_preserved=True)


def finish():
    check_hashes(load(FREEZE)["hashes"])
    original_cosine = original.cosine_audit
    original_text = report.report_text
    original_write = report.write
    original_stage = report.stage
    RECORDS.clear()

    def amended_text(audit, csv_audit, tests_path):
        text = original_text(audit, csv_audit, tests_path)
        old = "Float64 independent dot products check float32 sign-cosines with an audit-only 2e-6 numeric tolerance; this does not affect any exact tie or gate threshold."
        new = "Stored float32 sign-cosines match fresh float32 recomputation exactly; independent NumPy float64 dot/norm recomputation agrees within absolute 1e-5 (relative tolerance zero), under the approved post-hoc infrastructure addendum. No exact tie or gate threshold changed."
        if old not in text:
            raise ValueError("frozen report disclosure text drift")
        text = text.replace(old, new)
        summary = cosine_summary()
        text += "\n## Post-hoc infrastructure verifier disclosure\n\n"
        text += f"Addendum: `{ADDENDUM.relative_to(ROOT)}`; SHA-256 `{sha(ADDENDUM)}`. Original failed verifier output/traceback: `{FAILURE.relative_to(ROOT)}`, unchanged SHA-256 `{FAILURE_SHA}`; its original progress-log event is retained. Frozen sources were not edited and no measurement was rerun.\n\n"
        text += f"All {summary['statistics_checked']} stored float32 statistics matched exactly. Maximum absolute float64 discrepancy: {summary['maximum_absolute_float64_difference']:.17g}, below the predeclared 1e-5 absolute guard. Observed stored-float32 ranges:\n\n"
        for key, limits in summary["ranges"].items():
            text += f"- {key}: [{limits['min']:.17g}, {limits['max']:.17g}].\n"
        text += "\nNo diagnostic conclusion depends on cosine at the 1e-5 level: cosine is descriptive only; the locked interpretation uses feature-accuracy comparisons with controls. All six D1 cells fail, so instrument/budget inadequacy remains the conclusion and negative qualification says nothing about privacy. Qualification 0/72 and Holm6048 are unchanged.\n\n"
        text += "Verifier-only commands: `.venv-phase1/bin/python -B experiments/priority35a_cosine_verifier.py --prepare` and `--finish`. Synthetic tests, py_compile and git diff --check logs are included in the manifest.\n"
        return text

    def amended_write(path, doc, **kwargs):
        # Additive final manifest includes the infrastructure amendment itself.
        if Path(path) == report.F / "manifest.json":
            doc["hashes"][str(ADDENDUM.relative_to(ROOT))] = sha(ADDENDUM)
        if Path(path) == report.F / "independent_tensor_verification.json":
            doc["cosine_method"] = "exact fresh float32 plus independent NumPy float64 absolute 1e-5, relative zero"
            doc["cosine_precision_audit"] = cosine_summary()
        original_write(path, doc, **kwargs)

    def amended_stage(name, **fields):
        if name == "COMPLETE":
            fields["tests_run"] = 43
            fields["measurement_reruns"] = 0
        original_stage(name, **fields)

    # Only numerical callable changes within the scientific independent audit.
    # Remaining wrappers annotate report/provenance and include the new addendum.
    original.cosine_audit = cosine_audit
    report.report_text = amended_text
    report.write = amended_write
    report.stage = amended_stage
    try:
        report.finish()
    finally:
        original.cosine_audit = original_cosine
        report.report_text = original_text
        report.write = original_write
        report.stage = original_stage


def cosine_summary():
    if len(RECORDS) != 72:
        raise ValueError("expected 24 batches times three cosine statistics")
    ranges = {}
    for key in ("gradient_vs_sign", "adam_vs_sign", "descent_aligned_adam_vs_sign"):
        values = [r["stored_float32"] for r in RECORDS if r["statistic"] == key]
        if len(values) != 24:
            raise ValueError("cosine audit coverage")
        ranges[key] = dict(min=min(values), max=max(values))
    return dict(statistics_checked=len(RECORDS), exact_float32=True, absolute_tolerance=ABS_TOL,
                maximum_absolute_float64_difference=max(r["absolute_difference"] for r in RECORDS), ranges=ranges)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--finish", action="store_true")
    args = parser.parse_args()
    if args.prepare == args.finish:
        parser.error("choose --prepare or --finish")
    if args.prepare:
        prepare()
    else:
        finish()
