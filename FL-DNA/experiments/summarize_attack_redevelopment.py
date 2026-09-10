"""Summarize the bounded attack redevelopment and its scale check."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from experiments.run_phase3_full_client import dump


def _load(path: Path):
    return json.loads(path.read_text())


def _comparison_row(name, comparison):
    return {
        "tier": name,
        "prior_wins": comparison["prior"]["wins"],
        "prior_n": comparison["prior"]["n"],
        "prior_mean_delta": comparison["prior"]["mean_difference"],
        "prior_median_delta": comparison["prior"]["median_difference"],
        "prior_sign_p": comparison["prior"]["one_sided_sign_p"],
        "prior_gate": comparison["prior"]["gate"],
        "zero_wins": comparison["zero"]["wins"],
        "zero_n": comparison["zero"]["n"],
        "zero_mean_delta": comparison["zero"]["mean_difference"],
        "zero_median_delta": comparison["zero"]["median_difference"],
        "zero_sign_p": comparison["zero"]["one_sided_sign_p"],
        "zero_gate": comparison["zero"]["gate"],
    }


def _mean_selected(selected, field):
    return float(np.mean([row[field] for row in selected]))


def _write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_bool(value):
    return "PASS" if value else "FAIL"


def _write_markdown(path, summary):
    one = summary["one_batch"]
    dna = summary["dna_transform"]
    scale = summary["scale"]
    text = f"""# Phase 3 Attack Redevelopment Report

This report summarizes the follow-up attack development after the earlier full-client Adam attempts failed to beat controls reliably. The scope is deliberately bounded: the validated attack uses a four-record, one-step Adam local update with known labels, known record order, known local RNG, known model architecture, known optimizer, and the pre-local checkpoint. It is not a full FedAvg client-update attack.

## Protocol

- Attack input parameterization: PaySim manifold with six optimized base numeric fields, two derived balance-difference fields, and softmax transaction-type logits.
- Objective: balanced per-tensor update matching with BatchNorm running buffers included at weight 1.0.
- Candidate selection: minimum attacker objective within each group/restart family; ground truth is used only for evaluation metrics.
- Confirmation gate: baseline must beat both prior and zero-update controls with negative mean delta, negative median delta, and one-sided exact sign p < 0.05.
- Frozen attack: `{one["frozen_attack"]["variant"]}`, lr={one["frozen_attack"]["attack_lr"]}, iterations={one["frozen_attack"]["iterations"]}, restarts={one["frozen_attack"]["restarts"]}.

## Results

| Tier | Status | Prior wins | Prior mean delta | Prior p | Zero wins | Zero mean delta | Zero p |
|---|---|---:|---:|---:|---:|---:|---:|
| One-batch, 4 records, 1 Adam step | {_format_bool(one["baseline_gate_passed"])} | {one["confirmation"]["prior"]["wins"]}/{one["confirmation"]["prior"]["n"]} | {one["confirmation"]["prior"]["mean_difference"]:.6f} | {one["confirmation"]["prior"]["one_sided_sign_p"]:.6f} | {one["confirmation"]["zero"]["wins"]}/{one["confirmation"]["zero"]["n"]} | {one["confirmation"]["zero"]["mean_difference"]:.6f} | {one["confirmation"]["zero"]["one_sided_sign_p"]:.6f} |
| Scale check, 16 records, 4 Adam steps | {_format_bool(scale["baseline_gate_passed"])} | {scale["confirmation"]["prior"]["wins"]}/{scale["confirmation"]["prior"]["n"]} | {scale["confirmation"]["prior"]["mean_difference"]:.6f} | {scale["confirmation"]["prior"]["one_sided_sign_p"]:.6f} | {scale["confirmation"]["zero"]["wins"]}/{scale["confirmation"]["zero"]["n"]} | {scale["confirmation"]["zero"]["mean_difference"]:.6f} | {scale["confirmation"]["zero"]["one_sided_sign_p"]:.6f} |

Negative deltas mean the attack reconstruction MSE is lower than the control. The one-batch tier passes both controls. The 16-record tier beats zero-update but misses the prior-control gate, so it is not strong enough for a defense comparison at that scale.

## DNA Checks

| DNA setting | Scope | Result |
|---|---|---|
| Lossless DNA encode/decode | One-batch confirmation signals | max abs error = {dna["lossless_dna"]["max_abs_error"]:.1f}; this is a bit-exact transport comparator. |
| DNA Transform conservative + BPDA attacker | Same one-batch targets and seeds | higher-MSE wins = {dna["dna_transform"]["higher_mse_wins"]}/{dna["dna_transform"]["n"]}; mean DNA-baseline MSE delta = {dna["dna_transform"]["mean_mse_difference"]:.6f}; median delta = {dna["dna_transform"]["median_mse_difference"]:.6f}; p = {dna["dna_transform"]["one_sided_sign_p"]:.6f}. |

The DNA Transform result is mixed. The median moves in the intended direction, but the mean is slightly lower than baseline because of large paired outliers, and the sign test is not significant. This is not enough to claim robust inversion resistance.

## Decision

- Bounded one-batch diagnostic attack: validated.
- Lossless DNA: confirmed bit-exact, so it does not add reconstruction resistance in this protocol.
- DNA Transform conservative: inconclusive under the adaptive BPDA diagnostic.
- 16-record/four-step scale tier: baseline attack not validated against prior, so no DNA defense conclusion should be drawn there.
- Full FedAvg client-update attack: still not validated.

## Artifacts

- One-batch run: `{summary["paths"]["one_batch"]}`
- Scale run: `{summary["paths"]["scale"]}`
- Summary JSON: `{summary["paths"]["summary_json"]}`
- Summary CSV: `{summary["paths"]["summary_csv"]}`

## Reproduce

```bash
.venv-phase1/bin/python -m experiments.run_attack_redevelopment
.venv-phase1/bin/python -m experiments.run_adaptive_dna_confirmation artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.run_attack_scale_confirmation --source-run artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.summarize_attack_redevelopment artifacts/attack_redevelopment/run_20260908T180405538992Z artifacts/attack_redevelopment/scale_20260908T184713943293Z
```
"""
    path.write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("one_batch_run", type=Path)
    parser.add_argument("scale_run", type=Path)
    args = parser.parse_args()

    one_path = args.one_batch_run.resolve()
    scale_path = args.scale_run.resolve()
    one = _load(one_path / "attack_redevelopment_report.json")
    dna = _load(one_path / "dna_confirmation_report.json")
    scale = _load(scale_path / "attack_redevelopment_report.json")

    rows = [
        _comparison_row("one_batch_4_records_1_adam_step", one["confirmation"]),
        _comparison_row("scale_16_records_4_adam_steps", scale["confirmation"]),
    ]
    summary_json = one_path / "attack_redevelopment_final_summary.json"
    summary_csv = one_path / "attack_redevelopment_final_summary.csv"
    report_md = one_path / "attack_redevelopment_report.md"

    summary = {
        "status": "BOUNDED_ATTACK_VALIDATED_SCALE_INCONCLUSIVE",
        "one_batch": {
            "baseline_gate_passed": one["baseline_gate_passed"],
            "confirmation": one["confirmation"],
            "mean_baseline_mse": _mean_selected(one["selected_confirmation"], "baseline_mse"),
            "mean_prior_mse": _mean_selected(one["selected_confirmation"], "prior_mse"),
            "mean_zero_mse": _mean_selected(one["selected_confirmation"], "zero_mse"),
            "frozen_attack": one["frozen_attack"],
            "elapsed_seconds": one["elapsed_seconds"],
        },
        "dna_transform": dna,
        "scale": {
            "baseline_gate_passed": scale["baseline_gate_passed"],
            "confirmation": scale["confirmation"],
            "mean_baseline_mse": _mean_selected(scale["selected_confirmation"], "baseline_mse"),
            "mean_prior_mse": _mean_selected(scale["selected_confirmation"], "prior_mse"),
            "mean_zero_mse": _mean_selected(scale["selected_confirmation"], "zero_mse"),
            "elapsed_seconds": scale["elapsed_seconds"],
        },
        "decision": {
            "phase3_bounded_one_batch": "validated",
            "dna_transform_one_batch": "inconclusive",
            "scale_16_record_tier": "baseline_failed_prior_gate",
            "full_fedavg_update_attack": "not_validated",
        },
        "paths": {
            "one_batch": str(one_path),
            "scale": str(scale_path),
            "summary_json": str(summary_json),
            "summary_csv": str(summary_csv),
            "report_md": str(report_md),
        },
    }
    dump(summary_json, summary)
    _write_csv(summary_csv, rows)
    _write_markdown(report_md, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
