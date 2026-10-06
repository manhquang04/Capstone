# Phase 3 Attack Redevelopment Report

This report summarizes the follow-up attack development after the earlier full-client Adam attempts failed to beat controls reliably. The scope is deliberately bounded: the validated attack uses a four-record, one-step Adam local update with known labels, known record order, known local RNG, known model architecture, known optimizer, and the pre-local checkpoint. It is not a full FedAvg client-update attack.

## Protocol

- Attack input parameterization: PaySim manifold with six optimized base numeric fields, two derived balance-difference fields, and softmax transaction-type logits.
- Objective: balanced per-tensor update matching with BatchNorm running buffers included at weight 1.0.
- Candidate selection: minimum attacker objective within each group/restart family; ground truth is used only for evaluation metrics.
- Confirmation gate: baseline must beat both prior and zero-update controls with negative mean delta, negative median delta, and one-sided exact sign p < 0.05.
- Frozen attack: `manifold_bn1`, lr=0.05, iterations=300, restarts=3.

## Results

| Tier | Status | Prior wins | Prior mean delta | Prior p | Zero wins | Zero mean delta | Zero p |
|---|---|---:|---:|---:|---:|---:|---:|
| One-batch, 4 records, 1 Adam step | PASS | 10/12 | -342.856957 | 0.019287 | 11/12 | -450.825332 | 0.003174 |
| Scale check, 16 records, 4 Adam steps | FAIL | 9/12 | -199.871000 | 0.072998 | 11/12 | -248.595570 | 0.003174 |

Negative deltas mean the attack reconstruction MSE is lower than the control. The one-batch tier passes both controls. The 16-record tier beats zero-update but misses the prior-control gate, so it is not strong enough for a defense comparison at that scale.

## DNA Checks

| DNA setting | Scope | Result |
|---|---|---|
| Lossless DNA encode/decode | One-batch confirmation signals | max abs error = 0.0; this is a bit-exact transport comparator. |
| DNA Transform conservative + BPDA attacker | Same one-batch targets and seeds | higher-MSE wins = 8/12; mean DNA-baseline MSE delta = -6.004947; median delta = 10.610876; p = 0.193848. |

The DNA Transform result is mixed. The median moves in the intended direction, but the mean is slightly lower than baseline because of large paired outliers, and the sign test is not significant. This is not enough to claim robust inversion resistance.

## Decision

- Bounded one-batch diagnostic attack: validated.
- Lossless DNA: confirmed bit-exact, so it does not add reconstruction resistance in this protocol.
- DNA Transform conservative: inconclusive under the adaptive BPDA diagnostic.
- 16-record/four-step scale tier: baseline attack not validated against prior, so no DNA defense conclusion should be drawn there.
- Full FedAvg client-update attack: still not validated.

## Artifacts

- One-batch run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/run_20260908T180405538992Z`
- Scale run: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/scale_20260908T184713943293Z`
- Summary JSON: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/run_20260908T180405538992Z/attack_redevelopment_final_summary.json`
- Summary CSV: `/Users/rhys24z/Documents/UNIVERSITY/Capstone/FL-DNA/artifacts/attack_redevelopment/run_20260908T180405538992Z/attack_redevelopment_final_summary.csv`

## Reproduce

```bash
.venv-phase1/bin/python -m experiments.run_attack_redevelopment
.venv-phase1/bin/python -m experiments.run_adaptive_dna_confirmation artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.run_attack_scale_confirmation --source-run artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.summarize_attack_redevelopment artifacts/attack_redevelopment/run_20260908T180405538992Z artifacts/attack_redevelopment/scale_20260908T184713943293Z
```
