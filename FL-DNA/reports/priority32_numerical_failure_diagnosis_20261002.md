# Priority 32 numerical-failure diagnosis

Date: 2026-10-02. Status: diagnosis completed; scientific repair awaits approval
of the specific buffer-handling change. Original artifacts remain unchanged.

## Original execution

All 189 jobs were submitted and the queue drained. The supervisor exited at
2026-10-01T21:56:13.803919Z (2026-10-02 04:56:13 Asia/Ho_Chi_Minh).
There are 88 valid results and 101 missing results:

| Dataset | Baseline valid/missing | V1 valid/missing | V2 valid/missing |
|---|---:|---:|---:|
| PaySim | 21/0 | 9/12 | 3/18 |
| IEEE-CIS | 21/0 | 10/11 | 0/21 |
| BAF | 21/0 | 3/18 | 0/21 |

Every missing path is recorded in
artifacts/priority32_multidataset/INCOMPLETE_NUMERICAL_FAILURE.json.
Only BAF/v1/321001 has a captured original worker traceback. Other missing
outputs must not be asserted to have the same root cause without evidence.
No incomplete-pair statistics have been computed.

## Frozen diagnostic replay and observed cause

Amendment: protocols/amendments/2026-10-02_priority32_numerical_failure_diagnosis.md,
written before execution. Command:

```text
.venv-phase1/bin/python -B experiments/diagnose_priority32_numerical_failure.py
```

One diagnostic replay, no interruption, no early stopping, all 50 rounds;
BAF/v1 conservative/seed321001, original prepared data, training configuration
and RNG contract. Only quality_validation_only is set true to forbid test
evaluation; it does not affect training or validation probabilities. No
scientific result replaces the original job. Runtime through final round
approximately 103 seconds; one Torch thread. Imported train_job and original
transform math are unchanged. Hooks only read tensors; no RNG draws added.

| Observation | Exact value |
|---|---|
| First aggregate with negative variance | round 2, network.1.running_var |
| Round-2 minimum / negative coordinates | -11.915035247802734 / 7 |
| Round-50 minimum / negative coordinates | -2.616039752960205 / 3 |
| Round-50 network.5.running_var minimum | 0.36758148670196533 |
| Round-50 network.9.running_var minimum | 0.28450557589530945 |
| First nonfinite validation layer | network.1 (BatchNorm) |
| Previous layer | network.0 input and output both finite |
| Reproduced exception | ValueError: Input contains NaN. |

Evidence: diagnosis_20261002/evidence.json and final_state.pt under the P32
artifact root; full round evidence and per-layer finite flags retained.
All final variance values themselves are finite, but the first BN has
negative variance far below -eps. BatchNorm evaluation normalizes with
sqrt(running_var + eps); those coordinates are outside its mathematical
domain. Training-mode BN uses batch statistics, explaining why training can
complete despite invalid running statistics.

Code chain: experiments/priority32_multidataset.py:363-372 applies the state
transform before FedAvg. experiments/run_fraud_fl_dna_transform.py:71-86
filters only floating tensors and adds the transformed delta back to global
state, including running_var. dna_encoder/transform_defense.py:77 applies
residual mixing to signed updates; it provides no positivity guarantee on
the resulting variance state. models/fraud_mlp.py:18 is the first BN.
experiments/priority32_multidataset.py:296-303 evaluates in eval mode.

This establishes the cause for the diagnosed BAF/v1 seed, not all 101
missing jobs. It is not resolved by catching sklearn errors, replacing
probabilities, dropping seeds or retrying the same configuration.

## Required scientific choice before repaired execution

Proposed explicit extension: transforms operate on trainable parameter
deltas only; all BN buffers retain their ordinary client values and are
aggregated with existing FedAvg. This is raw BN aggregation, NOT FedBN.
It changes the transmitted protocol and leaves a raw BN leakage channel;
therefore it must be separately labeled and must not replace original RQ2
findings. No clamp is proposed. Rerun all 126 transform jobs in a new folder,
including the 25 originally finite transform results, to avoid mixing
protocol definitions. Reuse the 63 original baseline jobs with checksummed
pairing. Seeds, margins, data, architecture and training remain fixed.
This specific choice is not executed yet; repair amendment is a draft.

## Provenance and checks

New-file SHA-256 values, including this report and the draft repair amendment,
are in artifacts/priority32_multidataset/diagnosis_20261002/final_hashes.json.
Diagnostic model and evidence hashes are also in diagnosis_20261002/hashes.json.
py_compile and git diff --check are checked after saving the new files.
No remaining diagnostic/training workload; Priority 33 has not started.
