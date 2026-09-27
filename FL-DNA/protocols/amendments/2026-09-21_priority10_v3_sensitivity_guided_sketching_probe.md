# Amendment: Priority 10 Transform v3 Sensitivity-Guided Sketching Probe

**Timestamp:** 2026-09-21, before target generation or execution.  
**Status:** DEVELOPMENT-GATE ONLY — no confirmatory execution authorized.  
**Applies to:** IEEE-CIS n=8 exploratory probe for a new candidate mechanism,
Transform v3.

## Purpose

RQ1/RQ2/RQ3 conclusions for v1, v2, and IEEE-CIS remain frozen. This amendment
authorizes exactly one cheap exploratory n=8 development gate for a new
candidate mechanism, tentatively named DNA Transform v3.

The probe asks:

> Does v3, a non-uniform sketch that compresses the pre-registered sensitive
> final classifier layer more strongly than the rest of the update, reach
> resistance comparable to uniform v2 ratio 0.90 while strongly compressing
> only 33 parameters?

This is not a confirmatory run and cannot change any frozen conclusion.

## Literature basis for the sensitive block

The sensitive block is fixed before execution from the iDLG result of Zhao et
al. (2020): label information can be inferred from the sign pattern of the
final classification-layer gradient. For the project backbone
`models/fraud_mlp.py`, the final classifier is:

```text
nn.Linear(32, 1)
```

Therefore the sensitive block is exactly:

- `network.9.weight`: 32 parameters;
- `network.9.bias`: 1 parameter;
- total: 33 parameters.

No other layer, feature subset, or learned saliency criterion is authorized.

## Mechanism

Transform v3 is implemented as a wrapper around the frozen v2 primitive. The
file `dna_encoder/transform_defense_v2.py` must not be modified.

For each flattened update:

1. Split floating-point parameters into two blocks:
   - `sensitive`: `network.9.weight` and `network.9.bias`;
   - `rest`: all other floating-point parameters.
2. Apply the existing v2 SRHT plus quantization independently to each block:
   - `rest`: `compression_ratio=0.95`, `quantization_eta=0.01`;
   - `sensitive`: `compression_ratio=0.50`, `quantization_eta=0.01`.
3. Use the existing deterministic per-round/per-client seed derivation style.
4. Server-side lift is applied blockwise using the existing v2 lift, then
   blocks are unpacked into their original tensor positions.

The defense label for this probe is:

```text
v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01
```

## Development gate

Run exactly one IEEE-CIS development gate:

- dataset: IEEE-CIS Fraud Detection;
- scope: 4 rows/group, 1 fraud row/group, 1 client step;
- groups: 8;
- defense: `v3_sensitive_final_layer_ratio0p50_rest0p95_eta0p01`;
- attacker generation: `GEN_COSINE_TV`;
- controls: Prior and Zero-update;
- gate: exact one-sided sign test, `alpha=0.05`, must beat both controls.

The target set must be newly generated and source-disjoint with all prior
IEEE-CIS development, post-hoc, confirmatory, and Priority 6-9 target sets
found by the target-firewall logic.

## Stop rule

Stop after this n=8 gate regardless of the result:

- no n=24 escalation is authorized;
- no confirmatory run is authorized;
- no parameter change, layer change, or retuning after seeing results is
  authorized.

## Interpretation rules

If v3 is not broken by `GEN_COSINE_TV`, interpret only as a reason to consider
future investment: v3 may offer resistance comparable to uniform v2 ratio 0.90
while compressing far fewer parameters, and RQ2 utility would need separate
testing.

If v3 is broken, interpret as evidence against the narrow hypothesis that
strongly compressing only the final classifier layer is sufficient against
`GEN_COSINE_TV`. In that case, stop this direction and do not try additional v3
variants in this probe.

Attacker failure must not be converted into a privacy-success claim without
root-cause analysis.
