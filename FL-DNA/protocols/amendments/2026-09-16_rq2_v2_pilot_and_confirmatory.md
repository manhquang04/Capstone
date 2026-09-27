# Amendment: RQ2 DNA Transform v2 pilot and confirmatory design

**Date:** 2026-09-16  
**Status:** PRE-PILOT FROZEN

DNA Transform v2 (`compression_ratio=0.95`, `quantization_eta=0.01`) is
evaluated under the existing RQ2 framework: paired F1 and AUC-ROC
non-inferiority margins 0.02 and 0.005, respectively; alpha 0.05; target
power 0.80; both endpoints must pass. No margin or endpoint rule is changed
for v2.

The v2 pilot uses eight new, deterministic seeds
`[733003850,1065351580,880535557,48511685,1771003338,672591598,589150035,848504097]`.
They are generated as the first 31 bits of SHA-256(`RQ2-V2|PILOT|2026-09-16|i`),
i=0..7, and are disjoint from all prior RQ2 v1 and v2 smoke seeds. Every seed
controls one shared split/partition/model-init/loader-order realization for
Baseline and v2. The pilot is 500k rows, 50 rounds, three clients, one local
epoch, focal loss as in frozen RQ2.

Use the conservative upper variance bound specified in `rq2_multiseed_protocol.md`
to calculate v2-specific required n. If required n exceeds the frozen cap 30,
report underpowered/precision-limited; do not enlarge the cap or alter margins.
Before any confirmatory run, freeze a separate v2 seed list and machine-readable
contract, source-disjoint from all development and historical RQ2 seeds.

Exactly one confirmatory execution is allowed after that freeze. No parameter,
seed, method, threshold, or stopping rule is tuned after pilot or confirmatory
outcomes. `torch.set_num_threads(1)` is unchanged.
