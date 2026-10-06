# Independent replication: five CIFAR-10 image targets

**Verdict: Partially reproduced.** This is an independent implementation. It
does not import or call `experiments/priority30_native_defenses/`. The
undefended reconstruction reproduces all five S0u records exactly; the
70%-pruning mask-aware condition matches S1c to displayed precision; Soteria
matches within 1 dB on four of five targets. The independently implemented
known-key v2-style sketch-space attack does not reproduce S1c: every target
differs by more than 1 dB and three reverse the defense-versus-undefended PSNR
direction.

## Frozen target and execution

The frozen inputs and acceptance rules are in `target.md`. Target IDs 0--4 map
to CIFAR-10 test indices `2310, 2290, 241, 5402, 8641`, respectively. Each is
a known-label, batch-one attack on untrained official
`construct_model('LeNetZhu', seed=42)`. The official
`GradientReconstructor` configuration is cosine cost, signed Adam, learning
rate 0.1, 4,800 iterations, one restart, TV 0.01, boxed projection,
learning-rate decay, and loss scoring. Initial reconstruction seed is
`30600 + target_id` for every condition.

The independent adapter implements: Soteria's client-side representation
sensitivity mask then E2 omission of the final-classifier weight gradient;
per-tensor 70% lowest-magnitude pruning then a retained-coordinate global
cosine loss; and a known-key v2-style, 95%-retained signed-Hadamard projection
with eta 0.01 stochastic quantization and a sketch-space MSE loss. The v2
projection/quantizer was independently written in `run_replication.py`; it
does not invoke the Priority 30 implementation.

Command (exit 0):

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
VECLIB_MAXIMUM_THREADS=2 external_defenses/.venv/bin/python -B \
  literature/replication/run_replication.py
```

The successful corrected run is compute job `fc27e967-11c`. Raw results are
`replication_results.json` (SHA-256
`dab596b362b794dc70456442d6d438fbc27d5f038d6d9401e55b40a4917291c8`).

## Per-target results and comparison

Each metric cell is `PSNR dB / SSIM`. `Codex` is S0u for undefended and S1c E2
for the three adaptive conditions. `Delta` is independent PSNR minus Codex
PSNR. Direction compares each condition against the same target's S0u
undefended PSNR: `+` means defense reconstruction is better and `-` worse.
`FLAG` means absolute PSNR difference exceeds 1 dB or the directions disagree.

| Target | Condition | Independent | Codex | Delta dB | Independent/Codex direction | FLAG |
|---:|---|---|---|---:|---|---|
| 0 | undefended | 23.272770 / 0.873442 | 23.272770 / 0.873442 (S0u) | 0.000000 | baseline | no |
| 0 | Soteria omit | 24.119666 / 0.900665 | 24.280833 / 0.900550 | -0.161167 | + / + | no |
| 0 | pruning mask | 20.757107 / 0.768740 | 20.757107 / 0.768740 | 0.000000 | - / - | no |
| 0 | v2 sketch | 21.199524 / 0.833769 | 23.306274 / 0.872460 | -2.106750 | - / + | **yes** |
| 1 | undefended | 16.023932 / 0.662844 | 16.023932 / 0.662844 (S0u) | 0.000000 | baseline | no |
| 1 | Soteria omit | 21.085723 / 0.879059 | 19.686325 / 0.843917 | +1.399398 | + / + | **yes** |
| 1 | pruning mask | 14.081990 / 0.496642 | 14.081990 / 0.496642 | 0.000000 | - / - | no |
| 1 | v2 sketch | 11.193930 / 0.053458 | 17.962465 / 0.775486 | -6.768536 | - / + | **yes** |
| 2 | undefended | 23.827506 / 0.871490 | 23.827506 / 0.871490 (S0u) | 0.000000 | baseline | no |
| 2 | Soteria omit | 26.122227 / 0.929974 | 26.106041 / 0.929876 | +0.016186 | + / + | no |
| 2 | pruning mask | 21.371213 / 0.750353 | 21.371213 / 0.750353 | +0.000000 | - / - | no |
| 2 | v2 sketch | 21.718851 / 0.795480 | 23.763523 / 0.871784 | -2.044672 | - / - | **yes** |
| 3 | undefended | 30.845444 / 0.845649 | 30.845444 / 0.845649 (S0u) | 0.000000 | baseline | no |
| 3 | Soteria omit | 33.110135 / 0.910246 | 33.197059 / 0.910612 | -0.086924 | + / + | no |
| 3 | pruning mask | 28.553187 / 0.742733 | 28.553187 / 0.742733 | +0.000000 | - / - | no |
| 3 | v2 sketch | 28.290812 / 0.779500 | 30.959682 / 0.844933 | -2.668870 | - / + | **yes** |
| 4 | undefended | 24.845092 / 0.849610 | 24.845092 / 0.849610 (S0u) | 0.000000 | baseline | no |
| 4 | Soteria omit | 26.500477 / 0.896753 | 26.487643 / 0.897679 | +0.012834 | + / + | no |
| 4 | pruning mask | 22.312950 / 0.742067 | 22.312950 / 0.742067 | +0.000000 | - / - | no |
| 4 | v2 sketch | 22.878502 / 0.798470 | 24.814026 / 0.848137 | -1.935523 | - / - | **yes** |

## Findings

- **Undefended:** all five PSNR, SSIM, MSE, and optimizer scores equal S0u to
  stored floating-point precision. This validates the target IDs, CIFAR data,
  model initialization, normalization, reconstruction seed, and official
  optimizer configuration.
- **Soteria E2 omission:** four targets meet the pre-specified 1 dB tolerance;
  target 1 is 1.399398 dB higher than S1c. All five have the same direction as
  S1c, namely better reconstruction than the undefended control.
- **Gradient pruning E2:** all five match S1c to the displayed precision, both
  in PSNR/SSIM and direction. Every target is worse than undefended.
- **Known-key v2 sketch E2:** all five exceed the 1 dB PSNR threshold
  (2.044672--6.768536 dB). Targets 0, 1, and 3 disagree with S1c on direction:
  the independent sketch attack is worse than undefended, while S1c is slightly
  better. Targets 2 and 4 agree on the worse-than-undefended direction.

The v2 discrepancy is not evidence that either result is invalid. It localizes
an under-specified implementation-sensitive component: the exact sketch
operator/quantization/optimization treatment used by S1c is not recoverable
from the user-provided setting alone. The independent runner specifies its
operator explicitly, and its non-v2 controls establish that this discrepancy
is not explained by target selection, model initialization, or the official
Geiping attack configuration.

## Corrections retained

Two earlier raw outputs are retained for auditability but are not scientific
results: `replication_results_normalized_metric_bug.json` scored normalized
reconstructions against raw images, and
`replication_results_per_tensor_cosine_bug.json` used per-tensor rather than
the official global cosine. The final output corrected these two reporting or
loss-definition errors without changing targets, defense definitions, attack
budget, or seeds.

## Scope limits

This is a five-target replication check, not a 39-target inferential audit. No
sign tests, Holm correction, training, or dataset download were performed.
