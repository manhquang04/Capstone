# Amendment — Priority 23 BN channel, repaired CIFAR attack, and RQ2 facts

Date written: 2026-09-28  
Status: FROZEN / AUTHORIZED FOR PRIORITY 23 EXECUTION  
Scope: supplementary evidence only; no change to frozen RQ1/RQ2/RQ3 conclusions and no edits under `Latex/`.

## Motivation

A code-reading review identified three issues requiring new evidence or formal
re-analysis:

1. The implemented RQ2 transform and DP transport paths operate on every
   floating `state_dict` entry, including BatchNorm `running_mean` and
   `running_var`. These buffers may leak batch statistics independently of the
   gradient-inversion attackers.
2. The CIFAR-10 Transform-v2 `GEN_COSINE_TV` break needs a repaired audit with
   stronger controls and a positive-control branch.
3. RQ2 utility reports have not summarized PR-AUC, and the v1 RQ2
   configuration label must be checked against the frozen config.

This amendment freezes the design before any Priority 23 new attack outputs are
generated.

## Part A — BatchNorm-statistics leakage channel

Targets: reuse the Priority-16 confirmatory targets only:

- `artifacts/priority16_v1_medium_dp_head_to_head/confirmatory_targets_20260927/paysim_priority16_v1_medium_confirmatory_targets.pt`
- expected SHA-256 prefix: `645fe7ff`
- target count: 39 bounded groups, 4 records/group.

Capture path: use the same PaySim bounded-update capture path as Priority 16:
`experiments.priority16_v1_medium_dna_vs_dp_probe.capture_paysim`, with the
same target-group records and derived local seed convention.

Defenses to evaluate on the full transmitted state/update:

- `none`;
- Transform v1 conservative: `mix=0.08`, `keep=0.88`, `shrink=0.45`;
- Transform v1 medium: `mix=0.10`, `keep=0.85`, `shrink=0.40`;
- Transform v1 stronger: `mix=0.12`, `keep=0.82`, `shrink=0.35`;
- Transform v2 primary: `compression_ratio=0.95`, `quantization_eta=0.01`,
  decoded with the seed/metadata that the server holds by construction
  (Level 2 for the BN-buffer channel);
- DP at clip norm `100` and noise multipliers `0.000315`, `0.0004`,
  `0.00105`;
- DP at the trainable-vector recalibrated multiplier(s) computed in Part C.

Attack: closed-form recovery of the target batch mean from the first BatchNorm
`running_mean` update after the first linear layer:

`batch_mean_z = running_mean_before + delta_running_mean / momentum`, then
solve `W1 * mean_x = batch_mean_z - b1` by least squares/pseudoinverse.

For Transform v1, the attacker observes only the transmitted transformed
buffer, does not invert the v1 seed, and applies the recovery directly to that
transmitted buffer (Level 1). For Transform v2, the server-held decode metadata
is used before applying the recovery because decoded updates are what the
server aggregates.

Also report whether `running_var` yields second-moment information. This is
descriptive: `running_var` reveals statistics of first-layer activations, not a
unique closed-form feature second moment without additional assumptions.

Metrics:

- raw robust-scaled feature MSE between recovered mean and true 4-record mean;
- per-feature-standardized MSE using the empirical feature standard deviation
  used for the diagnostic population;
- baselines: population feature mean and zero vector;
- one-sided exact sign test, recovered mean vs population-mean baseline, with
  a win defined as strictly lower MSE for recovery than for the baseline.

Realistic-setting descriptive check: for one existing full RQ2-style
configuration client update, report what the transmitted first BN
`running_mean` delta reveals and compare it to the client data mean and the
mean of the final observed mini-batch where available. This is descriptive only.

## Part B — repaired CIFAR-10 attack

Target domain: reuse the same CIFAR-10 confirmatory target bundle used by
Priority 7:

- `artifacts/priority7_image_domain/confirmatory_v2_cosine_targets_20260917/cifar10_priority7_targets.pt`
- target count: 39 scope-4 image groups.

Branches:

- `none`: undefended trainable update;
- Transform v2 `compression_ratio=0.95`, `quantization_eta=0.01`, seed/metadata
  known to the server (Level 2);
- Transform v1 medium, using the same Level-1 surrogate family as Priority 7.

Budgets:

1. Continuity budget: original Priority-7 budget, 25 Adam iterations,
   cosine-only objective, `attack_lr=0.05`.
2. Stronger repaired budget: 250 Adam iterations (10x original), cosine plus
   an image total-variation prior. This is selected as the largest fixed
   budget to attempt first under roughly one day of compute on the project
   machine while preserving the existing single-thread-per-process constraint.
   No iteration count, learning rate, TV weight, target count, or branch set may
   be changed after seeing outcomes.

Controls:

- Prior: the attacker's own initial image tensor;
- decoy: the same attacker and initialization, but with a same-defense signal
  from a different target group;
- constant gray image with pixel value `0.5`;
- CIFAR-10 training-set mean image.

Metrics: input-image MSE, PSNR, SSIM, and defense-efficacy ratio
`branch_psnr / none_psnr` for matched targets. Primary test is own-signal vs
decoy, one-sided exact sign test. Secondary tests compare reconstruction to
gray and mean-image controls. Restart selection must use the
attacker-observable objective only.

Save a PNG grid of example reconstructions for 4 fixed groups per branch and
budget. No early stopping; report every target/branch/budget attempted.

## Part C — DP recalibration on trainable-only vectors

Calibration only; do not rerun utility or head-to-head experiments.

For Transform v1 conservative/medium/stronger and v2 `0.95/0.01`, compute the
Gaussian multiplier at clip norm `100` that matches the transform's L2
distortion over trainable parameters only on the Priority-16 target captures.
Report the multiplier, single-release epsilon, 50-release epsilon, and the
ratio to the corresponding multiplier used in Priority 14/16/20.

RDP convention: same as Priority 3: delta `1e-5`, update-level add/remove
adjacency, sensitivity ratio `1`, one-release and 50-release composition.

## Part D — RQ2 factual re-analysis

Use only existing RQ2 artifacts. Report:

- exact v1/v2 configuration used, with config and metrics evidence;
- paired mean delta, sample SD, and 95% CI for F1, ROC-AUC, and PR-AUC vs the
  unprotected baseline;
- baseline absolute F1, ROC-AUC, and PR-AUC means;
- whether BN buffers were transformed/noised in each RQ2 path by code citation;
- whether any v1-medium or v1-stronger RQ2 run exists.

## Non-negotiables

- No edits under `Latex/`.
- Do not modify existing artifacts or reports.
- Keep `torch.set_num_threads(1)` behavior unchanged.
- Parts A and B are the only new attack/recovery computations. Parts C and D
  are calibration/re-analysis only.
- No early stopping; state negative outcomes plainly.
