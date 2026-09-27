# Amendment: Priority 7 CIFAR-10 v2 GEN_COSINE_TV n=24 Development Escalation

**Timestamp:** 2026-09-17, before generating n=24 targets or running the
escalated development pilot.  
**Status:** DEVELOPMENT ESCALATION ONLY — no confirmatory execution authorized.  
**Applies to:** Priority 7 CIFAR-10 image-domain generalization.

## Authorized cell

Only the following cell is escalated:

- dataset: CIFAR-10;
- model: minimal LeNet-5-style CNN frozen in
  `2026-09-17_priority7_image_domain_minimal_lenet_timeout.md`;
- defense: `v2_ratio0p95_eta0p01`;
- attacker generation: `GEN_COSINE_TV`;
- scope: `4 images / 1 client step`;
- optimizer budget: `iterations=25`, `restarts=4`;
- execution: sequential (`workers=1`), `torch.set_num_threads(1)`.

No other Priority 7 cell is escalated:

- `v1_medium__GEN_IDLG_STYLE` failed at image-domain scope 1;
- `v1_medium__GEN_RAW_LIFT` failed at image-domain scope 1;
- `v2_ratio0p95_eta0p01__GEN_RAW_LIFT` failed at image-domain scope 1.

## Rationale

`v2_ratio0p95_eta0p01__GEN_COSINE_TV` is the only image-domain cell that passed
both development gates already run:

- scope 1 image: `8/8` vs Prior and `8/8` vs Zero-update,
  one-sided exact sign-test `p=0.00390625` for both controls;
- scope 4 images: `8/8` vs Prior and `8/8` vs Zero-update,
  one-sided exact sign-test `p=0.00390625` for both controls.

Because the pass was strong and consistent at both scopes, this amendment
authorizes exactly one n=24 development pilot at the harder scope-4 setting.

## Target generation

Create a new CIFAR-10 target set with:

- `groups=24`;
- `images_per_group=4`;
- CIFAR-10 train split only;
- source-disjoint with all previous Priority 7 CIFAR target manifests,
  including:
  - `minimal_scope1_targets_20260917`;
  - `scope4_targets_20260917`.

The target manifest must record overlap evidence and must pass with
`max_overlap=0` before execution.

## Gate and stopping rule

Use the same development-gate rule as Priority 7:

- exact one-sided sign test;
- `alpha=0.05`;
- must beat both Prior and Zero-update controls;
- primary metric: image MSE.

If the n=24 pilot does not maintain a strong pass, stop and report possible
small-sample optimism. Do not run confirmatory.

If the n=24 pilot passes strongly, stop after reporting n=24 results. A separate
confirmatory amendment and supervisor quick nod are required before any
confirmatory target generation or execution. Any confirmatory power analysis
must use the exact-binomial method used in Priority 6; if the chosen `p1`
differs from the historical `0.70`, supervisor quick nod is required.

