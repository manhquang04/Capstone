# Amendment: Priority 7 CIFAR-10 v2 GEN_COSINE_TV Confirmatory Run

**Timestamp:** 2026-09-17, before confirmatory target generation or execution.  
**Status:** CONFIRMATORY AUTHORIZED FOR EXACTLY ONE RUN.  
**Applies to:** Priority 7 CIFAR-10 image-domain generalization.

## Confirmatory cell

This amendment authorizes exactly one confirmatory run for:

- dataset: CIFAR-10;
- model: minimal LeNet-5-style CNN;
- scope: `4 images / 1 client step`;
- defense: DNA Transform v2 `ratio=0.95`, `eta=0.01`;
- attacker generation: `GEN_COSINE_TV`;
- controls: Prior and Zero-update;
- primary metric: image MSE;
- gate: exact one-sided binomial/sign-test, `alpha=0.05`, must beat both controls.

No other image-domain cell is authorized for confirmatory execution by this
amendment.

## Power analysis and target count

The n=24 development pilot passed strongly (`24/24` vs both controls), but the
confirmatory effect size is not refit from that observed win rate.

The confirmatory power target uses the RQ1/ Priority 6 precommitted value:

- null win probability: `p0 = 0.50`;
- minimum meaningful win probability: `p1 = 0.70`;
- alpha: `0.05`;
- target power: `0.80`;
- test: exact one-sided binomial sign test.

Required non-tied groups: `n = 37`.

Following Priority 6 practice, draw `39` target groups to retain a small buffer
for possible infrastructure failures or ties. Do not increase or repeat this
run after seeing results.

## Target generation

Create a fresh CIFAR-10 target set with:

- `groups=39`;
- `images_per_group=4`;
- CIFAR-10 train split only;
- source-disjoint with all previous Priority 7 CIFAR target manifests,
  including:
  - scope-1 n=8;
  - scope-4 n=8;
  - scope-4 n=24.

The target manifest must show `max_overlap=0` before execution.

## Execution rule

Run exactly once:

- `workers=1`;
- `torch.set_num_threads(1)`;
- `iterations=25`;
- `restarts=4`;
- no tuning after target generation;
- no scope changes;
- no confirmatory rerun if the result is unfavorable.

## Interpretation

If the confirmatory run passes both controls, it is confirmatory image-domain
evidence that `GEN_COSINE_TV` successfully attacks DNA Transform v2 in this
CIFAR-10 minimal-LeNet setting. It is a domain-generalization result and does
not rewrite frozen tabular reports, but it should be reported as an important
supplementary finding.

If the run fails either control, report the failure as the official confirmatory
outcome for this image-domain cell. Do not create additional target sets or
rerun.

