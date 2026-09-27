# Amendment: Priority 7 Minimal LeNet Sequential Timeout Probe

**Timestamp:** 2026-09-17, before rerunning any image-domain development gate
after the initial feasibility blocker.  
**Status:** DEVELOPMENT-GATE TECHNICAL AMENDMENT ONLY.  
**Applies to:** `protocols/config/rq1_image_domain_pre_pilot.yaml` and
`experiments/run_priority7_image_domain_gate.py`.

## Supervisor decision

After the original LeNet-style CIFAR CNN proved computationally infeasible for
the image-domain development gate under `torch.set_num_threads(1)`, supervisor
authorized a narrower technical feasibility probe:

1. Reduce the CNN to a minimal LeNet-5-style variant while keeping CIFAR-10
   images at the original `3 x 32 x 32` resolution. No image downscaling is
   allowed.
2. Run exactly one prioritized scope-1 development-gate cell first:
   `GEN_IDLG_STYLE` vs `v1_medium`.
3. Run fully sequentially (`workers=1`) with no multiprocessing.
4. Stop the job if it does not finish within `10800` seconds (3 hours).
5. If this prioritized cell completes with a valid gate result, continue with
   the remaining pre-approved image-domain cells under the same scope-1 gate
   plan. If it still does not complete within 3 hours, stop Package C entirely
   and report image-domain generalization as deferred due to feasibility limits.

## Frozen technical choices

The minimal CNN is:

- `Conv2d(3, 6, kernel_size=5, padding=2)`;
- ReLU;
- `MaxPool2d(2)`;
- `Conv2d(6, 16, kernel_size=5, padding=2)`;
- ReLU;
- `MaxPool2d(2)`;
- Flatten;
- `Linear(1024, 120)`;
- ReLU;
- `Linear(120, 84)`;
- ReLU;
- `Linear(84, 10)`.

This is a LeNet-5-style channel profile adapted to CIFAR-10 RGB input and
32x32 resolution. It changes only the reference CNN architecture used for the
image-domain feasibility gate; it does not change:

- CIFAR-10 target data;
- DNA v1/v2 formulas or parameters;
- attacker generation definitions;
- optimizer/loss definitions;
- exact sign-test gate;
- tabular RQ1/RQ2/RQ3 conclusions.

## Interpretation restriction

If the prioritized cell fails to finish within the 3-hour timeout, Package C is
stopped. This outcome must be reported as a computational feasibility limit of
the reference implementation under the single-thread-per-process constraint,
not as a privacy result.

