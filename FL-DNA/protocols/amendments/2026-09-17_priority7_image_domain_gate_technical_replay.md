# Amendment: Priority 7 Image-Domain Development-Gate Technical Replay

**Timestamp:** 2026-09-17, before any authorized image-domain development-gate
result is run or interpreted.  
**Status:** DEVELOPMENT-GATE TECHNICAL AMENDMENT ONLY.  
**Applies to:** `protocols/config/rq1_image_domain_pre_pilot.yaml` and
`experiments/run_priority7_image_domain_gate.py`.

## Reason

The first implementation smoke test exposed a mechanical implementation problem
when adapting the historical tabular v1 surrogate to CIFAR-10 CNN gradients:
the old surrogate materialized a dense `block_size x block_size` matrix for
every 256-element block. This is feasible for the small tabular MLP update but
becomes prohibitively large for the LeNet-style CNN gradient.

No development-gate result was completed or interpreted before this amendment.

## Technical replay changes

1. Replace dense v1 surrogate materialization inside the image-domain runner
   with the mathematically equivalent sparse/blockwise application:

   `((1 - mix) * block) + mix * shrink * block[permutation]`

   This preserves the exact frozen v1 surrogate formula and changes only the
   implementation strategy.

2. Freeze the image-domain development-gate optimizer budget at
   `iterations = 25` for this screening run. The original `iterations = 300`
   setting is computationally disproportionate for second-order CNN inversion
   under `torch.set_num_threads(1)`. This gate is explicitly development-only;
   a future image-domain confirmatory or escalation run would require a new
   amendment and a separate compute budget.

3. Keep all scientific/protocol choices unchanged:

   - CIFAR-10 dataset and BrainChip mirror provenance;
   - LeNet-style CNN;
   - SGD learning rate `0.01`, local steps `1`;
   - scope `1 image -> 4 images`;
   - authorized attacker matrix;
   - target seeds and data firewall;
   - exact one-sided sign-test gate against Prior and Zero-update;
   - no confirmatory execution.

## Interpretation restriction

Because `iterations = 25` is a practical development-screening budget, a FAIL
result cannot be interpreted as evidence that DNA v1/v2 is private on images.
A PASS result may justify a separately amended, larger-budget escalation, but
does not by itself change any frozen tabular RQ1/RQ2/RQ3 conclusion.
