# ATS Porting Notes

Pinned official checkout: `2496fc9571fe68bc65e839cbf6ccb6d58d27c93a`.

## Mechanism

ATS searches a policy of up to three image transformations and trains on transformed, not original, images. The implementation applies parsed policy IDs sequentially in `benchmark/comm.py:55-65,189-196`; the 50 fixed transformations are instantiated in `policy.py:3-58`, and `autoaugment.py:22-85` defines their magnitude grids and application.

## Original configuration

The paper searches 1,500 candidate policies of at most three functions, uses CIFAR-100/ResNet20 with 200 epochs, and evaluates Adam+cosine reconstruction for 4,800 iterations (`benchmark/comm.py:481-496`). Its published headline is 13.88 dB (no transform) versus 7.64 dB (hybrid ATS) PSNR and 76.88% versus 77.92% accuracy for CIFAR-100/ResNet20 (Table 1); PSNR is against the transformed target.

## Server information

The server receives ordinary gradients and aggregates them normally. It does not need the client policy to aggregate, but a realistic white-box attacker may know the transformation family and policy distribution; the raw image was discarded by the client.

## Porting

ATS does not have a direct semantics-preserving analogue for a 13-feature tabular MLP: do not map image augmentations onto arbitrary feature perturbations and call it ATS. A tabular port requires a pre-registered, domain-valid transformation library and must score reconstruction against both raw and transformed records.

For a LeNet-style CNN, use exactly the supplied `3-1-7` or a fixed searched policy before tensor conversion/normalization, as shown by `benchmark/comm.py:152-156`; maintain the original paper's separation between the protected raw image and the attacker target. The documented pipeline cannot run unmodified on CPU: it hard-codes a one-GPU Lightning trainer and contains a syntax error in `searchalg/search_best.py:47`.

## Known adaptive attacks

Balunovic et al. (ICLR 2022, arXiv:2111.04706) re-evaluate ATS and show leakage early in training. Gao et al. themselves state in Section 6 that a more sophisticated attacker may bypass the defense; Yue et al. 2023 does not evaluate ATS.

## Reproduction status

`../run_cpu_qualitative_checks.py` executes the pinned `3-1-7` policy after a temporary in-process compatibility alias for removed `np.int`, without editing the checkout. It confirms the policy changes a fixed image (logged raw-to-transformed PSNR 5.4597 dB); it is not a trained-model inversion reproduction.
