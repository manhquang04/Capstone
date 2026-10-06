# Gradient-Pruning Porting Notes

Pinned official checkout: `d21007fa1540ba2303ebc034976aa331814727c7`.

## Mechanism

Zhu et al. Section 5.2 define the defense as setting small-magnitude gradient coordinates to zero before sharing. The official DLG checkout implements the attack (`main.py:67-92`) but does **not** contain the Section 5 pruning implementation; the closest retained implementation is ATS's explicit per-tensor magnitude pruning in `../ats/inversefed/training/training_routine.py:156-163,181-184`.

## Original configuration

The paper evaluates sparsity from 1% to 70% and reports DLG failure once the pruning ratio exceeds roughly 20%; it uses its L-BFGS DLG implementation and a CIFAR-100 model. Its noisy comparison uses zero-central Gaussian/Laplacian noise with reported variance range `1e-4` to `1e-1`.

## Server information

The server needs no key or decoder: it averages zeros and surviving coordinates directly. It knows the architecture, weights, and which coordinates are transmitted, and a defense-aware reconstruction objective can simply compare masked dummy gradients.

## Porting

For a 13-feature BatchNorm MLP and for the LeNet-style CNN, flatten each trainable gradient tensor and zero the bottom `p%` of absolute values **per tensor** if following ATS's available implementation; alternatively implement one global threshold, but label it as a protocol change because DLG Section 5 does not specify this detail. Do not prune BatchNorm buffers, as they are not gradient coordinates.

## Known adaptive attacks

Balunovic et al. (ICLR 2022, arXiv:2111.04706) re-evaluate pruning with a Bayesian defense-aware attacker. Yue et al. (USENIX Security 2023, arXiv:2206.04055) reconstruct Top-k compressed gradients at 95% sparsity with ROG.

## Reproduction status

`main.py --index 25` was executed unchanged on CPU under PyTorch 2.5.1 for its 300 outer iterations (`../logs/dlg_original_cpu.log`), but the gradient-match loss plateaued at 352.5049 rather than converging, so no paper-level privacy result is claimed. The separately logged fallback verifies exact 70% lowest-magnitude coordinate removal (`../run_cpu_qualitative_checks.py`).
