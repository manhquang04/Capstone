# Adaptive Defense and Attack Implementations

## Contract and scope

Code is in `adaptive/` and implements the `Defense.encode`, `Defense.server_view`, and `Attack.reconstruct` contracts from `experiments/harness/interfaces.py`, without modifying that file. The harness attack signature has no label, input shape, transformation, or stochastic-forward parameter; attack constructors therefore require those inputs explicitly. All random initializations use a private, fixed-seed `torch.Generator`, and CPU scripts set four threads.

`adaptive/defenses.py` contains pure update-dictionary transforms where the mechanism permits one. PRECODE and ATS cannot be completely represented as a transform of an already-computed update: PRECODE changes the model forward pass before gradients exist, and ATS changes the raw input before gradients exist. Their implementations are therefore faithful harness adapters that preserve the update and describe the required forward/input-stage operation in metadata and docstrings.

## Implementation map

| Defense | Original-style attacker | Adaptive attacker | Source / exact implementation quote | Status |
|---|---|---|---|---|
| Soteria | `GradientMatchingAttack`: full received gradient matching (the defense-unaware DLG/IGA style). | `OmitGradientAttack`: excludes `defended_gradient`; with no name it enumerates every observed `*.weight` candidate and retains the lowest objective. | Balunovic et al. (ICLR 2022), §5.2: "we simply remove the gradients of the defended layer from the set of observed gradients." Implemented in `adaptive/attacks.py:OmitGradientAttack`. | Implemented. The defense requires client-computed representation scores, supplied to `SoteriaDefense`, because scores cannot be derived from an update alone. |
| PRECODE | `GradientMatchingAttack` over all update entries. | `OmitGradientAttack` can omit listed stochastic/VB gradients. `StochasticNoiseAttack` jointly optimizes input plus noise only when a model-specific `noise_setter` is supplied. | Scheliga et al. (arXiv:2309.04515), §4.2: "omitting the gradients of the stochastic layers"; Scheliga et al. (arXiv:2208.06163), abstract: "DIA jointly optimizes for client data and dropout masks to approximate the stochastic client model." | Partial but explicit. A generic PyTorch model has no API to inject reparameterization noise, so the joint-noise variant raises `NotImplementedError` without a supplied hook rather than pretending it was applied. |
| Gradient pruning | `GradientMatchingAttack` treats zeroed coordinates as ordinary received values, reproducing a defense-unaware attack. | `MaskAwareGradientAttack` matches only retained coordinates, using masks retained in transmitted metadata or a nonzero-derived fallback. | Yue et al. (USENIX 2023), §3.2: "the attacker can incorporate the gradient post-processing function into the reconstruction optimization." | Implemented. Pruning uses deterministic per-tensor lowest-absolute-value masking. |
| ATS | `GradientMatchingAttack` acts on the transmitted gradient and thus targets the transformed input. | `KnownTransformAttack` optimizes a raw dummy input through a supplied differentiable known transform, and scores its returned estimate in raw-input space. | Gao et al. (CVPR 2021), §6: "a more sophisticated adversary may try to bypass our defense." | Implemented for differentiable known transforms. The original released ATS policies include non-invertible and PIL/random operations, so a faithful generic inverse of policy `3-1-7` is not available and is not claimed. |
| Count-Sketch / FetchSGD | No original privacy attacker exists in FetchSGD; code explicitly labels this `not_published`. | `SketchGradientAttack` minimizes the sketch-space gradient discrepancy with known `rows`, `columns`, and `seed`; if hashes are withheld it returns an explicit unavailable result. | Song et al. (arXiv:2210.08371), §7.2, Eq. (7.2): "L_R(x) := ||R(∇_wF(w,x)) − R(g)||²." The same section: "It is reasonable to assume the attacker has access to R." | Implemented for the fixed-hash Count-Sketch map used by `CountSketchDefense`; known- and unknown-hash cases are separate. |

The non-adaptive base follows Zhu et al. (NeurIPS 2019), Eq. (4): "min_{x',y'} ||∇_W L(F(x',W),y') − ∇_W L(F(x,W),y)||²." For controlled tests it fixes a known label rather than adding the paper's dummy-label optimization.

## Tests

Command:

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  external_defenses/.venv/bin/python -m unittest external_defenses.tests.test_adaptive -v
```

Latest successful run: eight tests passed in 1.572 s (`logs/adaptive_unittest.log`). The test suite includes a lossless/identity sanity check for every adaptive attack: omit-layer, mask-aware pruning, known-transform ATS, joint stochastic noise, and known-hash sketch. Each must recover a one-record linear input to maximum absolute error below `1e-3` when the defense transform/noise is switched off. This guards against a weak attacker being mistaken for an effective defense.

Covered checks:

- Identity paths for all five defenses, including `CountSketchDefense(rows=0)` as the explicit test-only undefended path.
- Exact recovery to `1e-3` maximum absolute error on a one-record, one-step undefended linear MSE model for `GradientMatchingAttack` and known-hash `SketchGradientAttack`.
- Deterministic output across repeated seeded mask-aware attack executions.
- Pruning determinism and lowest-coordinate masking.
- Explicit unavailable state when Count-Sketch hashes are not disclosed.

## Updated Smoke Test

Command (run with the supplied datasets read-only):

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 \
  external_defenses/.venv/bin/python -m external_defenses.adaptive.smoke
```

This is qualitative only: random, untrained models; four records; 20 Adam steps; and a recovery call is counted only when its raw-input MSE is lower than **every** named baseline. The tabular source is four real, loader-preprocessed PaySim test records from `data.load_creditcard.load_paysim_splits(max_rows=5000)`; it has 13 features. The CNN source is CIFAR-10 test records 0–3 read from `datasets/cifar10/`, `download=False`. Tabular baseline is the pooled PaySim training-population mean. CIFAR baselines are constant gray (MSE 0.068134) and the CIFAR test-set mean image (MSE 0.066099).

| Defense | PaySim undefended positive control | PaySim original-style blocked? | PaySim adaptive succeeds? | CIFAR undefended positive control | CIFAR original-style blocked? | CIFAR adaptive succeeds? |
|---|---|---|---|---|---|---|
| Soteria | Yes, MSE 1.841402 < mean 7.741337 | No, MSE 1.896152 also beats mean | Yes, omit-layer MSE 1.216818 | No, MSE 0.980882 exceeds both baselines | Inconclusive: positive control failed; observed MSE 0.998067 | Inconclusive/no qualitative success; omit-layer MSE 0.919715 |
| PRECODE adapter | Yes | No, MSE 1.841401 beats mean | Yes, omit-gradient MSE 1.216818 | No | Inconclusive; MSE 0.980882 | Inconclusive/no qualitative success; omit-gradient MSE 0.919715 |
| Gradient pruning 50% | Yes | No, MSE 2.067864 beats mean | Yes, mask-aware MSE 2.025366 | No | Inconclusive; MSE 1.428042 | Inconclusive/no qualitative success; mask-aware MSE 1.115970 |
| ATS affine smoke policy | Yes | No, MSE 1.389466 beats mean | Yes, known-transform/raw-input MSE 1.648360 | No | Inconclusive; MSE 0.983516 | Inconclusive/no qualitative success; MSE 1.146268 |
| Count-Sketch | Yes | Not published | Yes with known hashes, MSE 1.656393; unavailable with unknown hashes | No | Not published | Inconclusive/no qualitative success with known hashes, MSE 1.220915; explicitly unavailable without hashes |

The complete machine-readable output is `logs/adaptive_smoke.log`. These values neither reproduce published attacks nor establish privacy: the CNN control failure specifically prevents interpreting the apparent CNN "blocked" outcomes as defense effects.

## Deviations and limitations

- The base gradient matcher fixes labels at attacker construction; this is a controlled positive-control simplification, not Zhu et al.'s joint label optimization.
- Soteria's representation sensitivity is client-side model/Jacobian work, so `SoteriaDefense` receives scores instead of deriving them from an update dict.
- PRECODE's actual variational bottleneck is not created by the update adapter. Faithful joint stochastic-noise optimization requires a compatible model hook; no generic hook exists in the harness.
- ATS's paper evaluates PSNR against transformed inputs. `KnownTransformAttack` instead returns the raw dummy input and requires a differentiable known policy; it cannot invert arbitrary random/non-bijective released PIL transforms.
- FetchSGD uses an external `csvec` implementation with unspecified hash synchronization. `CountSketchDefense` implements an explicit, deterministic fixed-hash Count-Sketch rather than claiming byte-for-byte equivalence to that unpinned dependency.
- `PRECODEDefense` is an update adapter, so this smoke does not instantiate the actual variational bottleneck. Its tabular/CNN PRECODE rows verify the harness attack path only; they must not be interpreted as a PRECODE privacy evaluation.
