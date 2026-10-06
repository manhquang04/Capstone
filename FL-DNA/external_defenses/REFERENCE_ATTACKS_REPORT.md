# Official reference gradient-inversion controls

**Outcome:** both official attackers work in their native published settings. Geiping's attack reaches **24.76 dB / SSIM 0.861** on one CIFAR LeNet-Zhu instance; TabLeak reaches **95.54%** on one Adult batch. On the actual project LeNet, the reference succeeds on two additional targets where the matched local attack fails, but fails on test index 0. On the real PaySim/project model, an explicitly adapted deterministic TabLeak instrument reaches **76.39%**, above measured gradient-free controls; the project's stochastic training setting remains unqualified. Confidence is high in these saved single-instance measurements, limited in transfer to other targets/model states.

## Scope and deliverable checklist

- [x] Pin official image and tabular repositories.
- [x] Run one undefended CIFAR-10 single-image instance of a published configuration on CPU; compare raw PSNR/SSIM with gray and CIFAR mean baselines.
- [x] Run one native Adult TabLeak instance and its accuracy metric.
- [x] Import the actual project LeNet read-only and attack one SGD update at learning rate 0.01, batch one, known label.
- [x] Apply TabLeak to loader-provided PaySim and document its compatibility limitations.
- [x] Inspect the specified project image attackers and record concrete differences.
- [x] Integrate tabular results, recommend instruments, and verify final saved metrics.

The evaluation domain is undefended updates, not defense efficacy. Primary image checks use CIFAR-10 test index 0 (known label 3), model seed 42, batch one. Supplementary transfer checks use test indices 1 and 2 (label 8), preserving the failed index-0 result. A working image control must have lower raw-input MSE (equivalently higher unit-range PSNR) than both gray and the CIFAR training mean image. SSIM is additionally reported with scikit-image, channel axis 2 and data range 1. All results are single-instance qualitative controls, not estimates of general attack success. Tabular scope is batch eight, known labels, initialized networks: one native Adult batch and one real PaySim benign-transaction batch. No fraud-positive PaySim case, trained checkpoint, full local-training update or later temporal regime is validated.

## Commits and environment

| Repository | Commit | Local checkout |
|---|---|---|
| [JonasGeiping/invertinggradients](https://github.com/JonasGeiping/invertinggradients) | `1157b61c6704df42c497ab9eb074c75da5204334` | `external_defenses/invertinggradients/` |
| [eth-sri/tableak](https://github.com/eth-sri/tableak) | `6b8c1c82ffe85bc6fdd96938ac42b6c9b5685300` | `external_defenses/tableak/` |

Python is `external_defenses/.venv/bin/python`, torch 2.5.1, CPU. Image and tabular branches each use one process with two compute threads. Project imports run with `PYTHONDONTWRITEBYTECODE=1`; data loaders have no child workers. Only `external_defenses/` is writable. CIFAR is loaded read-only with `download=False`; its data are not copied or downloaded.

## Image reference: published configuration versus measured control

The primary configuration is the paper's untrained **LeNet (Zhu)** baseline, not the repository's different generic `DEFAULT_CONFIG`. Appendix C.1 and Table 4 prescribe cosine loss, signed-image-gradient Adam at learning rate 0.1, 4,800 iterations, one restart, TV 0.01 for the untrained LeNet. Learning rate decays by 0.1 after approximately 3/8, 5/8 and 7/8 of the budget. Optimization is directly in normalized image coordinates with box projection. The official `GradientReconstructor` and `construct_model('LeNetZhu', seed=42)` run unchanged; CIFAR channel normalization uses official constants. No augmentation or training is performed.

[Geiping et al., arXiv:2003.14053](https://arxiv.org/abs/2003.14053), Table 1 reports **18.00 ± 3.33 dB** (mean ± SD over the first 100 CIFAR validation images) for this proposed attack/untrained LeNet setting. Our first-image result is **24.758866 dB**, SSIM **0.860512**, raw MSE **0.003342823**. This reproduces a published *setting and method on one image*, not the 100-image aggregate. The paper does not publish a matching SSIM target or this seed-specific individual PSNR; no numerical SSIM agreement claim is possible.

| Reconstruction, same CIFAR image | Raw-input MSE | PSNR (dB) | SSIM |
|---|---:|---:|---:|
| Official paper-setting LeNet-Zhu | 0.003342823 | 24.758866 | 0.860512 |
| Constant gray 0.5 | 0.039078020 | 14.080675 | 0.085711 |
| CIFAR training mean image | 0.039032694 | 14.085715 | 0.085193 |

The attack clearly beats both baselines. Output: `external_defenses/reference_image/published/{metrics.json,reconstruction.pt,reconstruction.png}`. The model has 15,826 parameters, sigmoid activations, and no BatchNorm or pooling.

## Project image setting and paired local control

`LeNetCIFAR` and `_named_update` are imported from `experiments/run_priority7_image_domain_gate.py`. The actual model has 136,886 parameters, ReLU and max pooling, no BatchNorm, and is untrained. The observed one-step SGD delta is generated at learning rate 0.01 and converted to a gradient via `g = delta / -0.01`; this sign is essential for cosine matching. In this network train/eval forward behavior is identical because it has no BatchNorm or dropout.

The paired local run imports the actual `optimize_one` used by Priority 28 from `experiments/priority23_repaired_cifar_attack.py`, with 3,000 iterations, Adam 0.05, TV 1e-4, seed 42, one restart, exactly the same model/image/update. It obtains **12.549776 dB**, SSIM **0.233039**, MSE **0.055593293**, failing both baselines. The primary official configuration obtains **13.291390 dB**, SSIM **0.388979**, MSE **0.046866339**, also failing both MSE baselines. Better structural similarity alone does not satisfy the positive control.

| Project run | Iterations × restarts | TV | MSE | PSNR (dB) | SSIM | Beats both MSE baselines? |
|---|---|---:|---:|---:|---:|---|
| Actual local Priority-28 optimizer | 3,000 × 1 | 1e-4 | 0.055593293 | 12.549776 | 0.233039 | No |
| Official paper configuration | 4,800 × 1 | 0.01 | 0.046866339 | 13.291390 | 0.388979 | No |
| Official extended budget | 24,000 × 4 | 1e-4 | 0.078263484 | 11.064408 | 0.203842 | No |
| Official low TV | 4,800 × 4 | 1e-6 | 0.073931336 | 11.311714 | 0.185825 | No |
| Official generic code defaults (unsigned) | 4,800 × 1 | 0.1 | 0.054120284 | 12.666399 | 0.215852 | No |
| Official stronger TV (signed) | 4,800 × 4 | 0.05 | 0.050675780 | 12.951996 | 0.304058 | No |

Lower gradient discrepancy did not imply lower raw-image error: the extended-budget result's cosine discrepancy is about 0.000110, yet its PSNR is worse than gray. Official restart selection uses gradient discrepancy, not ground-truth MSE. We preserve every result; a longer budget is not treated as an automatic improvement.

### Transfer to two additional project targets

The same official **paper configuration**, without retuning, works on CIFAR test images 1 and 2 in the actual project LeNet/SGD setting. Matching local runs use the same model seed, image, known label and SGD delta. This is a target-sensitivity check after index 0 failed, not a held-out population performance estimate or a replacement of that failure.

| Test index | Official MSE / PSNR / SSIM | Local MSE / PSNR / SSIM | Gray MSE / PSNR | Mean MSE / PSNR |
|---|---|---|---|---|
| 1 | 0.041033 / 13.868684 / 0.430913 | 0.169671 / 7.703925 / 0.159818 | 0.106617 / 9.721720 | 0.102708 / 9.883967 |
| 2 | 0.040497 / 13.925723 / 0.367982 | 0.118409 / 9.266159 / 0.087282 | 0.066448 / 11.775191 | 0.061724 / 12.095449 |

PSNR is in dB. The official attack beats both baselines on both targets; the local attack fails both. Thus a working project-setting reference exists, but it is not reliable on every image. Outputs: `reference_image/project_index{1,2}/` and `reference_image/project_local_index{1,2}/` under `external_defenses/`.

## Concrete image-attack differences

| Component | Official paper-setting attack | Priority 28 / Priority 23 implementation | Priority 7 implementation | Interpretation |
|---|---|---|---|---|
| Optimizer / learning rate | Adam 0.1 | Adam 0.05 | Adam, job-supplied default 0.05 | Both already use Adam; optimizer family alone is not the explanation. |
| Schedule | MultiStepLR, factor 0.1 at ~3/8, 5/8, 7/8 | Constant learning rate | Constant learning rate | Local code lacks late-stage refinement schedule. |
| Signed gradients | Image-variable gradient replaced with its sign before Adam | Unsigned latent gradient | Unsigned latent gradient | Changes Adam's moment estimates. |
| Image parameterization | Direct image coordinates, Gaussian initialization, projection to valid normalized pixel box | Gaussian logits then sigmoid to [0,1] | Same sigmoid-logit representation | Sigmoid Jacobian can attenuate updates near saturated pixels; this is a plausible optimization limitation, not an isolated causal finding. |
| Objective | Global cosine of all parameter gradients + TV | Global cosine of update dictionaries + TV | `GEN_COSINE_TV` branch uses cosine alone | The Priority-7 generation name includes TV, but `_image_job` never adds a TV term. Priority 28 does include TV. |
| TV | 0.01 in untrained LeNet-Zhu paper baseline | 1e-4 (Priority-28 CLI default) | None in `_image_job` | TV and input normalization change the smoothing scale. |
| Iterations | 4,800 | Priority-28 default 3,000 | Default 300 | Budget differences exist, but longer runs here did not fix raw-image error. |
| Augmentation | None during reconstruction/control | None | None | Not an explanation for these untrained controls. |
| Restarts / selection | Paper baseline one; final gradient-only discrepancy selects restart | Priority-28 default four; best cosine+TV objective and best iterate | Default four; best objective / iterate | Paired local run uses one restart; does not replicate the user's historical four-restart population. |
| Model mode | `reconstruct(..., eval=True)` forces eval | Model created in train | Model created in train | No effect for project LeNet without BatchNorm/dropout; cannot explain this failure. |
| BatchNorm | Absent in LeNet-Zhu, present in official ConvNet (not used) | Absent in project LeNet | Absent in project LeNet | No BN mismatch in these image controls. |
| Activation / pooling | LeNet-Zhu: sigmoid convs, no pooling, uniform parameter initialization | Project LeNet: ReLU, max pooling, standard torch initialization | Same project LeNet | Different gradient geometry and non-smoothness make transferring a published hyperparameter choice nontrivial. |

Source locations: official `inversefed/reconstruction_algorithms.py:13–26,130–207` and `inversefed/nn/models.py:300–325`; project Priority 7 `:78–111,303–345`; Priority 23 `:111–159`; Priority 28 `:83–129,181–191`. These are observed implementation differences. Their individual causal effects have not been ablated; it would be incorrect to claim they establish why the historical attack failed.

## Tabular reference and PaySim

### Published Adult setting

[Vero et al., TabLeak, ICML 2023 / arXiv:2210.01785](https://arxiv.org/abs/2210.01785), Table 1 reports **95.2 ± 8.8%** accuracy (mean ± SD of 50 random batches) for Adult, batch eight, true labels. Our one-batch measurement is **95.535714%**, categorical **98.437500%**, continuous **91.666667%**. This is one instance of the published protocol, not a reproduction of its aggregate SD or mean. Genuine Adult files came with the official clone; no substitute dataset was used.

The driver reads official configuration **46** from `tableak/run_inversion_attacks.py` using AST. It runs official `invert_grad` with an initialized 105→100→100→2 ReLU model, cross-entropy, cosine discrepancy, signed Adam 0.06, **1,500 iterations × 30 restart/ensemble members**, uniform initialization, categorical softmax, continuous sigmoid bounds, median+softmax pooling, and no learning-rate schedule. Restart alignment and selection use gradients, never target accuracy (`perfect_pooling=False`); the attacker receives an empty tensor for shape, not target feature values.

The official accuracy scores 14 *mixed* Adult features, not 105 encoded coordinates. Categorical correctness is exact; numeric correctness is inclusive ±0.319 SD. Official integer decode, continuous clipping and Hungarian row alignment are applied. For Adult, SD/standardization use the official concatenated train/test convention. Our measured mean/mode baseline scores **59.821429%**; a separate empirical-marginal baseline over 30 random guesses scores **56.428571 ± 2.395787%**. These measured controls use this exact batch; the paper's random baseline is **53.9 ± 4.4%** over its 50 batches. TabLeak clearly beats these controls.

### PaySim transfer and retained adaptation

The read-only project loader `data/load_creditcard.py` supplies a stratified seed-42 **100,000-row** sample, train-only RobustScaler and one-hot type encoding. Eight records are sampled from client 0; they contain **zero fraud-positive labels**. The adapter preserves those model-input coordinates, uses train-only bounds and raw SD, decodes continuous raw money without Adult's integer rounding, and scores nine mixed features (eight numeric/derived features plus transaction type), not 13 one-hot coordinates. Auxiliary training statistics/bounds are assumed known. This is explicit schema adaptation of the official metric and attack.

| Scenario | Combined accuracy (%) | Category (%) | Continuous (%) |
|---|---:|---:|---:|
| Adult, native defaults | 95.535714 | 98.437500 | 91.666667 |
| PaySim, official FC, default sigmoid | 25.000000 | 62.500000 | 20.312500 |
| PaySim, actual project `FraudMLP`, eval, default sigmoid | 34.722222 | 62.500000 | 31.250000 |
| PaySim, official FC, sigmoid disabled | 83.333333 | 87.500000 | 82.812500 |
| PaySim, actual project `FraudMLP`, eval, sigmoid disabled | 76.388889 | 87.500000 | 75.000000 |
| PaySim, data-free training mean + categorical mode | 56.944444 | 0.000000 | 64.062500 |
| PaySim, independent empirical marginals (30 guesses, mean ± SD) | 66.527778 ± 2.833493 | — | — |

Both PaySim no-sigmoid variants beat the measured controls. Defaults fail them. The improved project result is the same saved batch/checkpoint and criterion; restart random streams are not paired between default and no-sigmoid runs. This establishes a useful configuration and supports poor conditioning of wide-bound sigmoid variables as a hypothesis, not an isolated causal proof.

The actual `FraudMLP` is imported unchanged: 13→128→64→32→1 with BatchNorm and dropout 0.3/0.2/0.1. The instrument uses **eval mode and weighted BCE**, not default focal loss or actual train-mode local epochs. Five same-input train-mode gradient replays have cosine similarities **0.378546, 0.244517, 0.391045, 0.373063, 0.316194** to the first gradient. Eval-mode same-input cosine is **1.0**, maximum gradient difference **0.0**. This is direct evidence that a deterministic gradient objective does not match the stochastic observation even at the true input, unless dropout realization and BN/state semantics are controlled or modeled.

Tolerance accuracy is not exact transaction reconstruction: improved project-eval **amount MAE = 235,700.974931 simulated monetary units**, origin/destination derived-balance consistency MAE = **76,645.711688 / 581,166.970437**. Heavy-tailed SD tolerances are permissive, and independently optimized derived features can violate accounting identities. In particular, amount-feature correctness is only **62.5%**, below the mean baseline's **87.5%**, despite better overall/category accuracy. No raw-money precision claim is warranted.

### Concrete tabular differences and diagnosis

| Component | Official/native TabLeak | Project / transfer implication |
|---|---|---|
| Objective and optimizer | Cosine, sign-Adam 0.06, 1,500 steps, no schedule | Native success demonstrates the official implementation works; image TV is not a tabular prior. |
| Restart/ensemble | 30 aligned reconstructions, median+softmax pooling | A single generic matching run omits a central published component. |
| Structured variables | Categorical softmax, bounded sigmoid continuous variables | PaySim RobustScaler and extreme monetary ranges make default sigmoid transfer poor here; disabling it improves overall score. |
| Scaling and metrics | Adult mean/SD, integer numerical decode | PaySim median/IQR, continuous raw-money decode, train-only SD tolerance; copying Adult preprocessing/rounding would change the problem. |
| Network and activations | ReLU FC, no BN/dropout, two output logits | Project ReLU FC includes BN/dropout and one output logit; train-mode randomness gives measured same-input gradient mismatch. |
| Loss | Cross-entropy | Project defaults to focal loss; measured project instrument uses optional weighted BCE explicitly. A mismatched criterion is not the same inverse problem. |
| Observation | One FedSGD minibatch gradient at initialization | Project normally trains in train mode, Adam 0.001, local epochs and batch 1,024. Full multi-step update inversion is not validated by a raw-gradient control. |
| Server information | Known labels, dataset feature structure/statistics | These are supplied for the controlled PaySim run; unknown-label/statistics reconstruction is not measured. |

Source details: `external_defenses/tableak/run_inversion_attacks.py:243–264,377–383`, `tableak/attacks/gradient_inversion_attack.py:65–143,666–836`, `tableak/utils/matching.py:29–50`, `models/fraud_mlp.py:14–36`, `experiments/fraud_fl_common.py:33–38,51–69,82–116`, and `data/load_creditcard.py:24–43,263–310`. The detailed tabular source-location table is in `external_defenses/reference_tabular/tabular_reference_notes.md`.

### Compatibility and blocked/unrun components

- The official TabLeak source remains unchanged. PyTorch 2.5 rejects an official leaf-tensor in-place slice update. A driver-only wrapper passes `x.clone()` to the same official sigmoid transform; it preserves values/autograd and avoids the API error. The macOS driver bypasses Linux `taskset`/multiprocessing, using one process. Preserved failures: `failed_preflight.log` (JSON float32 serialization), `failed_torch_leaf.log` (leaf in-place operation), and earlier run directories.
- **Blocked/unrun: matching the project's stochastic train-mode TabLeak control.** Deterministic replay already fails at the true input; exact masks/BN semantics or a validated stochastic attack are needed. The requested real PaySim application *did run*, but its working result is the explicitly deterministic eval/BCE variant.
- **Unrun:** project default focal-loss inversion, trained checkpoints, full train-mode FedAvg/local-epoch updates, fraud-positive PaySim targets, population/temporal generalization, and the papers' 50/100-instance aggregate reproductions. They are outside the measured single-instance controls and have no reported reconstruction numbers.

## Harness recommendation

1. **Images:** use pinned official `inversefed.GradientReconstructor`, with the paper-setting signed cosine+TV configuration, correct `-delta/lr` conversion, direct image parameterization and the decay schedule. `reference_image.py` is the working CPU instrument/driver. Keep the native LeNet-Zhu control as an implementation check; qualify it separately on each project image/model population. The actual project LeNet succeeds on indices 1 and 2 but fails index 0, so it is not yet a universally qualified image instrument. Never treat a failed positive control as defense protection.
2. **Tabular:** use pinned official TabLeak `invert_grad`, its structured categorical representation and full 30-member ensemble. Keep native Adult as an implementation check. For the measured PaySim deterministic `FraudMLP` instrument, retain **sigmoid disabled, eval mode, weighted BCE**, using the project-preserving adapter in `reference_tabular/run_reference.py` and the saved-batch adaptation in `run_paysim_ablation.py`. This wins on overall tolerance accuracy but not exact monetary recovery. Qualify additional fraud/temporal cases and the actual training observation before using it to estimate project defense efficacy.
3. **Failure diagnosis:** image failures are reproducible in the actual local optimizer and improve on matched targets using the reference implementation, but this does not isolate schedule/sign/parameterization/TV individually. Tabular train-mode randomness has a directly measured forward-gradient mismatch; wide-bound conditioning is supported by a successful configuration change with an RNG-pairing caveat. These are concrete evidence-backed limitations, not a claim that every historical failure has one known cause.

## Reproducibility and verification

Image producer: `external_defenses/reference_image.py`. From FL-DNA:

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
external_defenses/.venv/bin/python -B external_defenses/reference_image.py --setting published
# Same command prefix, then:
# --setting project
# --setting project_local
# --setting project --tv 0.0001 --iterations 24000 --restarts 4 --output-name project_tv1e4_24k_r4
# --setting project --tv 0.000001 --restarts 4 --output-name project_tv1e6_r4
# --setting project --tv 0.1 --unsigned --output-name project_code_defaults
# --setting project --tv 0.05 --restarts 4 --output-name project_tv005_r4
# --setting project --index 1 --output-name project_index1
# --setting project --index 2 --output-name project_index2
# --setting project_local --index 1 --output-name project_local_index1
# --setting project_local --index 2 --output-name project_local_index2
```

The script refuses to overwrite existing `metrics.json`; use `--output-name` with a new name on a rerun. `project_local/metrics.json` from the first producer version mistakenly lists the official config dictionary, although it ran the local 3,000-step implementation. The source was corrected and the local run repeated end-to-end in `project_local_corrected/`; all metrics are identical and its configuration metadata is correct. The historical file is preserved. Runtime of the first paper-setting inversion was 8.54 seconds; project primary 16.13 seconds; extended-budget/four-restart inversion 275.10 seconds, on two threads. Compute jobs `7e968751-fe8`, `8c612da3-1ed` and `e9592e2e-e4b` exited 0.

Independent saved-tensor metric recomputation: `external_defenses/verify_reference_image.py` exited 0; it compared scikit-image PSNR, torch raw MSE, SSIM, and official unit-range PSNR against all twelve saved outputs (including the preserved local metadata version). Output is `external_defenses/reference_image/verification.json`. PSNR differences between independent implementations are below 1e-5 dB, raw-MSE differences below 1e-7. The primary reconstruction PNG and the project/local PNGs were also inspected directly.

Tabular producer/verification commands, from FL-DNA (one process, two threads; use a new run name to preserve existing results):

```bash
export PYTHONDONTWRITEBYTECODE=1 DATALOADER_NUM_WORKERS=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
external_defenses/.venv/bin/python -B external_defenses/reference_tabular/run_reference.py --run-name default_batch8_seed42_v3
external_defenses/.venv/bin/python -B external_defenses/reference_tabular/run_paysim_ablation.py
external_defenses/.venv/bin/python -B external_defenses/reference_tabular/verify_reference.py
external_defenses/.venv/bin/python -B external_defenses/reference_tabular/verify_reference.py --run-name paysim_no_sigmoid --scenarios paysim_official_fc paysim_project_eval --output-name verification_ablation.json
external_defenses/.venv/bin/python -B external_defenses/reference_tabular/measure_baselines.py
external_defenses/.venv/bin/python -B external_defenses/verify_reference_image.py
```

All final verification commands were run by the lead in fresh processes, exit 0. Tabular verification checks finite arrays, eight records, 30 ensemble members, saved configurations, official accuracy, independently constructed feature-cost/Hungarian accuracy, input/model-source hashes and the unchanged official tracked source. Results are in `reference_tabular/verification{,_ablation}.json` and `baselines.json`. Default tabular job `7d8cdb2e-8df` and adapted PaySim job `045d7775-d16` exited 0. Adult inversion took 87.28 s; default project PaySim 91.14 s; adapted project PaySim 82.42 s. Exact inputs, labels, model weights, adapter schema/statistics, reconstruction arrays and run parameters are saved in `reference_tabular/runs/`. The tabular outputs manifest records branch file hashes; the measured baseline is additionally preserved in `baselines.json`.

**Bottom line:** adopt the official instruments with explicit native and project positive-control gates. Both native controls succeeded; project transfer works in the measured image examples and deterministic PaySim setting, with the failed image and stochastic-training limitation retained rather than hidden.
