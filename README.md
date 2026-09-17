# FL-DNA: Evaluating a DNA-Inspired Update Transform for Federated Fraud Detection

This repository holds the code and result data for a capstone project that tests whether a DNA-inspired transform on federated learning (FL) updates can substitute for differential privacy (DP) in a fraud detection setting. Short answer, after several months of experiments: no. The transform keeps model utility intact but does not resist a properly matched reconstruction attack, and its transport cost does not clear a realistic deployment budget either. The rest of this document explains how we got there and how to check the numbers yourself.

## What this project actually asks

Federated learning is often pitched as privacy-friendly because raw data never leaves the client. That claim only holds if the *update* a client sends is also safe, and gradient inversion attacks exist precisely because it often isn't. This project evaluates a proposed defense — a DNA-sequence-inspired transform applied to the update before it is sent — against three questions that were fixed before any confirmatory experiment ran:

1. Does the transform reduce gradient-inversion reconstruction quality more than a distortion- or utility-matched DP baseline?
2. Does it preserve fraud-detection utility (F1, AUC-ROC), tested as non-inferiority against an unprotected FL run?
3. Is its computational and bandwidth overhead acceptable for deployment?

We ended up testing two structurally different versions of the transform:

- **v1** — a DNA-sequence-seeded permutation, selective attenuation, and residual mixing applied block-wise to the flattened update. Configs: conservative, medium, stronger.
- **v2** — a subsampled randomized Hadamard sketch followed by scalar quantization (a Johnson–Lindenstrauss-style projection, lossy by construction). Configs: compression ratio 0.95 and 0.90, both at quantization step 0.01.

Both are deterministic functions of a secret seed, not randomized mechanisms with independent noise. That distinction turned out to matter a lot — see the theory note below.

## How the evaluation is structured

Every claim in `results/` traces back to a pre-registered protocol: a frozen statistical test, a frozen sample-size calculation, and a data firewall that keeps development, pilot, and confirmatory target sets from overlapping. We adopted this discipline the hard way. Early in RQ1-v2 development, an attacker variant looked strong at a 24-sample pilot (20/24 wins, p = 0.00077), which under a naive read would have counted as a finding. Running the pre-registered confirmatory sample at n = 176 instead gave 85/176 wins, p = 0.70 — not significant. That single episode is why nothing in this repo gets reported as a result until it clears an independently powered confirmatory sample, not just a promising pilot.

The gate used throughout RQ1 is an exact one-sided sign test comparing reconstruction MSE under the transformed update against two controls (a plausible prior guess and an all-zero guess), with p1 = 0.70, alpha = 0.05, power = 0.80 fixed once and never re-fit from observed data.

## Results, in order

**RQ1 — does the transform beat DP at resisting reconstruction?** No. Against a distortion-matched DP baseline, v1 shows no significant advantage (25/44 wins, p = 0.226). More importantly, once we stopped using a custom-built attacker and instead re-implemented published loss objectives — iDLG-style label fixing and Geiping-style cosine-similarity matching — both transform versions were broken outright, at confirmatory sample sizes:

| Setting | Attacker | n | Result |
|---|---|---|---|
| PaySim, v1-medium | GEN_IDLG_STYLE | 39 | 29/39 vs prior (p=0.0017), 32/39 vs zero (p=3.5e-5) |
| IEEE-CIS, v2 (ratio 0.95) | GEN_COSINE_TV | 39 | 36/39 both controls, p=1.8e-8 |
| CIFAR-10, v2 (ratio 0.95) | GEN_COSINE_TV | 39 | 39/39 both controls, p=1.8e-12 |

That last row is the one worth pausing on. The same attack objective breaks v2 on tabular gradients (PaySim, IEEE-CIS) and on CNN gradients trained on CIFAR-10 — different data, different model, different loss surface, same failure. We also checked whether the break holds across each transform's own parameter range: v1 stays broken from medium through stronger (only the weakest, conservative, config survives at n=8), while v2's break reverses at a more aggressive compression ratio (0.90 resists at n=8, though we're careful not to call that "private" — it's an untested boundary, not a confirmed safe zone).

There's a structural reason this isn't a coincidence. Both transforms are deterministic given a seed, and a deterministic mechanism with distinct outputs for distinct inputs cannot satisfy (ε, δ)-DP for any finite ε once the seed is known — the probability mass either lands on one output or it doesn't, there's no distribution to bound. Whatever protection these transforms offer at the seed-unknown level is closer to obscurity than to a calibrated guarantee, which is exactly what the empirical breaks show once the attacker stops guessing and starts matching the mechanism's structure.

**RQ2 — does it keep the model useful?** Yes, on both versions. v1-medium passes non-inferiority for F1 and AUC-ROC against the unprotected baseline. v2's confirmatory run (n=52, after a pre-registered amendment raised the seed cap to accommodate its higher variance) also clears both margins: F1 delta −0.0117 (margin −0.02), AUC-ROC delta −0.0020 (margin −0.005).

**RQ3 — is it cheap enough to deploy?** No, for either version, and this isn't a code-quality problem. We profiled the reference implementation, found the actual bottlenecks (string-based DNA sequence handling for v1, a per-block Python loop in the Hadamard transform for v2), rewrote both paths with vectorized operations, and cut CPU time by 5–9x. The optimized implementation still fails the pre-registered cost budget — now on end-to-end round overhead and peak memory rather than raw compute — which points to a structural cost in the transport design, not a slow implementation.

## What's in the repository

Only code and curated result summaries are pushed here; raw per-restart artifacts (model checkpoints, per-target `.pt` files) and internal protocol/report drafts stay local since they add bulk without adding anything checkable.

```text
FL-DNA/
  dna_encoder/           transform_defense.py (v1), transform_defense_v2.py (v2)
  attacks/               gradient inversion attackers, including the literature-
                         matched generations (GEN_IDLG_STYLE, GEN_COSINE_TV) and
                         the tabular/image reconstruction metric adapters
  experiments/           runners and analyzers for every RQ1/RQ2/RQ3 evaluation,
                         the strong-DP epsilon calibration, and the RQ3 benchmark
  tests/                 unit tests for the v2 transform and metric adapters
  results/
    rq1/, rq1_v2/                        distortion-/utility-matched DP comparisons
    rq2_v2/                              v2 utility non-inferiority confirmatory
    rq3/                                 reference and optimized cost benchmarks
    strong_update_dp/                    genuinely strong DP: utility ceiling result
    dna_transform_v2/                    v2 development-stage utility smoke test
    priority6_9_literature_attackers/    published-attacker confirmatory results
                                         (PaySim, IEEE-CIS, CIFAR-10) and the
                                         parameter-sensitivity boundary checks
```

## Running it yourself

Everything runs single-threaded (`torch.set_num_threads(1)`) for reproducibility; this was kept fixed across every experiment in this repo and should not be changed when re-running anything here.

```bash
python -m venv .venv
pip install -r requirements.txt
```

Each script under `experiments/` is self-contained and takes a config path — see the `run_*` and matching `analyze_*` pairs. `run_priority6_sota_style_attackers.py` and `run_priority7_image_domain_gate.py` are the ones behind the literature-attacker results above; `run_rq3_benchmark.py` is the one behind the cost tables.

The dataset is expected at `FL-DNA/datasets/creditcard.csv` (a PaySim-style mobile-money fraud dataset — `step, type, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, isFraud`) and, for the generalization tests, the IEEE-CIS fraud detection dataset and CIFAR-10 (the latter downloads automatically via torchvision).

## Honest caveats

RQ1's non-significant cells (v1-conservative, v2 at ratio 0.90) were only tested at n=8. Given the IHT episode above, that's absence of evidence, not evidence of a safe configuration — a stronger or better-matched attacker might well break them too. The cross-domain CIFAR-10 test uses a deliberately minimized CNN chosen so second-order gradient inversion stays tractable single-threaded; nothing here says the result holds for larger vision models. And the theoretical argument about seed-keyed determinism applies to the seed-known case — it does not by itself derive a success rate for an attacker who doesn't know the seed, which is why that half of the story still had to be settled empirically.
