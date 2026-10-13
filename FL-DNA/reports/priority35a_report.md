# Priority35A: Config-C record recovery evaluation

Date: 2026-10-10. Final negative qualification result with a separately authorized post-hoc descriptive diagnostic.

## Outcome

**0/72 cells qualified.** All failed at n8; fresh n24 and n39 were NOT_RUN. No confirmatory, v1/v2 or DP measurements were made.
No qualified instrument recovered records from realistic Adam uploads even without a defense. This is a result about the tested instruments and budgets, not evidence of privacy, immunity, or transform superiority over DP.

## Design, knowledge and metrics

Config-C is P34A FedBN: K=3, Adam lr .001, binary focal alpha .95/gamma 2; only eight non-BN trainable tensor differences upload. BAF (primary), IEEE-CIS, PaySim; seed 321000 rounds 10/30. All three round-50 replays are bit-exact against historical P34A. Client BN affine/statistics and dropout masks are unavailable in deployable Config-C. BN stays train mode; the qualification surrogate optimizes free default BN affine and replaces only dropout with identity. Labels given means unordered multiset; labels estimated enumerates every fraud count with 64 steps then two 4,000-step starts. TabLeak-style uses cosine plus .1 normalized L2; Wan update matching uses cosine only, both simulating fresh Adam exactly in float32.
b=1 is NOT_APPLICABLE: training-mode BN cannot process it. A no-BN synthetic ratio test is a correctness check only. No eval-BN or oracle-BN substitution occurred in qualification.
The experimental unit is one batch (b=4/8/16), not a row or coordinate. Qualification compares paired feature accuracies to mean/mode, empirical marginal and cyclic next-target decoy; exact one-sided sign p<.05 must hold for all three, at n8 and fresh n24. Ties use exact equality and are excluded only from the binomial count. No failed/missing/numerical targets were discarded. All 144 distinct original batches passed the P24–34 original-source-ID firewall.
Accuracy uses native TabLeak blind Hungarian alignment, .319 training SD (ddof1) continuous tolerance and exact category match. Its absent-feature-type denominator convention is retained. Standardized MSE uses the same alignment; zero SD fallback 1 applies only to MSE, while accuracy tolerance stays zero. Bounded score is min(MSE, mean/mode MSE). Frequency-encoded categories are evaluated in encoded coordinates, not recovered original categories. Mean/mode and empirical marginal are training-derived, update-free references, not literally distribution-free.

## Qualification table

W/L/T below is estimator minus reference accuracy; p is one-sided greater. Each entry has n=8 batches. All n24 entries are NOT_RUN because n8 failed at least one locked control.

| Cell | Mean/mode W/L/T; p | Marginal W/L/T; p | Decoy W/L/T; p | n24 | Decision |
|---|---|---|---|---|---|
| baf_r10_b4_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b4_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b4_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b4_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b8_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b8_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/2/2; 0.34375 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b8_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b8_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b16_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b16_given_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b16_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 4/3/1; 0.5 | NOT_RUN | NOT_QUALIFIED |
| baf_r10_b16_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b4_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b4_given_wan | 0/8/0; 1 | 0/8/0; 1 | 7/1/0; 0.03515625 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b4_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/0/2; 0.015625 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b4_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 7/1/0; 0.03515625 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b8_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 7/1/0; 0.03515625 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b8_given_wan | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b8_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b8_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b16_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b16_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b16_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| baf_r30_b16_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b4_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 7/1/0; 0.03515625 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b4_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b4_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b4_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b8_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b8_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b8_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/2/1; 0.2265625 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b8_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b16_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b16_given_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b16_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r10_b16_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b4_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b4_given_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b4_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b4_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 5/2/1; 0.2265625 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b8_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 6/2/0; 0.14453125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b8_given_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b8_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b8_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b16_given_tableak | 0/8/0; 1 | 0/8/0; 1 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b16_given_wan | 0/8/0; 1 | 0/8/0; 1 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b16_estimated_tableak | 0/8/0; 1 | 0/8/0; 1 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| ieee_cis_r30_b16_estimated_wan | 0/8/0; 1 | 0/8/0; 1 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b4_given_tableak | 4/3/1; 0.5 | 5/3/0; 0.36328125 | 4/2/2; 0.34375 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b4_given_wan | 2/6/0; 0.96484375 | 5/3/0; 0.36328125 | 5/2/1; 0.2265625 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b4_estimated_tableak | 5/3/0; 0.36328125 | 5/3/0; 0.36328125 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b4_estimated_wan | 1/5/2; 0.984375 | 3/3/2; 0.65625 | 6/1/1; 0.0625 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b8_given_tableak | 8/0/0; 0.00390625 | 5/2/1; 0.2265625 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b8_given_wan | 6/2/0; 0.14453125 | 4/4/0; 0.63671875 | 5/2/1; 0.2265625 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b8_estimated_tableak | 7/1/0; 0.03515625 | 5/3/0; 0.36328125 | 6/1/1; 0.0625 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b8_estimated_wan | 6/2/0; 0.14453125 | 1/6/1; 0.9921875 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b16_given_tableak | 8/0/0; 0.00390625 | 5/3/0; 0.36328125 | 4/2/2; 0.34375 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b16_given_wan | 6/2/0; 0.14453125 | 3/3/2; 0.65625 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b16_estimated_tableak | 5/3/0; 0.36328125 | 4/4/0; 0.63671875 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r10_b16_estimated_wan | 6/2/0; 0.14453125 | 3/5/0; 0.85546875 | 3/4/1; 0.7734375 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b4_given_tableak | 6/1/1; 0.0625 | 7/1/0; 0.03515625 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b4_given_wan | 1/7/0; 0.99609375 | 5/3/0; 0.36328125 | 4/2/2; 0.34375 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b4_estimated_tableak | 3/5/0; 0.85546875 | 5/2/1; 0.2265625 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b4_estimated_wan | 1/7/0; 0.99609375 | 4/3/1; 0.5 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b8_given_tableak | 6/1/1; 0.0625 | 6/2/0; 0.14453125 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b8_given_wan | 3/4/1; 0.7734375 | 4/3/1; 0.5 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b8_estimated_tableak | 5/2/1; 0.2265625 | 5/3/0; 0.36328125 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b8_estimated_wan | 3/3/2; 0.65625 | 2/6/0; 0.96484375 | 3/4/1; 0.7734375 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b16_given_tableak | 6/2/0; 0.14453125 | 3/3/2; 0.65625 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b16_given_wan | 7/1/0; 0.03515625 | 1/7/0; 0.99609375 | 3/5/0; 0.85546875 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b16_estimated_tableak | 6/2/0; 0.14453125 | 5/3/0; 0.36328125 | 5/3/0; 0.36328125 | NOT_RUN | NOT_QUALIFIED |
| paysim_r30_b16_estimated_wan | 5/3/0; 0.36328125 | 1/7/0; 0.99609375 | 4/4/0; 0.63671875 | NOT_RUN | NOT_QUALIFIED |

### Closest-to-qualifying evidence

Ranked by the worst of the three gate p values, then cell ID; this ordering is descriptive, not a retest.

- paysim_r10_b16_given_tableak: decoy W/L/T 4/2/2; 0.34375, median accuracy delta 0.0104166667; empirical_marginal W/L/T 5/3/0; 0.36328125, median accuracy delta 0.0277777778; mean_mode W/L/T 8/0/0; 0.00390625, median accuracy delta 0.0625
- paysim_r10_b8_estimated_tableak: decoy W/L/T 6/1/1; 0.0625, median accuracy delta 0.0208333333; empirical_marginal W/L/T 5/3/0; 0.36328125, median accuracy delta 0.0138888889; mean_mode W/L/T 7/1/0; 0.03515625, median accuracy delta 0.0555555556
- paysim_r30_b16_estimated_tableak: decoy W/L/T 5/3/0; 0.36328125, median accuracy delta 0.0173611111; empirical_marginal W/L/T 5/3/0; 0.36328125, median accuracy delta 0.00694444444; mean_mode W/L/T 6/2/0; 0.14453125, median accuracy delta 0.0381944444
- paysim_r30_b4_given_tableak: decoy W/L/T 5/3/0; 0.36328125, median accuracy delta 0.0555555556; empirical_marginal W/L/T 7/1/0; 0.03515625, median accuracy delta 0.111111111; mean_mode W/L/T 6/1/1; 0.0625, median accuracy delta 0.0555555556
- paysim_r30_b8_given_tableak: decoy W/L/T 5/3/0; 0.36328125, median accuracy delta 0.0208333333; empirical_marginal W/L/T 6/2/0; 0.14453125, median accuracy delta 0.0625; mean_mode W/L/T 6/1/1; 0.0625, median accuracy delta 0.0763888889

## Post-hoc descriptive diagnostic (not qualification)

The dated addendum was written and frozen before D1/D2. Six cells use the exact same eight n8 batches, b=4, round10, labels given and the same public initialization seeds, two starts and 4,000 iterations/start. D1 observes plain non-BN gradients with true local BN supplied; D2 observes the unchanged Adam upload with true local BN supplied. True BN affine is fixed, buffers supplied but BN remains train mode. D3 reuses qualification without new runs. Both oracle conditions remain dropout-identity surrogates and receive no private dropout seed/masks or true rows/label alignment. Original stochastic dropout is retained in the evaluator's actual gradient/Adam capture; that Adam step reproduces existing uploads bit for bit.
The controls are unchanged; decoy is cyclic next-target reconstruction within the same condition and cell. 'Beats all controls' retains the original n8 three-sign-test intersection, but only as unadjusted descriptive classification outside Holm and without changing eligibility. These 24 paired batches are reused across conditions/instruments, not 144 independent batches.

| Cell | Condition | Median accuracy | Mean/mode W/L/T; median delta | Marginal W/L/T; median delta | Decoy W/L/T; median delta | Beats all (descriptive) |
|---|---|---:|---|---|---|---|
| baf_r10_b4_given_tableak | D1 | 0.216216216 | 0/8/0; -0.185810811 | 0/8/0; -0.344594595 | 4/4/0; -0.00675675676 | False |
| baf_r10_b4_given_tableak | D2 | 0.212837838 | 0/8/0; -0.172297297 | 0/8/0; -0.351351351 | 3/3/2; 0 | False |
| baf_r10_b4_given_tableak | D3 | 0.236486486 | 0/8/0; -0.152027027 | 0/8/0; -0.300675676 | 4/4/0; 0.0202702703 | False |
| baf_r10_b4_given_wan | D1 | 0.216216216 | 0/8/0; -0.172297297 | 0/8/0; -0.337837838 | 5/2/1; 0.0101351351 | False |
| baf_r10_b4_given_wan | D2 | 0.206081081 | 0/8/0; -0.185810811 | 0/8/0; -0.364864865 | 4/3/1; 0.00337837838 | False |
| baf_r10_b4_given_wan | D3 | 0.222972973 | 0/8/0; -0.172297297 | 0/8/0; -0.341216216 | 4/4/0; 0.00675675676 | False |
| ieee_cis_r10_b4_given_tableak | D1 | 0.363392857 | 0/8/0; -0.400595238 | 0/8/0; -0.479166667 | 3/5/0; -0.00357142857 | False |
| ieee_cis_r10_b4_given_tableak | D2 | 0.38422619 | 0/8/0; -0.372619048 | 0/8/0; -0.457440476 | 6/2/0; 0.00327380952 | False |
| ieee_cis_r10_b4_given_tableak | D3 | 0.433630952 | 0/8/0; -0.33452381 | 0/8/0; -0.403571429 | 7/1/0; 0.019047619 | False |
| ieee_cis_r10_b4_given_wan | D1 | 0.370535714 | 0/8/0; -0.40327381 | 0/8/0; -0.480654762 | 4/4/0; -0.00178571429 | False |
| ieee_cis_r10_b4_given_wan | D2 | 0.382738095 | 0/8/0; -0.376190476 | 0/8/0; -0.460119048 | 6/2/0; 0.0068452381 | False |
| ieee_cis_r10_b4_given_wan | D3 | 0.40952381 | 0/8/0; -0.351785714 | 0/8/0; -0.438095238 | 4/4/0; 0.00119047619 | False |
| paysim_r10_b4_given_tableak | D1 | 0.680555556 | 4/4/0; 0.0138888889 | 5/2/1; 0.0416666667 | 5/2/1; 0.0277777778 | False |
| paysim_r10_b4_given_tableak | D2 | 0.666666667 | 5/3/0; 0.0277777778 | 5/2/1; 0.0416666667 | 6/1/1; 0.0833333333 | False |
| paysim_r10_b4_given_tableak | D3 | 0.666666667 | 4/3/1; 0.0138888889 | 5/3/0; 0.0555555556 | 4/2/2; 0.0138888889 | False |
| paysim_r10_b4_given_wan | D1 | 0.722222222 | 4/4/0; 0.0138888889 | 6/2/0; 0.0694444444 | 5/2/1; 0.0277777778 | False |
| paysim_r10_b4_given_wan | D2 | 0.611111111 | 3/3/2; 0 | 4/3/1; 0.0138888889 | 5/2/1; 0.0277777778 | False |
| paysim_r10_b4_given_wan | D3 | 0.638888889 | 2/6/0; -0.0555555556 | 5/3/0; 0.0277777778 | 5/2/1; 0.0555555556 | False |

### Interpretation under the pre-run diagnostic rules

- baf_r10_b4_given_tableak: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.
- baf_r10_b4_given_wan: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.
- ieee_cis_r10_b4_given_tableak: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.
- ieee_cis_r10_b4_given_wan: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.
- paysim_r10_b4_given_tableak: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.
- paysim_r10_b4_given_wan: INSTRUMENT_OR_BUDGET_INADEQUATE; negative qualification says nothing about privacy.

If D1 fails, the fixed estimators/budgets are inadequate for this model under the declared approximations; the negative qualification result says nothing about privacy. A positive D1 and failed D2 are a pattern consistent with an Adam-step limitation; positive D1/D2 with failed D3 are consistent with unknown-BN limitations. These diagnostic attributions are conditional, not causal proof; shared dropout mismatch, small post-hoc-selected samples and optimizer sensitivity remain. No diagnostic result qualifies a cell or authorizes confirmatory measurements.

### Upload sign-likeness

Adam upload is theta_after minus theta_before: cosine with +sign(g) is generally negative because descent is in the opposite direction. The additional descent-aligned cosine uses -upload versus sign(g). No absolute-value substitution. D1 observed gradient uses +g. Cosines are per actual batch, not model-coordinate replication.

| Dataset | Median gradient vs sign(g) | Median Adam upload vs sign(g) | Median -Adam upload vs sign(g) | Zero gradient/upload norms |
|---|---:|---:|---:|---|
| baf | 0.183802493 | -0.983512729 | 0.983512729 | 0/0 |
| ieee_cis | 0.0256446833 | -0.956988752 | 0.956988752 | 0/0 |
| paysim | 0.286499396 | -0.968077779 | 0.968077779 | 0/0 |

## Confirmatory table and fixed Holm family

| Eligible cells | Measured arms/targets | Intervals ranks 13/27 | Transform-versus-DP conclusion |
|---:|---|---|---|
| 0 | None; empty declared CSV schema | NOT_APPLICABLE; no n39 pairs | NOT_APPLICABLE |
All 6,048 original reservations remain: 72 cells × 2 transforms × 7 comparators × 3 metrics × 2 directions. Each is p=1 with NOT_APPLICABLE_UNQUALIFIED, n=0, Holm-adjusted p=1, reject=false. These values reserve the family, not measured p values or evidence of equality. No DP, v1 or v2 arm is measured. The proposed P34B comparator was joint C=.01/per-tensor, replace-one sensitivity 2C, 50 releases, delta1e-5, epsilon1/3/10; no calibrated sigma or empirical DP outcome is claimed here. That mechanism is contribution/transcript-level, not record-level DP. No transform privacy theorem follows.

## Independent verification and provenance

Independent tensor audit: PASS; 576 qualification and 96 new diagnostic reconstructions, 48 reused D3, 144 original uploads. Source firewall, exact checkpoint/upload replay, known/private knowledge separation, BN finite/variance, budget and observable-selection checks PASS. Separate vectorized native TabLeak cost/SciPy Hungarian reload every metric; SciPy exact binomial tests independently agree with production integer-combinatorial tests.
CSV readback: PASS; all 270 tests recomputed (216 qualification, 54 descriptive), all paired IDs/numerics and medians bound to reloaded evidence; NumPy Holm recomputed over all6048. No missing/nonfinite CSV outputs. Stored float32 sign-cosines match fresh float32 recomputation exactly; independent NumPy float64 dot/norm recomputation agrees within absolute 1e-5 (relative tolerance zero), under the approved post-hoc infrastructure addendum. No exact tie or gate threshold changed.
Main amendment SHA-256: `0a9cd3c1e9b9ccdce2fbacfdc2f4d326b535b109e63af8163f0cdbf2b575a0f7`.
Post-hoc addendum SHA-256: `08cfe137d2e0ca85a1f2410c1671ee8b60780b0d97452e7ddac71bf69e5f5da7`.
Qualification freeze SHA-256: `10bfa6410f31d66dd3a5dac38cec77dad7f6bf2fdcb3bb3fee4c44a40a7c2bf3`.
Diagnostic freeze SHA-256: `d9c444f67eb2abc14ace2f4bd5b66d98fbd46f7c7852063ece0b82e9ae77cf8e`.
Qualification scientific manifest SHA-256: `f8fa4bf50a78dc547dedab19d9c42347e79a81e3cc6da29c341f922926b424be` (original unchanged).
Final manifest and COMPLETE receipt carry the final report digest; no self-referential report hash is embedded in this document.
Earlier post-exit administrative seals are preserved snapshots. Their old progress/checklist hashes are not commitments to later authorized administrative updates. Bytecode caches are excluded as reproducible intermediates; scientific outputs and closed detached logs are hashed. No Latex/, external_defenses/, dataset, earlier artifact/report, commit or push was changed.

## Exact commands

Run from FL-DNA/ with .venv-phase1, CPU and torch/BLAS thread1. All launch/preflight receipts retain exact paths and stdout/stderr.

```text
.venv-phase1/bin/python -B experiments/priority35a_qualification.py --preflight
.venv-phase1/bin/python -B experiments/priority35a_qualification.py --prepare
.venv-phase1/bin/python -B experiments/priority35a_qualification.py --launch
.venv-phase1/bin/python -B experiments/priority35a_postexit.py
.venv-phase1/bin/python -B experiments/priority35a_diagnostic.py --prepare
.venv-phase1/bin/python -B experiments/priority35a_diagnostic.py --launch
.venv-phase1/bin/python -B experiments/priority35a_report.py --freeze
.venv-phase1/bin/python -B experiments/priority35a_report.py --finish
```

Final tests/py_compile/git diff --check commands, return codes and logs: `artifacts/priority35a/final/verification_commands/1791614151217490000`. Python 3.9.6, torch 2.2.2, NumPy 1.26.4. CPU only, no GPU workloads.

## Limitations

This is a negative instrument qualification result, not a guarantee that records are private. The oracle-BN diagnostic is privileged and post hoc, limited to b4/r10/labels given/n8. Dropout masks remain unknown and its fixed identity approximation is not an expectation of the complete nonlinear network/update. Labels estimated is focal-loss count enumeration, not a universal sign-based label theorem. Adam arithmetic is reproduced exactly but its near-sign update and rounded weak derivatives can limit reconstruction optimization. True BN affine was fixed only in privileged D1/D2, never deployable qualification. Population-derived references, encoded-category ambiguity, native absent-type normalization, natural fraud prevalence, two-start finite budget, shared clients/checkpoints and finite-population sampling all bound interpretation. A failure to beat references is not a zero-leakage test. Conditional n24/n39 were not reached, so there is no adjusted comparison of DNA transforms versus DP and no record-level DP certification.

Sources: [TabLeak, Vero et al. (2023)](https://proceedings.mlr.press/v202/vero23a.html), [Wan et al., optimizer-update matching (2026)](https://proceedings.mlr.press/v306/wan26c.html). These are model-specific adaptations, not reproductions of every official-paper configuration.

## Post-hoc infrastructure verifier disclosure

Addendum: `protocols/amendments/2026-10-10_priority35a_cosine_verifier_infrastructure_posthoc.md`; SHA-256 `669142eec793f1779db2fa2d87afd718a502d520325a79543f8415f7aa553c2d`. Original failed verifier output/traceback: `artifacts/priority35a/final/REQUIRES_DIRECTION_cosine_audit_20261010.json`, unchanged SHA-256 `92921fec6f4403fffb50d9a908e30038b577957256a15f869e5ca0b1ac8d0ea1`; its original progress-log event is retained. Frozen sources were not edited and no measurement was rerun.

All 72 stored float32 statistics matched exactly. Maximum absolute float64 discrepancy: 2.6147116197750009e-06, below the predeclared 1e-5 absolute guard. Observed stored-float32 ranges:

- gradient_vs_sign: [0.011188816279172897, 0.45099937915802002].
- adam_vs_sign: [-0.98684942722320557, -0.70442962646484375].
- descent_aligned_adam_vs_sign: [0.70442962646484375, 0.98684942722320557].

No diagnostic conclusion depends on cosine at the 1e-5 level: cosine is descriptive only; the locked interpretation uses feature-accuracy comparisons with controls. All six D1 cells fail, so instrument/budget inadequacy remains the conclusion and negative qualification says nothing about privacy. Qualification 0/72 and Holm6048 are unchanged.

Verifier-only commands: `.venv-phase1/bin/python -B experiments/priority35a_cosine_verifier.py --prepare` and `--finish`. Synthetic tests, py_compile and git diff --check logs are included in the manifest.

## Additional post-hoc descriptive diagnostic: D1-prime (2026-10-10)

This appended section does not revise any earlier result or interpretation. Original report byte prefix and original qualification/COMPLETE/manifests remain preserved. Qualification remains 0/72; Holm6048 is unchanged. This is a privileged descriptive diagnostic, not evidence for deployable Config-C privacy or new qualification.

Pre-execution addendum `protocols/amendments/2026-10-10_priority35a_d1prime_descriptive.md`, SHA-256 `daeac58ff1f9b136fea73a06576fa66ea578ac6bc01b1de20e8755ca2abdd735`; freeze SHA-256 `97c19ff09c0ce92a1f1c5c55b8b8a1b4b60d40228e2e9c26bc05268711ebdb43`.

Six cells, b4/r10/labels given, same eight D1 targets. a: true local BN and actual dropout masks, 2x4000 steps. b: same plus 30x1500 members, coordinate lower-median pooling after categorical softmax, physical mask/label row positions retained. Same objectives, initialization seed rule, Adam .03 and annealing; only budget/ensemble changes. c: NOT_RUN, prospectively omitted. TabLeak reference budget [official code](https://github.com/eth-sri/tableak); this is a budget and position-aware ensemble adaptation, not a full reference reproduction.

Direct user approval additionally supplies labels attached to mask row positions. Original D1 knew an unordered multiset. Two IEEE-CIS batches have mixed labels; for them improvements involve both masks and label-position information. All other batches have homogeneous labels. No true feature rows, source IDs or private RNG enter the estimator. BN remains train mode and true affine is fixed in this privileged diagnostic.

Unit n=8 paired batches per cell, not ensemble members; 24 distinct reused batches. Data-free references unchanged; decoy is cyclic next-target reconstruction under the same condition/cell. Exact one-sided sign tests, exact ties excluded from binomial n, all three p<.05 for descriptive beats-all. Unadjusted post-hoc classifications lie outside unchanged Holm6048.

| Cell | Condition | Median feature accuracy | Mean/mode W/L/T; median delta; p | Marginal W/L/T; median delta; p | Decoy W/L/T; median delta; p | Beats all (descriptive) |
|---|---|---:|---|---|---|---|
| baf_r10_b4_given_tableak | D1-prime-a | 0.216216216 | 0/8/0; -0.172297297; 1 | 0/8/0; -0.344594595; 1 | 4/4/0; -0.00675675676; 0.63671875 | False |
| baf_r10_b4_given_tableak | D1-prime-b | 0.337837838 | 0/8/0; -0.0439189189; 1 | 0/8/0; -0.222972973; 1 | 4/3/1; 0.0101351351; 0.5 | False |
| baf_r10_b4_given_wan | D1-prime-a | 0.209459459 | 0/8/0; -0.175675676; 1 | 0/8/0; -0.361486486; 1 | 5/3/0; 0.0236486486; 0.36328125 | False |
| baf_r10_b4_given_wan | D1-prime-b | 0.334459459 | 0/8/0; -0.0506756757; 1 | 0/8/0; -0.236486486; 1 | 3/4/1; -5.55111512e-17; 0.7734375 | False |
| ieee_cis_r10_b4_given_tableak | D1-prime-a | 0.363095238 | 0/8/0; -0.398511905; 1 | 0/8/0; -0.482440476; 1 | 2/6/0; -0.00208333333; 0.96484375 | False |
| ieee_cis_r10_b4_given_tableak | D1-prime-b | 0.53125 | 0/8/0; -0.225595238; 1 | 0/8/0; -0.30922619; 1 | 4/4/0; -0.000892857143; 0.63671875 | False |
| ieee_cis_r10_b4_given_wan | D1-prime-a | 0.374702381 | 0/8/0; -0.39672619; 1 | 0/8/0; -0.472619048; 1 | 4/4/0; 0.000892857143; 0.63671875 | False |
| ieee_cis_r10_b4_given_wan | D1-prime-b | 0.523511905 | 0/8/0; -0.235416667; 1 | 0/8/0; -0.324107143; 1 | 3/5/0; -0.00476190476; 0.85546875 | False |
| paysim_r10_b4_given_tableak | D1-prime-a | 0.611111111 | 2/6/0; -0.0694444444; 0.96484375 | 2/6/0; -0.0416666667; 0.96484375 | 4/4/0; -0.0138888889; 0.63671875 | False |
| paysim_r10_b4_given_tableak | D1-prime-b | 0.694444444 | 6/2/0; 0.0555555556; 0.14453125 | 6/2/0; 0.0694444444; 0.14453125 | 4/3/1; 0.0138888889; 0.5 | False |
| paysim_r10_b4_given_wan | D1-prime-a | 0.625 | 2/5/1; -0.0277777778; 0.9375 | 3/4/1; -0.0138888889; 0.7734375 | 4/4/0; 0.0138888889; 0.63671875 | False |
| paysim_r10_b4_given_wan | D1-prime-b | 0.722222222 | 6/2/0; 0.0555555556; 0.14453125 | 7/1/0; 0.0972222222; 0.03515625 | 3/3/2; 0; 0.65625 | False |

### Interpretation under the pre-run D1-prime rules

- baf_r10_b4_given_tableak: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.
- baf_r10_b4_given_wan: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.
- ieee_cis_r10_b4_given_tableak: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.
- ieee_cis_r10_b4_given_wan: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.
- paysim_r10_b4_given_tableak: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.
- paysim_r10_b4_given_wan: Both fail: estimators remain inadequate for this model under the fixed evaluation; original negative qualification says nothing about privacy.

Small selected n8 samples, unadjusted exploratory tests, finite budgets, fixed BN, ordered-label privilege and changed pooling limit causal claims. A-versus-b significance differences alone do not prove an interaction. No change to earlier conclusions and no evidence of transform superiority versus DP.

Independent tensor/native mask/gradient/firewall/budget/objective/pooling/metric audit: PASS; CSV recomputation: PASS, all36 tests,384 per-target arm rows. Tests/py_compile/git diff --check logs: `artifacts/priority35a/d1prime/verification_commands/1791618746235698000`. CPU/thread1; no measurement exclusions or replacement.

Commands: `.venv-phase1/bin/python -B experiments/priority35a_d1prime.py --prepare`, `--launch`, `--finish`. Results, CSVs and separate SHA-256 manifest/COMPLETE: `artifacts/priority35a/d1prime/`. Original report prefix snapshot: `artifacts/priority35a/d1prime/base_report.md`.

## Post-hoc descriptive sanity diagnostic T (2026-10-10)

Earlier text and decisions above remain unchanged. T is a privileged diagnostic, not a qualification rerun or privacy comparison. Same24 distinct batches/six cells, b4/r10/given position-bound labels, actual dropout masks and true local BN; CPU/thread1.

Addendum SHA256 `4fdb005d5bfc20c4f9ea9c53855045c9372b2d2d3b970b5ec65efc1e243f37fa`; pre-measurement freeze SHA256 `9802d6c05eab3f97f5936a63d5ef076966fe44ab8a430fcf82978d0d9a27f32f`.

T0/T1: objectives evaluate true encoded rows directly; original objective uses stored soft output, not its hard-decoded metric representation. Per-batch soft/hard objectives and output/truth ratios are in `artifacts/priority35a/sanity_t/T0_T1.csv`. Ratios with abs(truth objective)<=1e-6 are NA, not infinity.

| Cell | Minimum T0 accuracy | Bit-exact gradients /8 | Max relative mismatch | Median truth objective | Median original soft objective | Truth strictly lower /8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baf_r10_b4_given_tableak | 1 | 8 | 0 | -1.07288361e-06 | 0.281132311 | 8 |
| baf_r10_b4_given_wan | 1 | 8 | 0 | -1.07288361e-06 | 0.211821109 | 8 |
| ieee_cis_r10_b4_given_tableak | 1 | 8 | 0 | -2.74181366e-06 | 1.05217439 | 8 |
| ieee_cis_r10_b4_given_wan | 1 | 8 | 0 | -2.74181366e-06 | 0.67741996 | 8 |
| paysim_r10_b4_given_tableak | 1 | 8 | 0 | -5.36441803e-07 | 0.40458627 | 8 |
| paysim_r10_b4_given_wan | 1 | 8 | 0 | -5.36441803e-07 | 0.292372078 | 8 |

T2: two starts x4000 iterations at each SD noise level. Initialization mapping to finite categorical logits was fixed before runs. Eight batches per cell/scale; starts/scales are not independent replicates. Final feature accuracy/MSE use hard-decoded rows, final objective uses soft rows.

| Cell | Noise SD | Median initial accuracy | Median final accuracy | Median final std MSE | Median final soft objective | Returns truth /8 | Far, strict <= truth /8 | Far, within1e-6 /8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baf_r10_b4_given_tableak | 0.01 | 1 | 0.408783784 | 29.706708 | 0.0454236679 | 0 | 0 | 0 |
| baf_r10_b4_given_tableak | 0.1 | 1 | 0.297297297 | 38.8773897 | 0.0790098459 | 0 | 0 | 0 |
| baf_r10_b4_given_tableak | 0.5 | 0.567567568 | 0.206081081 | 67.475926 | 0.297569826 | 0 | 0 | 0 |
| baf_r10_b4_given_wan | 0.01 | 1 | 0.435810811 | 11.2128191 | 0.0465428233 | 0 | 0 | 0 |
| baf_r10_b4_given_wan | 0.1 | 1 | 0.283783784 | 62.9741684 | 0.112466216 | 0 | 0 | 0 |
| baf_r10_b4_given_wan | 0.5 | 0.554054054 | 0.199324324 | 153.284323 | 0.295771062 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_tableak | 0.01 | 1 | 0.474404762 | 2781.51304 | 0.00455641793 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_tableak | 0.1 | 0.998511905 | 0.356547619 | 34514.5051 | 0.0707703754 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_tableak | 0.5 | 0.505059524 | 0.179464286 | 29879.9167 | 0.36800839 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_wan | 0.01 | 1 | 0.525892857 | 3928.47389 | 0.00417178869 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_wan | 0.1 | 0.998809524 | 0.350892857 | 18544.6899 | 0.0842808783 | 0 | 0 | 0 |
| ieee_cis_r10_b4_given_wan | 0.5 | 0.50297619 | 0.208333333 | 47422.5628 | 0.275151134 | 0 | 0 | 0 |
| paysim_r10_b4_given_tableak | 0.01 | 1 | 1 | 0.000558936892 | 0.00462785747 | 2 | 0 | 1 |
| paysim_r10_b4_given_tableak | 0.1 | 1 | 0.875 | 0.209341212 | 0.0838469826 | 0 | 0 | 0 |
| paysim_r10_b4_given_tableak | 0.5 | 0.555555556 | 0.402777778 | 1.21875809 | 0.611055374 | 0 | 0 | 0 |
| paysim_r10_b4_given_wan | 0.01 | 1 | 0.986111111 | 0.00700930307 | 0.0054461062 | 1 | 0 | 0 |
| paysim_r10_b4_given_wan | 0.1 | 1 | 0.888888889 | 0.0741482032 | 0.0865878463 | 0 | 0 | 0 |
| paysim_r10_b4_given_wan | 0.5 | 0.583333333 | 0.472222222 | 1.37937697 | 0.491071343 | 0 | 0 | 0 |

### Interpretation under the prospectively locked T rules

No T0/forward-model defect was detected in these tested batches. This checks the metric and privileged forward model only, not every component of the original estimators.
- baf_r10_b4_given_tableak: 0/24 scale-specific outputs return to truth; 0/24 far outputs are numerically no higher than truth. Cause remains unresolved: privileged near-truth optimization did not return to truth; landscape, budget and categorical representation remain possible limitations.
- baf_r10_b4_given_wan: 0/24 scale-specific outputs return to truth; 0/24 far outputs are numerically no higher than truth. Cause remains unresolved: privileged near-truth optimization did not return to truth; landscape, budget and categorical representation remain possible limitations.
- ieee_cis_r10_b4_given_tableak: 0/24 scale-specific outputs return to truth; 0/24 far outputs are numerically no higher than truth. Cause remains unresolved: privileged near-truth optimization did not return to truth; landscape, budget and categorical representation remain possible limitations.
- ieee_cis_r10_b4_given_wan: 0/24 scale-specific outputs return to truth; 0/24 far outputs are numerically no higher than truth. Cause remains unresolved: privileged near-truth optimization did not return to truth; landscape, budget and categorical representation remain possible limitations.
- paysim_r10_b4_given_tableak: 2/24 scale-specific outputs return to truth; 1/24 far outputs are numerically no higher than truth. Near-truth optimization success with higher-objective original outputs is consistent with an optimization/budget/landscape limitation. Far low-objective outputs indicate numerical non-identification by this objective; this is not a proof of exact model non-identifiability.
- paysim_r10_b4_given_wan: 1/24 scale-specific outputs return to truth; 0/24 far outputs are numerically no higher than truth. Near-truth optimization success with higher-objective original outputs is consistent with an optimization/budget/landscape limitation.

No new sign tests or Holm adjustment; original0/72 and Holm6048 remain unchanged. Truth-informed starts are not realistic server capabilities, softmax cannot encode exact one-hot truth with finite logits, and these selected n8 batches do not establish privacy or general estimator correctness.

Independent audit reloads every target/output, native gradient/masks, firewall, initialization, budgets, metric, float32 objective and CSV summary. All144 T2 jobs and48 T0/T1 checks included. Separate SHA256 manifest/COMPLETE in `artifacts/priority35a/sanity_t/`. Commands: `.venv-phase1/bin/python -B experiments/priority35a_sanity_t.py --prepare`, `--launch`, `--finish`. No earlier measurement was rerun.
