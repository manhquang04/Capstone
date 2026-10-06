# Independent check: S1 image results

## Scope and method

This check reads only `artifacts/priority30_native_defenses/audit/S1/` and the
Priority 30 report. It does not access `S2`, run an attack, train a model, or
alter artifacts. Each `ran` record was joined to its saved `raw.png` and
`reconstruction.png`. MSE, PSNR, and channel-aware SSIM were recomputed on
clipped 8-bit PNG values using the audit implementation's stated conventions:
raw scale `[0,1]`, PSNR `-10 log10(MSE)`, and `data_range=1`.

The gray comparator is the constant 0.5 image. The CIFAR-mean comparator is
the channel-constant image `(0.4914672375, 0.4822617471, 0.4467701316)`, from
the audited driver's `inversefed.consts.cifar10_mean`. A win means lower MSE
(equivalently higher PSNR) or higher SSIM than the comparator on the same
target. Exact one-sided sign tests omit ties and test the directional
alternative that reconstructions are better than the comparator. Values below
are recomputed from PNGs; JSON values are separately checked below.

## File and schema integrity

- Found 546 `result.json` files, exactly matching the report: 351 `ran` and
  195 `NOT_ASSESSABLE`.
- The nine emitted `ran` `(evaluation, defense)` cells each contain exactly 39
  unique target IDs, `0--38`, with no missing or extra ID.
- Within every cell there are no duplicate CIFAR-10 indices. The target-ID to
  CIFAR-index mapping is identical across all nine emitted cells.
- Every `ran` JSON agrees with its path for `stage=S1`, `domain=image`,
  `eval`, `defense`, and `target_id`. `NOT_ASSESSABLE` records intentionally
  contain only status/reason fields; no contradiction with their paths was
  found.
- `summary.csv` has 546 rows and matches every `ran` JSON metric exactly at
  the stored floating-point precision. Its mean JSON PSNR is
  14.907343312837888 dB, matching the report's 14.90734331283789 dB.
- No `undefended` result record is present in S1, and the report supplies no
  per-target undefended metrics for this 39-target pool. Thus paired wins and
  sign tests versus the undefended attack are **NOT RECOMPUTABLE** from the
  permitted records. This is a missing-comparator provenance issue, not a
  numerical result about a defense.

## Recomputed reconstruction metrics

`mean` and `median` order is MSE / PSNR dB / SSIM. Metrics recomputed from
saved PNGs can differ slightly from JSON values computed before PNG
quantization.

| Eval | Defense | n | Mean | Median |
|---|---|---:|---|---|
| E1 | ATS | 39 | 0.065385 / 12.190743 / 0.087199 | 0.064092 / 11.931985 / 0.071511 |
| E1 | DNA v1 conservative | 39 | 0.008012 / 22.460001 / 0.809645 | 0.005552 / 22.555391 / 0.827094 |
| E1 | DNA v2 0.95 | 39 | 0.007840 / 22.368704 / 0.798754 | 0.005877 / 22.308390 / 0.809364 |
| E1 | gradient pruning | 39 | 0.013737 / 19.601972 / 0.660500 | 0.011428 / 19.420196 / 0.673112 |
| E1 | PRECODE | 39 | 0.116316 / 9.452760 / 0.049398 | 0.109313 / 9.613300 / 0.046887 |
| E1 | Soteria | 39 | 0.048701 / 13.220821 / 0.396394 | 0.049309 / 13.070759 / 0.400048 |
| E2 | ATS | 39 | 0.065385 / 12.190743 / 0.087199 | 0.064092 / 11.931985 / 0.071511 |
| E2 | PRECODE | 39 | 0.116316 / 9.452760 / 0.049398 | 0.109313 / 9.613300 / 0.046887 |
| E2 | Soteria | 39 | 0.048701 / 13.220821 / 0.396394 | 0.049309 / 13.070759 / 0.400048 |

## Baseline comparisons and exact sign tests

All reconstructed images tie the gray comparator on both MSE and SSIM in all
nine cells: wins/losses/ties are `0/0/39`, so the exact one-sided p-value is
1. The table therefore gives the informative CIFAR-mean results. Each entry is
`wins/losses/ties; p`, with MSE first and SSIM second. PSNR has the same
target-wise ordering as MSE and is not duplicated.

| Eval | Defense | CIFAR mean: MSE | CIFAR mean: SSIM |
|---|---|---|---|
| E1 | ATS | 11/28/0; 0.9983110760 | 10/29/0; 0.9994674902 |
| E1 | DNA v1 conservative | 39/0/0; 1.818989404e-12 | 39/0/0; 1.818989404e-12 |
| E1 | DNA v2 0.95 | 39/0/0; 1.818989404e-12 | 39/0/0; 1.818989404e-12 |
| E1 | gradient pruning | 39/0/0; 1.818989404e-12 | 39/0/0; 1.818989404e-12 |
| E1 | PRECODE | 0/39/0; 1.0000000000 | 9/30/0; 0.9998529616 |
| E1 | Soteria | 28/11/0; 0.0047376521 | 37/2/0; 1.420630724e-09 |
| E2 | ATS | 11/28/0; 0.9983110760 | 10/29/0; 0.9994674902 |
| E2 | PRECODE | 0/39/0; 1.0000000000 | 9/30/0; 0.9998529616 |
| E2 | Soteria | 28/11/0; 0.0047376521 | 37/2/0; 1.420630724e-09 |

## Discrepancies and limitations

There is no discrepancy between the report's aggregate PSNR, `summary.csv`,
and the stored JSON values. PNG recomputation differs from the JSON metrics by
more than 0.001 in at least one metric for 190 of 351 files (maximum absolute
difference 0.0154699102). This is expected from saving/reloading 8-bit PNGs:
the audit computed JSON metrics from tensors before image quantization. The
JSON source, rather than PNGs, should therefore be used for final numerical
tables; the PNG calculation is an independent persistence check.

The report calls S1 "IMAGE E1 + E2, confirmatory n=39," but it is not itself a
complete paired defense-versus-undefended evaluation artifact because the
needed undefended per-target results are absent. No RQ4 verdict or defense
claim should be derived from this independent check alone.
