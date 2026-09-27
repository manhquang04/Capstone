# Amendment — Medium distortion-grid local interpolation

**Amendment ID:** `2026-09-13-RQ1-MEDIUM-DISTORTION-GRID-EXTENSION`  
**Timestamp:** 2026-09-13, before any medium confirmatory target exists  
**Evidence consulted:** distortion-only development calibration  
**Privacy/reconstruction outcomes consulted:** none  

The complete frozen variant grid was run. `DNA-TRANSFORM-MEDIUM-V1` produced
median relative-L2 `0.0987576365`. The nearest DP point, `0.0003`, produced
`0.0937010154`, a `5.120233%` mismatch and therefore correctly failed the 5%
tolerance by `0.120233` percentage points. The next existing point `0.00035`
brackets the DNA distortion on the monotonic grid.

As permitted prospectively by the variant amendment when no point passes, add
exactly three locally interpolated multipliers: `[0.00031, 0.000315, 0.00032]`.
All other calibration values remain unchanged: clip norm 100, seeds
`[11,22,33,44,55]`, eight development groups, median statistic, 5% tolerance,
and smallest-multiplier tie break. The initial full grid and failed result are
retained. No further medium grid expansion is authorized by this amendment.
