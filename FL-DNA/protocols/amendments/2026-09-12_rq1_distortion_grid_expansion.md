# RQ1 Stage-1 amendment: distortion-calibration grid expansion

**Timestamp:** 2026-09-12T15:24:00Z  
**Impact:** development calibration only; no privacy outcome inspected

The frozen initial multiplier grid
`[0.0001, 0.0005, 0.001, 0.005, 0.01]` was run in full on the registered
development groups and seeds. DNA's median relative-L2 was
`0.0755774997`; the nearest original candidate (`0.0001`) produced
`0.0310962571`, a relative mismatch of `58.855%`, so no candidate met the
frozen 5% tolerance.

As predeclared by RQ1 protocol section 7.2, the calibration grid is expanded
before any confirmatory target is generated. The original grid and complete
results are retained. Three log-local interpolation points between the two
original candidates bracketing the target are added:

```text
0.0002, 0.00025, 0.0003
```

All other values remain unchanged: clip norm 100, seeds
`[11, 22, 33, 44, 55]`, median relative-L2 statistic, 5% tolerance and the
smallest-multiplier tie-break. No attack reconstruction metric or privacy
direction was used to create this amendment.

## Implementation alignment addendum

The Phase-4 MC runner's legacy interface parameterized clipping/noise by an
exact target relative-L2 and fixed clip factor. The approved RQ1 comparator is
instead parameterized by clip norm and noise multiplier. Before confirmatory
freeze, a separate, backward-compatible `clip_norm_noise_multiplier` mode is
added to the runner and calibration is replayed through that exact shared
implementation. Historical legacy mode and artifacts are unchanged.

The selected distortion-only candidate (`noise_multiplier=0.00025`) is also
run on all eight RQ2 development seeds as the predeclared secondary
`FL_DP_DISTORTION_MATCHED` method. It is explicitly excluded from the frozen
five-candidate `DP_UTILITY_MATCHED` selection grid; adding it cannot change the
utility-matched selection result.
