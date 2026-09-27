# Stage-1 supervisor approvals for RQ1/RQ2 confirmatory freeze

**Timestamp:** 2026-09-13T00:24:52+07:00  
**Approved by:** Manh Quang (supervisor and research lead)  
**Evidence source:** chat-session instruction dated 2026-09-13  
**Scope:** administrative recording of approved Stage-1 outputs; no
confirmatory outcome has been generated or inspected

The following Stage-1 decisions are approved for the RQ1/RQ2 confirmatory
freeze:

1. `tie_threshold_mse = 0.0390625`, obtained from the 640-artifact
   development-only numerical-stability study, is the frozen RQ1 tie
   threshold.
2. `DP_DISTORTION_MATCHED` is frozen at `clip_norm = 100.0` and
   `noise_multiplier = 0.00025`. The valid grid expansion and implementation
   alignment are documented in
   `2026-09-12_rq1_distortion_grid_expansion.md`.
3. `required_confirmatory_seed_count = 21` is approved for RQ2 and remains
   below the previously frozen ceiling of 30.
4. `DP_UTILITY_MATCHED = NOT_FOUND` is accepted as the final calibration
   result. The search is closed: the grid must not be reopened and the nearest
   candidate must not be relabeled. RQ2 confirmatory execution contains
   Baseline, lossless DNA and DNA Transform as primary methods, with
   distortion-matched clipping/noise as a secondary/contextual method.
5. Before generation of the confirmatory seed list, the following
   interpretation rule is frozen for RQ2:

   > At `n = 21`, if the lower bound of the 95% paired confidence interval for
   > `F1(DNA Transform) - F1(Baseline)` is below `-0.02`, the official
   > conclusion is: “DNA Transform không thiết lập được non-inferiority F1 ở
   > margin đã duyệt.” This is a valid potentially negative result and is not
   > grounds to relax the margin, change the endpoint rule, add seeds, or
   > exclude an observation.

These approvals do not authorize confirmatory attack or training execution.
After the confirmatory configs, target/seed registries and freeze hashes are
recorded, a final execution authorization is still required before consuming
the one-run confirmatory budget.

