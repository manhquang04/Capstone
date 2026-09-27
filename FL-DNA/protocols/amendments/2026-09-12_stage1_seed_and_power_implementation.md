# Stage-1 technical amendment: paired RNG isolation and power implementation

**Timestamp:** 2026-09-12T14:57:00Z  
**Impact:** technical/reproducibility; no approved scientific threshold changed

Before viewing any RQ2 pilot utility outcome, the following implementation
details are fixed:

1. The RQ2 official development scale is `MAX_ROWS=500000`, matching the
   approved resource benchmark described as the official scale.
2. DP Gaussian noise uses a method-specific deterministic generator derived
   from `FL_RUN_SEED`, round and client. It must not advance the global PyTorch
   RNG used by dropout/training in later clients.
3. DNA Transform's run seed is derived from `FL_RUN_SEED` unless explicitly
   supplied; a fresh secure-random default is prohibited in paired RQ2 runs.
4. The conservative RQ2 variance estimate is the one-sided 95% chi-square
   upper confidence bound for the variance of paired development differences.
   Required sample size is computed with a one-sided paired non-inferiority
   t-test approximation separately for F1 and AUC; the larger requirement over
   both endpoints and both primary DNA methods is used.
5. Development split/partition and model-initialization checksums are
   materialized once per approved seed and replay-verified before the batch.
6. The eight methods in the shared development batch use a balanced Latin
   rotation across the eight seeds. At every scheduling wave each seed and
   method appears once; execution uses at most eight concurrent processes.

The original PRE-PILOT freeze manifest remains immutable. Hashes of code
changed under this amendment are recorded in the Stage-1 run manifest.

## Infrastructure failure and authorized identical retry

At `2026-09-12T15:04:00Z`, the first batch launch exposed a deterministic
startup error in the DP runner: `torch.Generator` was referenced without
importing `torch`. No DP utility result was produced. The mixed launch was
stopped so later jobs could not silently use a different code hash. The missing
import was added; all failed/interrupted jobs retain their original logs under
per-job `attempts/` directories and are retried with identical seeds and
scientific configuration. This is an infrastructure-only repair and does not
change any defense multiplier, margin, outcome, or selection rule.
