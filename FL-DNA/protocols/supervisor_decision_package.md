# Supervisor Decision Package — RQ1, RQ2 and RQ3

**Package ID:** `FL-DNA-SUPERVISOR-DECISIONS-V1`  
**Version:** `1.0-approved`  
**Prepared:** 2026-09-12  
**Approved:** 2026-09-12, by Manh Quang (supervisor and research lead), via chat
session confirmation  
**Status:** `APPROVED — PRE-PILOT FROZEN for RQ1/RQ2/RQ3 pre-pilot configs`  

## 1. Purpose

This package collects the decisions required to move the three research
protocols from `DRAFT` to `PRE-PILOT FROZEN`. A checked recommendation is a
proposal for supervisor review, not approval by itself.

The two freeze gates remain separate:

```text
DRAFT
  -> approve this package + resource caps + pre-pilot configs
PRE-PILOT FROZEN
  -> run Stage 1 and compute required sample sizes
CONFIRMATORY FROZEN
  -> run official experiments
```

## 2. Decisions that can be made now

### 2.1 Common statistical convention

| Field | Recommended proposal | Alternatives | Reason |
| --- | ---: | ---: | --- |
| Alpha | **0.05** | 0.025 | Conventional confirmatory error rate; 0.025 is stricter but increases sample size |
| Target power | **0.80** | 0.90 | Standard minimum with manageable compute; 0.90 is preferable if resources allow |
| Confidence level | **95%** | 97.5% | Consistent with alpha 0.05 and ordinary two-sided interval reporting |
| RQ1 null win probability `p0` | **0.50** | supervisor justification required for any other value | Represents no directional advantage in paired wins |

Recommended decision:

- [x] Approve the bold values. (alpha=0.05, power=0.80, confidence=95%, p0=0.50)
- [ ] Approve with changes recorded here: _________________________________
- [ ] Defer; reason: ______________________________________________________

### 2.2 RQ3 overall decision rule

Recommended rule:

> A method is `ACCEPTABLE` only if every mandatory criterion passes at every
> required client scale for every mandatory deployment profile. It is
> `NOT_ACCEPTABLE` if a mandatory criterion clearly fails, and `INCONCLUSIVE`
> if required cells are missing or the frozen uncertainty rule cannot resolve
> a threshold crossing.

- [x] Approve.
- [ ] Replace with: _______________________________________________________

## 3. RQ1 substantive-effect decision

### 3.1 Minimum meaningful win probability

The primary sign test asks whether DNA reconstruction MSE is higher than the
distortion-matched clipping/noise MSE on more than half of non-tied target
groups. The following exact one-sided binomial calculations use `p0=0.50` and
`alpha=0.05`. They are design calculations, not estimates from prior outcomes.

| Minimum meaningful win probability `p1` | Interpretation | Effective non-tied `n`, power 0.80 | Initial draw with 10% ties + 5% dropout | Effective non-tied `n`, power 0.90 |
| ---: | --- | ---: | ---: | ---: |
| 0.60 | Small directional advantage | 158 | 185 | 213 |
| 0.65 | Moderate advantage | 69 | 81 | 93 |
| **0.70** | Clear and practically interpretable advantage | **37** | **44** | **53** |
| 0.75 | Large advantage only | 23 | 27 | 33 |

Recommendation: `p1=0.70`, target power `0.80`, expected tie rate `0.10` and
technical-dropout allowance `0.05`, subject to the resource ceiling. It avoids
designing around a very large effect (`0.75`) while remaining far more feasible
than trying to detect a weak 0.60 advantage. The final initial draw must be
recomputed by the versioned power-analysis script after the tie/dropout
assumptions are approved.

- [ ] Option A — `p1=0.65` (more sensitive, substantially more compute).
- [x] **Option B — `p1=0.70` (recommended balance).**
- [ ] Option C — `p1=0.75` (lower compute, only detects a large advantage).
- [ ] Other: __________; justification: ___________________________________

### 3.2 Tie definition and multiplicity

Proposed choices:

- Tie threshold: define using a development-only numerical-stability study;
  freeze an absolute/relative MSE tolerance before confirmatory execution.
- Primary contrast: DNA Transform versus `DP_DISTORTION_MATCHED` on normalized
  feature-MSE.
- Primary hypothesis: one-sided.
- Secondary DNA-vs-raw, DP-vs-raw, PSNR and SSIM analyses: report with Holm
  adjustment as a family; they do not replace the primary contrast.

- [x] Approve this structure.
- [ ] Required changes: ___________________________________________________

## 4. RQ2 non-inferiority decisions

Margins are absolute metric-point losses relative to FL baseline. Smaller
margins demand stronger utility preservation and generally require more seeds.

| Option | F1 margin | AUC-ROC margin | Interpretation |
| --- | ---: | ---: | --- |
| A — strict | 0.01 | 0.0025 | Allows very little degradation; likely high sample requirement |
| **B — balanced** | **0.02** | **0.0050** | Detects a material F1 loss while respecting the already high AUC scale |
| C — prototype-tolerant | 0.05 | 0.0100 | Easier to establish but may permit operationally important fraud loss |

Recommendation: Option B and require both endpoints to pass. F1 is sensitive to
minority-class decisions and remains the primary operational endpoint; AUC-ROC
is co-primary under the wording of the RQ. The untouched test set must never be
used to revise these margins.

- [ ] Option A.
- [x] **Option B (recommended).**
- [ ] Option C.
- [ ] Custom F1: ________; custom AUC: ________; reason: _________________

Endpoint decision:

- [x] **Both F1 and AUC-ROC must establish non-inferiority (recommended).**
- [ ] F1 primary; AUC-ROC key secondary with multiplicity rule: ___________

Proposed inference settings:

- one-sided non-inferiority alpha: 0.05;
- 95% confidence intervals also reported descriptively;
- paired-continuous power analysis using the conservative upper variance bound
  from the shared 5–8-seed development batch;
- final `n` is the larger requirement among endpoints that must pass.

## 5. RQ3 deployment and acceptance options

### 5.1 Proposed network profiles

These values define reproducible research scenarios, not claims that they
represent every real deployment.

| Profile | Uplink | RTT | Packet loss | Mandatory proposal |
| --- | ---: | ---: | ---: | --- |
| LAN | 100 Mbps | 5 ms | 0% | Yes |
| Broadband | 20 Mbps | 30 ms | 0.1% | Yes |
| Constrained/mobile | 5 Mbps | 80 ms | 1.0% | Contextual by default |

- [ ] Approve LAN and Broadband as mandatory; constrained/mobile contextual.
- [x] Make all three mandatory.
- [ ] Replace profiles with deployment-specific values: _________________

Network numeric values (LAN 100Mbps/5ms/0% loss; Broadband 20Mbps/30ms/0.1%
loss; Constrained 5Mbps/80ms/1% loss) approved as proposed.

### 5.2 Acceptance bundles

All latency figures are p95 steady-state values. Payload ratio is measured from
the emitted application message, not `numel * 4`.

| Criterion | A — strict | **B — balanced prototype** | C — feasibility only |
| --- | ---: | ---: | ---: |
| End-to-end round overhead vs raw | 10% | **25%** | 50% |
| Client encode/serialize | 100 ms/client | **250 ms/client** | 500 ms/client |
| Server decode/aggregate | 250 ms/round | **500 ms/round** | 1000 ms/round |
| Peak client memory overhead | 10% and 256 MB | **25% and 512 MB** | 50% and 1024 MB |
| Peak server memory overhead | 10% and 512 MB | **25% and 1024 MB** | 50% and 2048 MB |
| Application payload expansion | 2x | **6x** | 8x |
| Added transfer latency, LAN | 0.10 s | **0.25 s** | 0.50 s |
| Added transfer latency, Broadband | 0.50 s | **1.00 s** | 2.00 s |
| Added transfer latency, Constrained | 1.50 s | **3.00 s** | 6.00 s |
| Authentication/correctness failures | 0 | **0** | 0 |
| Timeout rate | 0% | **0%** | <=1% |

Recommendation: Bundle B for an academic prototype. The 6x payload ceiling is
not chosen from a benchmark outcome: an uncompressed one-byte-per-DNA-symbol
representation can expand binary data by roughly 4x before Base64, whose
encoding adds approximately another 4/3 factor, so a ceiling close to 6x tests
whether framing/metadata remain controlled. If the implemented representation
packs bases into two bits, the supervisor should lower the threshold.

- [ ] Bundle A.
- [x] **Bundle B (recommended).**
- [ ] Bundle C.
- [ ] Custom thresholds attached as: _____________________________________

Repetitions increased beyond the balanced default per supervisor request:
measured repetitions 50 (was 30), warm-up repetitions 10 (was 5).

Borderline rule proposal: a criterion passes only when its entire paired 95%
confidence interval is within the threshold. Missing mandatory cells yield
`INCONCLUSIVE`, not automatic acceptance.

- [x] Approve.
- [ ] Replace with: _______________________________________________________

## 6. Resource ceiling worksheet

These ceilings must be approved before Stage 1. They are not produced by power
analysis.

### 6.1 Available resources

| Resource | Supervisor/team entry |
| --- | --- |
| Hardware allowed | __________________________________________ |
| Concurrent workers allowed | __________________________________________ |
| Total wall-clock deadline | __________________________________________ |
| Maximum machine/GPU hours for RQ1 | ___________________________________ |
| Maximum machine/GPU hours for RQ2 | ___________________________________ |
| Maximum machine hours for RQ3 | _______________________________________ |
| Maximum new artifact storage | ________________________________________ |
| Required completion date | ____________________________________________ |

### 6.1 Available resources (as provided by the supervisor, 2026-09-12)

| Resource | Supervisor/team entry |
| --- | --- |
| Hardware allowed | MacBook Pro, Apple M1 Pro, 32GB RAM, full machine use; MPS (GPU) available for RQ2 training, not used by the RQ1 attacker (CPU-only by implementation contract, see `rq1_pre_pilot.yaml` benchmark_provenance) |
| Concurrent workers allowed | All available CPU cores, minus 1 reserved for the OS (~9 on this machine); each worker process single-threaded to avoid oversubscription |
| Total wall-clock deadline | 2026-10-04 |
| Maximum machine/GPU hours for RQ1 | 40 hours |
| Maximum machine/GPU hours for RQ2 | 15 hours |
| Maximum machine hours for RQ3 | 15 hours |
| Maximum new artifact storage | 500 GB (of >800GB currently free) |
| Required completion date | 2026-10-04 |

> **Superseding resource amendment (2026-09-13):** Manh Quang removed the
> machine-hour ceilings for RQ1, RQ2 and RQ3 while retaining the 2026-10-04
> deadline, 500 GB artifact ceiling, RQ1 maximum feasible target count 150 and
> RQ2 maximum feasible seed count 30. The original entries above remain as the
> historical 2026-09-12 decision. The operative rule is recorded in
> `amendments/2026-09-13_resource_ceiling_removed.md`.

### 6.2 Resulting frozen caps

| Field | Approved value | Basis |
| --- | ---: | --- |
| `maximum_feasible_target_count` (RQ1) | **150 targets** | Measured runtime probe 2026-09-12 (raw 15.26s, DNA-surrogate 15.85s, clipping/noise-MC 14.72s per restart at 600 iterations) → ~747s/target under the frozen R8/4-realization/R8 budget → 150 is ~3.4× `required_confirmatory_target_count` (44) at ~31 CPU-hours serial |
| `maximum_feasible_seed_count` (RQ2) | **30 seeds** | Measured runtime probe 2026-09-12 at official scale (500k rows, 3 clients): ~9.26s fixed overhead + ~4.49s/round → ~234s per method/seed at 50 rounds → 30 seeds × ~5 methods ≈ 10 CPU-hours serial |
| RQ3 measured repetitions per cell | **50** (increased from the balanced default of 30 per supervisor request) | Stability of median/p95 versus matrix cost |
| RQ3 warm-up repetitions per cell | **10** (increased from the balanced default of 5 per supervisor request) | Excluded from official summaries |

Runtime probes for RQ1 and RQ2 were executed 2026-09-12 on development-pool
data only (`development_gate_targets.pt`, group 0; and the official FL
baseline training pipeline at reduced round counts, extrapolated to 50
rounds) — no post-hoc or confirmatory targets were touched, and no privacy or
utility conclusion is drawn from these timing runs. Full derivations are
recorded in `config/rq1_pre_pilot.yaml` and `config/rq2_pre_pilot.yaml`
(`benchmark_provenance` blocks). RQ3's ceiling remains a conservative,
non-benchmarked buffer pending its own Stage-1 pilot.

Use a predeclared safety factor of at least 1.25 for orchestration failures and
artifact verification. The feasible cap must not be silently increased after
pilot outcomes. A formally approved pre-confirmatory resource amendment may
change it only without reference to effect direction or p-values.

## 7. Approval summary

| Decision | Selected value/option | Approved by | Timestamp |
| --- | --- | --- | --- |
| Alpha / power / confidence | 0.05 / 0.80 / 95% | Manh Quang | 2026-09-12 |
| RQ1 `p0`, `p1`, ties/dropout | 0.50 / 0.70 / 10% / 5% | Manh Quang | 2026-09-12 |
| RQ1 multiplicity/tie rule | Approved as proposed (DNA-vs-DP primary, Holm for secondary) | Manh Quang | 2026-09-12 |
| RQ2 margins and endpoint rule | Bundle B (F1 0.02, AUC 0.005), both endpoints must pass | Manh Quang | 2026-09-12 |
| RQ3 profiles, bundle and decision rule | All 3 profiles mandatory, Bundle B, 50 measured/10 warm-up | Manh Quang | 2026-09-12 |
| RQ1/RQ2 resource ceilings | RQ1: 150 targets / 40h; RQ2: 30 seeds / 15h; RQ3: 15h; storage 500GB | Manh Quang | 2026-09-12 |
| Machine-hour ceiling amendment | No machine-hour ceiling for RQ1/RQ2/RQ3; all other caps and deadline unchanged | Manh Quang | 2026-09-13 |
| Pre-pilot config hashes | `attacks/inversion_metrics.py`=6ad0b2581d3..., `attacks/pseudo_image.py`=6273e59d73c..., code tree 65afddc3 | Manh Quang | 2026-09-12 |

Approval of this table plus the referenced config hashes authorizes changing
the protocols to `PRE-PILOT FROZEN`. It does not authorize confirmatory target
generation or official experiments.

**Status: all three pre-pilot configs (`rq1_pre_pilot.yaml`, `rq2_pre_pilot.yaml`,
`rq3_pre_pilot.yaml`) are now `PRE-PILOT FROZEN` and `execution_authorized: true`.**
Stage 1 (development audit, comparator calibration, pilot seed batch, runtime
finalization) may begin. The original pre-pilot package required later
resolution of the tie threshold, paired-variance sample size and provenance.
RQ1/RQ2 provenance is now closed by the file-level SHA-256 manifest plus the
recorded git commit/tree reference; a later repository commit remains useful
history but is not substituted for, or silently treated as, the signed file
manifest. The formerly open RQ3 network-emulation-tool choice was closed before
official measurement by
`amendments/2026-09-13_rq3_network_emulator_and_freeze.md`; the official result
is reported separately in `../reports/rq3_report.md`.

## 8. Supervisor notes

The post-Stage-1 approvals dated 2026-09-13 are recorded without changing the
original pre-pilot decision history in
`amendments/2026-09-13_stage1_supervisor_approvals.md`. They approve the RQ1
tie threshold and distortion comparator, RQ2 `n=21`, the terminal
`DP_UTILITY_MATCHED=NOT_FOUND` result, and the precommitted DNA Transform F1
interpretation rule. Confirmatory attack/training execution remains subject to
the requested final review of the generated freeze package.

___________________________________________________________________________

___________________________________________________________________________

___________________________________________________________________________
