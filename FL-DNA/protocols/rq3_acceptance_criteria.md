# RQ3 Protocol — Deployment Cost and Acceptance Criteria

**Protocol ID:** `RQ3-COST-ACCEPTANCE-V1`  
**Version:** `1.1-confirmatory-frozen`  
**Status:** `CONFIRMATORY FROZEN — SINGLE OFFICIAL BENCHMARK AUTHORIZED`  
**Parent plan:** `../../PROJECT.md`  
**Required approval:** research team and supervisor  

## 1. Research question

> Are the computational and bandwidth costs of DNA encoding within an
> acceptable range for real-world deployment?

This protocol defines what is measured and what “acceptable” means before the
official benchmark is run. Existing 3/5/10-client timings are pilot evidence
only and must not determine the thresholds retrospectively.

CPU utilization may be collected as descriptive instrumentation when reliable,
but energy and power consumption are out of scope for this protocol. Therefore,
the final RQ3 claim is limited to latency, compute-time, memory and bandwidth
costs; it must not be generalized to energy efficiency, battery impact or
carbon footprint. Adding an energy claim requires a separately approved
amendment with calibrated power instrumentation and predeclared criteria.

## 2. Deployment scenarios

Acceptance is evaluated separately for three frozen network profiles. The
numbers below describe fields that must be approved; proposed profile values
are deliberately not treated as final requirements.

| Profile | Intended setting | Down/up bandwidth | RTT | Loss |
| --- | --- | ---: | ---: | ---: |
| `LAN` | same-site/high-speed deployment | 100 Mbps uplink | 5 ms | 0% |
| `BROADBAND` | typical remote client | 20 Mbps uplink | 30 ms | 0.1% |
| `CONSTRAINED` | mobile/limited uplink | 5 Mbps uplink | 80 ms | 1.0% |

The team must state which profiles are mandatory deployment targets. Results
must not be generalized to networks outside the frozen profiles.

## 3. Methods

1. `RAW_FLOAT32`: actual baseline serialization and framing.
2. `LOSSLESS_DNA_AES_GCM`: float32 → binary → DNA → AES-256-GCM → actual
   Base64/JSON/message framing, followed by the complete inverse path.
3. `DNA_TRANSFORM_TRANSPORT`: frozen RQ1 DNA Transform followed by the actual
   transmission representation used by the system.

If the production-intended wire format differs from Base64/JSON, both formats
must have unique method IDs. Compression may be included only if it is actually
implemented symmetrically and its CPU cost is included.

## 4. Benchmark matrix

Required client counts are 3, 5 and 10. Evaluate:

- fixed total data while client count changes;
- fixed per-client data, if compute permits, to expose scaling behavior.

Every method/scale/profile cell uses the same model-update tensors and logical
round inputs. Freeze:

```yaml
warmup_repetitions: 10
measured_repetitions: 50
process_isolation: one_measured_cell_at_a_time_no_concurrent_benchmark_load
method_order_randomization: randomized_or_balanced_seed_271828
cpu_thread_policy: single_thread_per_process
network_emulation_tool: PENDING_STAGE1_BEFORE_RQ3_PILOT
timeout_seconds: 30
```

One timing sample is not sufficient. Repetition count must support stable
median and p95 estimates under the frozen machine/environment.

## 5. Measurement boundaries

Use a monotonic high-resolution timer and retain raw repetitions. Measure:

### 5.1 Client side

- raw tensor serialization;
- float-to-binary conversion;
- DNA mapping;
- DNA Transform separately;
- AES-GCM encryption;
- Base64/JSON or actual framing;
- total preparation-to-send latency;
- CPU utilization as an optional descriptive metric, not an acceptance
  criterion;
- peak process RSS and peak payload allocation where measurable.

### 5.2 Server side

- framing parse/Base64 decode;
- AES-GCM authentication/decryption;
- DNA decode/binary-to-float restoration;
- inverse transform or server-side handling actually used;
- aggregation;
- total receive-to-aggregated latency;
- peak process RSS.

### 5.3 End to end

- local training time, reported separately;
- client encode/serialization;
- actual or emulated transfer;
- server decode/aggregation;
- complete round wall-clock;
- failures, authentication errors and timeouts.

Do not call encoding time network latency. Report warm-up separately from
steady-state measurements.

## 6. Wire-byte accounting

Measure actual emitted bytes, including:

- tensor values and dtype/shape metadata;
- DNA ciphertext;
- nonce and AES-GCM authentication tag per encrypted unit;
- Base64 expansion;
- JSON/message framing and field names;
- compression headers, if used;
- transport/application framing visible to the benchmark where measurable.

Report per client and round:

```text
payload_bytes
payload_bytes_total
payload_expansion_ratio_vs_raw
metadata_fraction
bytes_per_model_parameter
```

`tensor.numel * 4` is a raw tensor lower bound, not the measured wire payload.
If protocol/TLS headers cannot be observed, label the value
`APPLICATION_PAYLOAD_BYTES`, state the exclusion and do not call it total wire
traffic.

## 7. Network evaluation

Prefer real distributed hosts when reproducible. Otherwise use a documented
network emulator. For each frozen profile report:

- configured bandwidth, RTT, loss and queue assumptions;
- measured upload/round latency and effective throughput;
- communication fraction of total round time;
- timeouts/failures and retry behavior;
- analytical transfer-time cross-check from measured application bytes.

All methods must run under the same profile order and emulator state. Record
host clocks and do not infer one-way delay from unsynchronized clocks.

## 8. Acceptance thresholds

All thresholds below must receive numerical values and approval before official
benchmark data are generated or inspected.

| Criterion | Statistic | Frozen threshold | Mandatory profiles |
| --- | --- | ---: | --- |
| End-to-end round overhead | p95 % versus raw | 25% | all |
| Client encode/serialize latency | p95 ms/client | 250 ms | all |
| Server decode/aggregate latency | p95 ms/round | 500 ms | all |
| Peak client memory overhead | p95 MB and % | 512 MB and 25% | all |
| Peak server memory overhead | p95 MB and % | 1024 MB and 25% | all |
| Application payload expansion | exact ratio versus raw | 6x | all |
| Added transfer latency | p95 per round | LAN 0.25 s; Broadband 1 s; Constrained 3 s | all |
| Authentication/correctness failures | count | 0 | all |
| Timeout rate | proportion | 0% | all |

Threshold rationale must cite a deployment requirement, service-level target,
hardware/network constraint or supervisor-approved research convention. It
must not cite the benchmark result it will judge.

## 9. Decision rule

Freeze one rule before measurement:

```yaml
acceptance_rule: all_mandatory_criteria_pass_all_profiles_and_scales
handling_of_borderline_ci: entire_paired_95pct_ci_within_threshold
handling_of_missing_cell: INCONCLUSIVE
```

Recommended conservative interpretation:

- `ACCEPTABLE`: every mandatory criterion passes.
- `NOT_ACCEPTABLE`: at least one mandatory criterion clearly exceeds its
  threshold.
- `INCONCLUSIVE`: required cells are missing or uncertainty overlaps a threshold
  under the frozen borderline rule.

If different profiles produce different decisions, report per-profile status
and do not collapse them into an unconditional real-world claim.

## 10. Statistical summaries

For every cell publish raw repetitions and:

- count, mean, median, SD, IQR and p95;
- paired absolute and percentage overhead versus raw;
- a confidence interval for the paired overhead using the frozen method;
- failure/timeout count and rate;
- scaling slope/model with uncertainty, used descriptively unless separately
  powered.

Use paired input tensors and, where possible, interleaved/balanced execution.
Do not delete thermal-throttled or slow observations unless a documented
infrastructure failure occurred. Report excluded repetitions and reasons.

## 11. Environment control

Record:

- CPU, RAM, storage, optional GPU and power mode;
- OS, Python, packages and cryptographic backend;
- process/thread affinity and background-load policy;
- thermal state/temperature information if available;
- model parameter count, tensor count and tensor-size distribution;
- code, dataset/input, protocol and config hashes;
- network/emulator versions and exact commands.

Run a preflight that verifies bit-exact lossless decode and AES-GCM
authentication before timing. Correctness failures invalidate the affected
benchmark cell; faster incorrect output never passes acceptance.

## 12. Pilot versus official benchmark

Pilot runs may estimate repetition count, validate instrumentation and detect
gross instability. They must use a separately labeled output directory and may
not be pooled with official measurements. If pilot evidence changes the
protocol, issue and approve an amendment before official execution.

Official benchmark inputs, thresholds, repetition count and environment must be
frozen and hashed in a separate confirmatory benchmark config. Existing
3/5/10-client results remain historical context.

## 13. Required outputs

```text
protocols/rq3_acceptance_criteria.md
protocols/config/rq3_pre_pilot.yaml
protocols/config/rq3_benchmark.yaml
artifacts/rq3/pilot/
artifacts/rq3/benchmark_<run_id>/
results/rq3/compute_summary.csv
results/rq3/payload_summary.csv
results/rq3/network_summary.csv
results/rq3/acceptance_decision.json
reports/rq3_report.md
```

## 14. Two-gate freeze checklist

### 14.1 `PRE-PILOT FROZEN`

- [x] All three network profiles and mandatory-profile set approved.
- [x] Every acceptance threshold has a numerical value and rationale.
- [x] Overall, borderline and missing-cell decision rules approved.
- [x] Methods use actual production-intended serialization/framing.
- [x] Matrix, warm-ups, repetitions and execution ordering frozen.
- [x] Timing boundaries and byte-accounting exclusions documented.
- [x] Energy/power is explicitly reported as out of scope unless added by an
  approved pre-benchmark amendment.
- [x] Network-emulation tool must be locked before the RQ3 pilot; by amendment,
  this does not block Stage 1 work for RQ1/RQ2.
- [x] Correctness preflight and failure rules machine-readable.
- [x] Pre-pilot config and code/environment/protocol hashes recorded.
- [x] Team and supervisor pre-pilot approvals recorded below.

### 14.2 `CONFIRMATORY FROZEN`

- [x] Pilot instrumentation results reviewed only under predeclared rules.
- [x] Any amendment completed before official benchmark measurements.
- [x] Official repetitions, inputs, environment and method order frozen.
- [x] Separate official benchmark config and all hashes recorded.
- [x] Team and supervisor official-benchmark approvals recorded below.

## 15. Approval record

| Gate | Role | Name | Decision | Timestamp | Signature/reference |
| --- | --- | --- | --- | --- | --- |
| Pre-pilot | Research lead | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Pre-pilot | Supervisor | Manh Quang | Approved | 2026-09-12 | Chat session approval; decision package v1.0 |
| Official benchmark | Research lead | Manh Quang | Approved | 2026-09-13 | Group 1 execution instruction; RQ3 emulator amendment |
| Official benchmark | Supervisor | Manh Quang | Approved | 2026-09-13 | Group 1 execution instruction; RQ3 emulator amendment |

`DRAFT` authorizes writing/instrumentation preparation only.
`PRE-PILOT FROZEN` authorizes the pilot benchmark. Only
`CONFIRMATORY FROZEN` authorizes the official benchmark and an
“acceptable/not acceptable” conclusion.
