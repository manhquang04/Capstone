# RQ3 DNA Transform v2 transport benchmark report

**Date:** 2026-09-16  
**Status:** COMPLETE — DNA_TRANSFORM_V2_TRANSPORT is NOT_ACCEPTABLE under Bundle B

## Đã làm gì

Ran the supervisor-authorized RQ3 benchmark extension for DNA Transform v2
transport. The benchmark used the frozen Bundle B criteria, three network
profiles, three client scales, and three methods:

- `RAW_FLOAT32`
- `DNA_TRANSFORM_TRANSPORT` (v1 reference method in the same matrix)
- `DNA_TRANSFORM_V2_TRANSPORT` (`compression_ratio=0.95`, `quantization_eta=0.01`)

Each cell used 10 warm-up repetitions and 50 measured repetitions. The run was
executed in isolation; Priority 5 / IEEE-CIS work was not started during this
benchmark to avoid timing interference.

One earlier audit run at `artifacts/rq3/v2_transport_benchmark_20260916/` was
aborted before any cell result was produced because the amendment text still
said "nine-cell" while the frozen matrix was 27 cells. That audit artifact is
excluded from this analysis. The corrected official run is
`artifacts/rq3/v2_transport_benchmark_20260916_rerun/`.

## Lệnh thực sự đã chạy

```bash
PYTHONPATH=. .venv-phase1/bin/python experiments/run_rq3_benchmark.py \
  --config protocols/config/rq3_v2_benchmark.json \
  --output artifacts/rq3/v2_transport_benchmark_20260916_rerun

PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq3_benchmark.py \
  --config protocols/config/rq3_v2_benchmark.json \
  --run artifacts/rq3/v2_transport_benchmark_20260916_rerun \
  --results results/rq3/v2_transport_benchmark_20260916
```

## Kết quả kèm uncertainty

The analyzer produced 27/27 completed cells and `COMPLETED.json`. The overall
matrix decision was `NOT_ACCEPTABLE`; both DNA transport methods were
`NOT_ACCEPTABLE` in all 9 non-raw cells. The v2-specific decision was
`NOT_ACCEPTABLE` in 9/9 cells.

| Profile | Clients | v2 decision | End-to-end overhead p95 | Encode/serialize p95 | Server decode/aggregate p95 | Payload expansion |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| LAN | 3 | NOT_ACCEPTABLE | 19.02x | 170.2 ms | 500.2 ms | 1.317x |
| LAN | 5 | NOT_ACCEPTABLE | 21.53x | 178.9 ms | 836.0 ms | 1.317x |
| LAN | 10 | NOT_ACCEPTABLE | 23.83x | 190.9 ms | 1719.0 ms | 1.317x |
| BROADBAND | 3 | NOT_ACCEPTABLE | 5.47x | 183.4 ms | 513.6 ms | 1.317x |
| BROADBAND | 5 | NOT_ACCEPTABLE | 5.51x | 178.8 ms | 851.6 ms | 1.317x |
| BROADBAND | 10 | NOT_ACCEPTABLE | 6.18x | 178.2 ms | 1704.1 ms | 1.317x |
| CONSTRAINED | 3 | NOT_ACCEPTABLE | 2.11x | 168.6 ms | 508.2 ms | 1.317x |
| CONSTRAINED | 5 | NOT_ACCEPTABLE | 2.15x | 171.3 ms | 846.7 ms | 1.317x |
| CONSTRAINED | 10 | NOT_ACCEPTABLE | 1.98x | 182.5 ms | 1694.5 ms | 1.317x |

Primary Bundle B failures for v2:

- `end_to_end_round_overhead_p95_fraction`: failed in 9/9 v2 cells.
- `server_decode_aggregate_p95_ms`: failed in 9/9 v2 cells.
- `peak_client_memory_overhead_fraction`: failed in 9/9 v2 cells.
- `peak_server_memory_overhead_fraction`: failed in 9/9 v2 cells.

Bundle B criteria that v2 passed across all cells:

- `client_encode_serialize_p95_ms` passed in 9/9 v2 cells.
- `added_transfer_latency_p95_seconds` passed in 9/9 v2 cells.
- `application_payload_expansion_ratio` passed in 9/9 v2 cells.
- `authentication_or_correctness_failures` passed in 9/9 v2 cells.
- `timeout_rate_fraction` passed in 9/9 v2 cells.

## Gate đạt/chưa đạt

| Gate | Status | Reason |
| --- | --- | --- |
| Matrix completion | PASS | 27/27 cells completed; `COMPLETED.json` present |
| Isolation requirement | PASS | no IEEE-CIS or other experiment workload was launched during benchmark |
| Analyzer execution | PASS | analyzer produced summary CSV, criteria CSV, and decision JSON |
| Bundle B acceptance for v2 | FAIL | v2 failed mandatory overhead and memory-fraction criteria in all 9 v2 cells |
| RQ3-v2 conclusion | NOT_ACCEPTABLE | mandatory criteria require every mandatory criterion to pass |

## Artifact/run ID

- Amendment: `protocols/amendments/2026-09-16_rq3_v2_transport_benchmark.md`
- Config: `protocols/config/rq3_v2_benchmark.json`
- Official run: `artifacts/rq3/v2_transport_benchmark_20260916_rerun/`
- Analysis: `results/rq3/v2_transport_benchmark_20260916/`
- Excluded aborted audit run: `artifacts/rq3/v2_transport_benchmark_20260916/`

Hashes recorded after analysis:

- `protocols/config/rq3_v2_benchmark.json`: `8d968e21a155a26b707dd4b2fc28299505966e78c0fe935d0b86f76211a82bfc`
- `experiments/run_rq3_benchmark.py`: `41e0d449babdf847ebe53764710a09749e84e877c6cd993c1d9708313463489c`
- `artifacts/rq3/v2_transport_benchmark_20260916_rerun/COMPLETED.json`: `ceedac5888afbff23781ae0e3ee07185ec638305a932d008c59cc98a3edbb03a`

## Conclusion and next permitted step

Under the frozen Bundle B thresholds, DNA Transform v2 transport is
**NOT_ACCEPTABLE** for RQ3. The payload expansion ratio is acceptable and there
were no correctness failures or timeouts, but the current v2 implementation has
too much end-to-end overhead and server-side decode/aggregate cost.

The next permitted task is Priority 5 / IEEE-CIS RQ1 development gate, starting
from the frozen amendment
`protocols/amendments/2026-09-16_priority5_ieee_cis_rq1_development_gate.md`.
No confirmatory RQ3-v2 rerun is authorized without a new amendment written
before any new benchmark.
