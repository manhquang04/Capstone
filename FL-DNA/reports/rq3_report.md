# RQ3 Group 1 — Official deployment-cost benchmark report

**Run ID:** `rq3_benchmark_20260913T034500Z`  
**Execution date:** 2026-09-13  
**Protocol/config:** `RQ3-COST-ACCEPTANCE-V1` /
`RQ3-CONFIRMATORY-BENCHMARK-V1-FROZEN`  
**Decision:** **NOT_ACCEPTABLE** under frozen Bundle B  

## 1. Đã làm gì

1. Gỡ riêng trần `maximum_machine_hours` cho RQ1/RQ2/RQ3 theo quyết định của
   supervisor, giữ deadline 2026-10-04, storage 500 GB, RQ1 target cap 150 và
   RQ2 seed cap 30.
2. Kiểm tra công cụ host. `dnctl` và `pfctl` tồn tại nhưng không thể mở socket/
   `/dev/pf` trong môi trường không đặc quyền.
3. Khóa emulator user-space có version/hash: truyền đầy đủ application payload
   qua loopback socket, kiểm tra byte hash, rồi áp bandwidth/RTT/loss và reliable
   retransmission theo profile đã duyệt.
4. Smoke-test ba method trên fixture riêng; tất cả round trip đúng và không
   timeout. Timing smoke không được dùng để thay đổi threshold hay matrix.
5. Đóng băng config official rồi chạy tuần tự đúng 27 cell: 3 profile × 3
   method × 3 client scale; mỗi cell 10 warm-up + 50 measured repetitions.
6. Chạy analyzer đã hash với paired percentile bootstrap 10.000 resamples và
   áp nguyên Bundle B.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-13_resource_ceiling_removed.md`
- `protocols/amendments/2026-09-13_rq3_network_emulator_and_freeze.md`
- `protocols/config/rq1_pre_pilot.yaml`
- `protocols/config/rq2_pre_pilot.yaml`
- `protocols/config/rq3_pre_pilot.yaml`
- `protocols/config/rq1_confirmatory.yaml`
- `protocols/config/rq2_confirmatory.yaml`
- `protocols/config/rq3_benchmark.yaml`
- `protocols/rq3_acceptance_criteria.md`
- `protocols/rq3_confirmatory_freeze_manifest_2026-09-13.json`
- `protocols/rq3_execution_manifest_2026-09-13.json`
- `experiments/rq3_user_space_network.py`
- `experiments/run_rq3_benchmark.py`
- `experiments/analyze_rq3_benchmark.py`
- `protocols/config/rq3_smoke.json`

Không sửa `torch.set_num_threads(1)`; runner RQ3 gọi rõ chính sách này và
environment artifact xác nhận giá trị thực tế bằng 1.

## 3. Lệnh thực sự đã chạy

```bash
/sbin/pfctl -s info
/usr/sbin/dnctl list

.venv-phase1/bin/python -m py_compile \
  experiments/rq3_user_space_network.py \
  experiments/run_rq3_benchmark.py \
  experiments/analyze_rq3_benchmark.py

.venv-phase1/bin/python experiments/run_rq3_benchmark.py \
  --config protocols/config/rq3_smoke.json \
  --output /tmp/rq3_smoke_pre_freeze.ruMZyp/output

.venv-phase1/bin/python experiments/run_rq3_benchmark.py \
  --config protocols/config/rq3_benchmark.yaml \
  --output artifacts/rq3/benchmark_20260913T034500Z

.venv-phase1/bin/python experiments/analyze_rq3_benchmark.py \
  --config protocols/config/rq3_benchmark.yaml \
  --run artifacts/rq3/benchmark_20260913T034500Z \
  --results results/rq3/benchmark_20260913T034500Z
```

`pfctl` trả `Permission denied`; `dnctl` trả `Operation not permitted`. Đây là
evidence cho lựa chọn emulator, không phải benchmark observation.

## 4. Kết quả kèm uncertainty

Integrity/completeness:

- 27/27 cell và 27/27 subprocess return code 0;
- 270 warm-up observations, tách khỏi 1.350 measured observations;
- 0 authentication/correctness failure và 0 timeout trong cả warm-up lẫn
  measured;
- config SHA-256 khi chạy khớp freeze: `8997e2f75343...`;
- thời gian wall-clock của ma trận khoảng 55 phút (10:44–11:39 UTC+7).

P95 end-to-end seconds (`RAW / LOSSLESS_DNA / DNA_TRANSFORM`):

| Profile | Clients | Raw | Lossless DNA | DNA Transform |
| --- | ---: | ---: | ---: | ---: |
| LAN | 3 | 0.0612 | 1.0342 | 0.8046 |
| LAN | 5 | 0.0890 | 1.7282 | 1.3083 |
| LAN | 10 | 0.1651 | 3.3678 | 2.6481 |
| Broadband | 3 | 0.2521 | 1.4517 | 0.9733 |
| Broadband | 5 | 0.3678 | 2.3530 | 1.5887 |
| Broadband | 10 | 0.6900 | 4.6339 | 3.1220 |
| Constrained | 3 | 0.7303 | 2.7878 | 1.4883 |
| Constrained | 5 | 1.1905 | 4.6071 | 2.4098 |
| Constrained | 10 | 2.2851 | 9.1619 | 4.7452 |

Bundle-B criterion coverage across the 18 non-raw cells:

| Criterion | Passed cells | Observed statistic range | Frozen threshold |
| --- | ---: | ---: | ---: |
| End-to-end p95 overhead fraction | 0/18 | 1.316–23.244 | ≤0.25 |
| Client encode/serialize p95 | 8/18 | 219.6–267.0 ms | ≤250 ms |
| Server decode/aggregate p95 | 15/18 | 0.96–902.1 ms | ≤500 ms |
| Client memory overhead fraction | 0/18 | 0.624–10.398 | ≤0.25 |
| Client memory overhead absolute | 18/18 | 0.158–4.102 MB | ≤512 MB |
| Server memory overhead fraction | 9/18 | 0.00023–4.680 | ≤0.25 |
| Server memory overhead absolute | 18/18 | 0.00021–1.546 MB | ≤1024 MB |
| Application payload expansion | 18/18 | 1.000–5.271× | ≤6× |
| Added transfer latency p95 | 16/18 | 0.005–4.242 s | profile-specific |
| Correctness/auth failures | 18/18 | 0 | 0 |
| Timeout rate | 18/18 | 0 | 0% |

Paired bootstrap CIs make the decisive failures unambiguous. Examples:

- LAN/Lossless/3 end-to-end overhead fraction: p95 `19.193`, 95% CI
  `[18.071, 19.902]`, threshold `0.25`.
- Broadband/Transform/3: p95 `3.898`, 95% CI `[3.792, 3.944]`.
- Constrained/Transform/10: p95 `1.316`, 95% CI `[1.278, 1.326]`.
- Constrained/Lossless/10 added transfer latency: p95 `4.242 s`, 95% CI
  `[4.127, 4.277]`, threshold `3 s`.
- Broadband/Lossless/10 added transfer latency is borderline-failing under the
  precommitted entire-CI rule: p95 `0.984 s`, CI upper `1.010 s`, threshold
  `1 s`.

Lossless DNA's application payload ratio is `5.2706×`, inside the `6×` ceiling;
DNA Transform uses the raw binary framing and is `1.0×`. Passing payload and
correctness criteria cannot override failures in mandatory latency/memory
criteria under the all-mandatory rule.

## 5. Gate đạt/chưa đạt và lý do

| Gate | Status | Evidence/reason |
| --- | --- | --- |
| Resource amendment | PASS | Only machine-hour ceiling removed; statistical/storage caps unchanged |
| Emulator/tool freeze | PASS | Tool versioned, smoke-tested, hashed before official run |
| Official config integrity | PASS | All implementation hashes and runtime config hash matched |
| Matrix completeness | PASS | 27/27 cells; 1.350/1.350 measured observations |
| Correctness/authentication | PASS | 0 failures |
| Timeout | PASS | 0 timeouts |
| Bundle B — LAN | **NOT_ACCEPTABLE** | Mandatory overhead/memory criteria fail |
| Bundle B — Broadband | **NOT_ACCEPTABLE** | Mandatory overhead/memory criteria fail |
| Bundle B — Constrained | **NOT_ACCEPTABLE** | Mandatory overhead/memory criteria fail |
| Overall RQ3 acceptance | **NOT_ACCEPTABLE** | At least one mandatory criterion fails; in fact all profiles contain clear failures |

RQ3's scoped answer is therefore: under the frozen prototype implementation,
sequential-client semantics, application-payload accounting and three mandatory
emulated profiles, the computational/bandwidth package is **not within the
approved acceptable range**. This does not claim that every optimized future
implementation or real distributed transport would fail.

Peak memory was measured as peak Python allocation with `tracemalloc`, not total
host RSS. Absolute overhead is small, while percentage overhead is large because
the raw serializer allocation baseline is small. This instrumentation boundary
was frozen and must be stated alongside the result.

## 6. Artifact/run ID

- Run: `artifacts/rq3/benchmark_20260913T034500Z/`
- Results: `results/rq3/benchmark_20260913T034500Z/`
- Artifact tree SHA-256: `2a39ce2a9a02...`
- Result tree SHA-256: `cd357a232bec...`
- Decision file SHA-256: `8a56ddabf7e9...`
- Execution manifest: `protocols/rq3_execution_manifest_2026-09-13.json`

## 7. Amendment

- `2026-09-13_resource_ceiling_removed.md`: supervisor-driven resource change,
  unrelated to observed effect direction.
- `2026-09-13_rq3_network_emulator_and_freeze.md`: tool/transport/uncertainty
  choices recorded before the official benchmark.

Không có amendment sau khi xem official results và không rerun cell theo
outcome.

## 8. Bước tiếp theo được phép

Group 1 đã có báo cáo nên protocol cho phép bắt đầu Group 2: calibration riêng
cho DNA medium/stronger, freeze từng comparator/power/target set source-disjoint,
Level 2 trên đúng conservative target n=39, và utility-matched grid extension
theo amendment pre-run. Group 3 và Group 4 vẫn phải chờ báo cáo của nhóm đứng
trước theo thứ tự supervisor đã khóa.

