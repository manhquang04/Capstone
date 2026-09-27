# Báo cáo thực thi Stage 1 — 2026-09-12

**Phạm vi:** RQ1 audit/smoke/calibration/power và RQ2 shared development batch  
**Trạng thái:** `STAGE 1 RQ1/RQ2 COMPLETE — CONFIRMATORY EXECUTION NOT AUTHORIZED`  
**Dữ liệu post-hoc đã dùng:** không  
**Confirmatory target đã tạo:** không

## 1. Gate trước thực thi

`PRE-PILOT EXECUTION GATE = PASSED` cho RQ1/RQ2. Ba YAML khớp decision
package, hai implementation hash RQ1 và dataset checksum đã được xác minh, và
freeze manifest post-sync pass. RQ3 pilot vẫn giữ riêng ở trạng thái chờ
`network_emulation_tool`; điều này không chặn RQ1/RQ2 theo amendment ngày
2026-09-12.

## 2. Đã làm gì

### RQ1

1. Audit artifact lịch sử, không sửa artifact cũ. Raw historical artifacts
   thuộc `development_targets.pt` (4 groups), trong khi DNA/DP-MC development
   artifacts thuộc `development_gate_targets.pt` (8 groups). Sau khi đối chiếu
   đúng target:
   - raw: `80/80` artifact đủ vector, source IDs đúng, finite và metric replay;
   - DNA Level-1: `512/512` pass;
   - clipping/noise-MC: `128/128` pass.
2. Bổ sung metadata bằng sidecar mới: feature order, numeric center/scale,
   categories, pseudo-image/metric hashes, checkpoint/dataset hashes. Không
   ghi đè `.pt` lịch sử.
3. Smoke raw, DNA Level-1 và clipping/noise-MC trên bản sao SHA-256-identical
   của `development_gate_targets.pt`, group 0, 600 iterations. Cả ba branch có
   đủ vector và replay pass; smoke `n=1` không được diễn giải như statistical
   branch gate.
4. Chạy initial distortion grid đầy đủ. Grid gốc không có candidate trong 5%
   tolerance; amendment thêm ba điểm nội suy và giữ toàn bộ grid cũ.
5. Đồng bộ implementation comparator: thêm chế độ backward-compatible
   `clip_norm_noise_multiplier` cho MC attacker. Calibration được replay bằng
   shared implementation này.
6. Tie numerical-stability study trên `640` artifact development. Sai khác
   replay lớn nhất `0.00390625`; rule kỹ thuật được thử cho threshold đề xuất
   `0.0390625`. Vì YAML ban đầu chưa định nghĩa cụ thể safety factor, giá trị
   này cần supervisor phê duyệt trước confirmatory freeze.
7. RQ1 exact-binomial power tái tạo `n=37` non-tied, rejection `>=24`, actual
   alpha `0.0494359`, power `0.8070957`, planning draw `44`. `44 <= 150`, nên
   không underpowered theo planning assumptions đã duyệt.

### RQ2

1. Materialize riêng checksum split/partition/model-init cho 8 seed
   `[101,202,303,404,505,606,707,808]`. Có 8 split checksum khác nhau; replay
   seed 101 khớp toàn bộ checksum.
2. Sửa seed plumbing và RNG isolation trước khi xem pilot outcome:
   `FL_RUN_SEED` điều khiển data/split/partition/init/order; DNA Transform seed
   suy ra từ seed này; DP noise dùng generator namespace riêng, không advance
   training/dropout RNG.
3. Shared batch hoàn tất `64/64` core jobs: 8 seed x Baseline, DNA lossless,
   DNA Transform và 5 DP utility-grid candidates. Sau RQ1 calibration, chạy
   thêm `8/8` job cho contextual `DP_DISTORTION_MATCHED=0.00025`; candidate này
   không được đưa vào utility-selection grid.
4. Một lỗi hạ tầng ban đầu (`torch` import thiếu trong DP runner) tạo 23 failed
   startup records và 8 interrupted attempts, không tạo DP utility outcomes.
   Tất cả 31 attempt được giữ dưới `attempts/`; retry dùng đúng seed/config và
   hoàn tất. Final registry: `72/72 SUCCESS`.
5. Replay seed 101 cho Baseline, DNA lossless, DNA Transform và DP `0.0001`
   khớp bit-exact ở mọi core metric qua 50 round; timing-only fields được loại
   khỏi equality gate.

## 3. Kết quả kèm uncertainty

### RQ1 distortion calibration

| Đại lượng | Kết quả |
| --- | ---: |
| DNA median relative-L2 | 0.07557750 |
| DP selected clip norm | 100 |
| DP selected multiplier | 0.00025 |
| DP median relative-L2 | 0.07808418 |
| Relative mismatch | 3.3167% |
| Frozen tolerance | 5% |
| Development match | PASS |

Candidate trên chỉ là đề xuất development cho confirmatory config; chưa được
gọi là confirmatory-frozen.

### RQ2 paired variance và power

Các CI dưới đây là paired t 95% CI mô tả trên 8 development seeds. Cỡ mẫu dùng
cận trên một phía 95% của paired variance và one-sided non-inferiority
noncentral-t approximation tại alpha `0.05`, power `0.80`.

| Method / endpoint | Mean paired delta | 95% CI | SD | Required n |
| --- | ---: | ---: | ---: | ---: |
| DNA lossless / F1 | -0.001023 | [-0.014340, 0.012294] | 0.015929 | 15 |
| DNA lossless / AUC | -0.000636 | [-0.002355, 0.001084] | 0.002057 | 6 |
| DNA Transform / F1 | -0.012430 | [-0.028601, 0.003742] | 0.019343 | 21 |
| DNA Transform / AUC | -0.000387 | [-0.001280, 0.000507] | 0.001069 | 3 |

Rule “cả F1 và AUC phải đạt” cho required count lớn nhất `21`; `21 <= 30`, nên
không vượt resource ceiling. Đây là sample-size input, không phải kết luận
non-inferiority từ development data.

### DP utility matching

Không candidate nào trong grid khóa trước đạt đồng thời tolerance F1 `0.01` và
AUC `0.0025`; do đó `DP_UTILITY_MATCHED = NOT_FOUND`. Candidate gần nhất theo
distance là `0.0001`, nhưng F1 deviation so với DNA Transform là `0.027144 >
0.01`, nên không được relabel thành matched. Contextual distortion-matched
`0.00025` có mean delta F1 `-0.0702491` và AUC `-0.00217124`; nó bị loại khỏi
utility selection theo thiết kế.

## 4. Artifact/run IDs

- `stage1_historical_raw_audit_20260912`
- `stage1_historical_defense_audit_20260912`
- `stage1_smoke_20260912` và `audit_shared_dp_impl`
- `stage1_distortion_calibration_shared_impl_initial_20260912`
- `stage1_distortion_calibration_shared_impl_expanded_20260912`
- `stage1_tie_threshold_20260912`
- `stage1_power_analysis_20260912`
- `stage1_development_20260912` (RQ2; 72 successful jobs)

## 5. File/config/code thay đổi

- Administrative/pre-pilot sync: xem
  `amendments/2026-09-12_pre_pilot_administrative_sync.md`.
- RQ2 seed/power implementation và failed-run retry:
  `amendments/2026-09-12_stage1_seed_and_power_implementation.md`.
- RQ1 grid expansion và DP attacker implementation alignment:
  `amendments/2026-09-12_rq1_distortion_grid_expansion.md`.
- Seed/RNG code: `data/load_creditcard.py`, `experiments/fraud_fl_common.py`,
  `privacy/dp_engine.py`, `experiments/run_fraud_fl_dp.py`,
  `experiments/run_fraud_fl_dna_dp.py`,
  `experiments/run_fraud_fl_dna_transform.py`.
- Versioned Stage-1 tooling: audit, calibration, tie study, per-seed contract,
  batch runner, replay verifier và RQ2 analyzer trong `experiments/`.
- Không thay alpha, power, margin, tolerance, resource ceiling, target/seed
  list hoặc `torch.set_num_threads(1)`.

## 6. Lệnh thực sự đã chạy

Các lệnh khoa học/chính (đều từ repo `FL-DNA/`):

```text
PYTHONPATH=. .venv-phase1/bin/python experiments/audit_rq1_development_artifacts.py ...
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_harddiff_reparam_for_misselected.py ... --groups 0 --restarts 1
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_dna_level1_forward_attack.py ... --groups 0 --candidates 1 --restarts 1 --iterations 600
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_simple_defense_attack.py ... --defense clipping_noise_mc --groups 0 --restarts 1 --iterations 600 --mc-noise-samples 100
PYTHONPATH=. .venv-phase1/bin/python experiments/calibrate_rq1_dp_distortion.py
PYTHONPATH=. .venv-phase1/bin/python experiments/calibrate_rq1_dp_distortion.py --multipliers 0.0001 0.0002 0.00025 0.0003 0.0005 0.001 0.005 0.01 ...
PYTHONPATH=. .venv-phase1/bin/python experiments/study_rq1_tie_threshold.py
PYTHONPATH=. .venv-phase1/bin/python experiments/rq1_power_analysis.py --output-dir artifacts/rq1/stage1_power_analysis_20260912
PYTHONPATH=. .venv-phase1/bin/python experiments/materialize_rq2_development_contract.py
PYTHONPATH=. .venv-phase1/bin/python -u experiments/run_rq2_stage1_batch.py --workers 8
PYTHONPATH=. .venv-phase1/bin/python experiments/verify_rq2_replay.py
PYTHONPATH=. .venv-phase1/bin/python experiments/analyze_rq2_stage1_batch.py
PYTHONPATH=. .venv-phase1/bin/pytest -q tests
shasum -a 256 -c protocols/pre_pilot_freeze_manifest.sha256
git diff --check
```

Full per-job commands, environment, stdout/stderr và runtime nằm trong từng
`job_record.json`. Tổng runtime cộng theo 72 successful RQ2 job là `9.7652`
machine-hours; median `487.59s/job`, range `[415.45, 536.66]s`, dưới trần RQ2
15 machine-hours. Parallelism tối đa là 8 cho batch, 9 khi có đúng một smoke
job độc lập.

## 7. Gate và bước tiếp theo được phép

- `PRE-PILOT EXECUTION GATE`: **PASSED**.
- `RQ1 STAGE-1 TECHNICAL GATE`: **PASSED**, với tie threshold và comparator
  cần approval confirmatory riêng.
- `RQ2 STAGE-1 TECHNICAL GATE`: **PASSED**; `required_n=21 <= 30`, utility
  comparator chính thức là `NOT_FOUND` theo frozen rule.
- `RQ3 PILOT GATE`: **HELD**, chờ khóa network-emulation tool.
- `CONFIRMATORY EXECUTION GATE`: **NOT PASSED / NOT REQUESTED**.

Trước confirmatory freeze, supervisor cần quyết định/duyệt:

1. tie threshold RQ1 đề xuất `0.0390625` và planning draw `44`;
2. `DP_DISTORTION_MATCHED` config `clip_norm=100`, multiplier `0.00025`;
3. ghi nhận `DP_UTILITY_MATCHED=NOT_FOUND`;
4. confirmatory RQ2 seed count `21` và danh sách seed mới;
5. multiplicity rule giữa hai primary RQ2 methods (endpoint IUT đã khóa, nhưng
   method-family rule chưa đủ tường minh trong YAML);
6. confirmatory configs/hashes và approval table;
7. network-emulation tool trước khi mở RQ3 pilot.

Cho tới khi các mục tương ứng được phê duyệt, chỉ được chuẩn bị/review
confirmatory config; không được tạo target set mới hoặc chạy thí nghiệm chính.
