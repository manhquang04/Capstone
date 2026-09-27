# Báo cáo chuẩn bị confirmatory freeze — 2026-09-13

**Phạm vi:** RQ1/RQ2 freeze sau Stage 1; RQ3 không thay đổi  
**Confirmatory attack/training đã chạy:** không  
**Post-hoc outcome đã đọc:** không; chỉ source IDs được đọc để kiểm tra overlap  
**Trạng thái execution:** `HELD — AWAITING FINAL SUPERVISOR CONFIRMATION`

## 1. Đã làm gì

1. Ghi 5 phê duyệt của Manh Quang vào amendment ngày 2026-09-13 và đồng bộ
   hai pre-pilot YAML, không thay alpha, power, margin hay resource ceiling.
2. Đồng bộ RQ1 distortion grid mở rộng và khóa comparator tại clip norm 100,
   multiplier 0.00025.
3. Vì comparator 0.00025 trước đó mới có smoke một group/một restart, chạy đủ
   8 development group x 8 restart bằng MC attacker đã khóa để đo tie-rate
   đúng comparator. Candidate được chọn chỉ bằng attacker-visible objective.
4. Đo được 0/8 tie ở threshold 0.0390625. Exact-binomial giữ effective
   non-tied `n=37`, rejection `>=24`, alpha thực `0.0494358748`, power thực
   `0.8070956917`; exact Clopper-Pearson 95% CI mô tả cho tie-rate là
   `[0, 0.3694]`. Theo yêu cầu dùng point estimate thật đo được, draw sau 0%
   tie và 5% dropout là `39`, so với planning 44; sensitivity table vẫn giữ
   các kịch bản tie-rate 5%, 10% và 20% để thể hiện bất định của mẫu 8 group.
5. Khóa 21 seed RQ2 mới bằng quy tắc SHA-256 đã ghi trước, sau timestamp của
   interpretation rule; materialize 21 checksum split/partition/model-init.
   Seed overlap với development `[101,...,808]` là 0; có 21 checksum split
   riêng biệt.
6. Tạo đúng một RQ1 target set confirmatory gồm 39 group/156 source row. Kiểm
   tra source IDs với toàn bộ 53 target file cũ cho overlap tối đa 0, gồm
   `level1_fresh_final_targets.pt`. Không đọc outcome post-hoc.
7. Tạo hai config confirmatory và manifest SHA-256. Cả hai giữ
   `confirmatory_execution_authorized: false`.

## 2. File/config thay đổi

- `protocols/amendments/2026-09-13_stage1_supervisor_approvals.md`
- `protocols/config/rq1_pre_pilot.yaml`
- `protocols/config/rq2_pre_pilot.yaml`
- `protocols/config/rq1_confirmatory.yaml`
- `protocols/config/rq2_confirmatory.yaml`
- `protocols/rq1_confirmatory_protocol.md`
- `protocols/rq2_multiseed_protocol.md`
- `protocols/supervisor_decision_package.md`
- `protocols/confirmatory_freeze_manifest_2026-09-13.json`
- `PROJECT.md`
- `experiments/measure_rq1_development_tie_rate.py`
- `experiments/verify_rq1_target_disjointness.py`
- `experiments/create_confirmatory_freeze_manifest.py`

Không sửa `torch.set_num_threads(1)`. Không tạo hoặc sửa RQ3 comparator/tool.

## 3. Lệnh thực sự đã chạy

```text
shasum -a 256 attacks/inversion_metrics.py attacks/pseudo_image.py experiments/rq1_power_analysis.py datasets/creditcard.csv

# 8 tiến trình độc lập, gid=0..7
PYTHONPATH=. .venv-phase1/bin/python experiments/run_phase4_simple_defense_attack.py artifacts/rq1/stage1_selected_dp_tie_rate_20260913/group_${gid}_run --target-file development_gate_targets.pt --defense clipping_noise_mc --groups ${gid} --restarts 8 --iterations 600 --attack-lr 0.1 --init-mode standard --nonnegative-lambda 0.001 --clip-norm 100 --noise-multiplier 0.00025 --mc-noise-samples 100 --defense-seed 314159265

PYTHONPATH=. .venv-phase1/bin/python experiments/measure_rq1_development_tie_rate.py --dna-dir artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/dna_level1_forward_attack_development_gate_c4_r8_i600_lr0p1_nonneg0p001 --dp-root artifacts/rq1/stage1_selected_dp_tie_rate_20260913 --target-file artifacts/phase4/pre_phase4_fraud_gate_20260909T070735493549Z/development_gate_targets.pt --tie-threshold 0.0390625 --output-dir artifacts/rq1/stage1_selected_dp_tie_rate_20260913

PYTHONPATH=. .venv-phase1/bin/python experiments/rq1_power_analysis.py --p0 0.50 --p1 0.70 --alpha 0.05 --power 0.80 --tie-rate 0.0 --dropout-rate 0.05 --maximum-n 150 --output-dir artifacts/rq1/confirmatory_power_analysis_20260913

PYTHONPATH=. .venv-phase1/bin/python experiments/materialize_rq2_development_contract.py --seeds <21 frozen seeds> --max-rows 500000 --batch-size 1024 --num-clients 3 --output artifacts/rq2/confirmatory_freeze_20260913/per_seed_contract.json

PYTHONPATH=. .venv-phase1/bin/python experiments/create_phase4_source_disjoint_targets.py artifacts/rq1/confirmatory_freeze_20260913 --output-name rq1_confirmatory_targets.pt --groups 39 --records-per-group 4 --fraud-per-group 1 --seed 1409291722 --purpose RQ1_CONFIRMATORY_FROZEN_SINGLE_DRAW

PYTHONPATH=. .venv-phase1/bin/python experiments/verify_rq1_target_disjointness.py --confirmatory-target artifacts/rq1/confirmatory_freeze_20260913/rq1_confirmatory_targets.pt --output artifacts/rq1/confirmatory_freeze_20260913/source_overlap_matrix.json

PYTHONPATH=. .venv-phase1/bin/python experiments/create_confirmatory_freeze_manifest.py
```

## 4. Artifact/run IDs và checksum chính

- `stage1_selected_dp_tie_rate_20260913`: tie-rate `0/8`.
- `confirmatory_power_analysis_20260913`: required target draw `39`.
- `rq1/confirmatory_freeze_20260913`:
  target SHA-256 `caee0024c318c2874037382996b843ead8024b02b132185e502e245d3266267f`;
  disjointness `PASS`.
- `rq2/confirmatory_freeze_20260913`: per-seed contract SHA-256
  `2015b6062245661109bb84c0c3c24ba589e1990747c9ff93582902337eed643c`.
- Git reference at freeze: commit `65afddc33d7012e0c0798ad4a210beebce718650`,
  tree `4430eb34e08e1728f04862b26a321dd0729c4e10`.

## 5. Gate

- RQ1 Stage-1 output approval: **PASSED**.
- RQ1 target-count/resource gate: **PASSED** (`39 <= 150`).
- RQ1 source-disjointness gate: **PASSED** (0 overlap across 53 prior target
  files; 156/156 source IDs unique internally).
- RQ2 seed-count/resource gate: **PASSED** (`21 <= 30`).
- RQ2 seed-disjointness/materialization gate: **PASSED**.
- RQ2 analysis-plan gate: **HELD**. The Stage-1 report explicitly listed the
  method-family multiplicity rule between the two primary methods as open; the
  five new approvals do not specify one. No Holm/Bonferroni/separate-claim rule
  was invented.
- RQ3 pilot gate: **HELD**, pending `network_emulation_tool` as instructed.
- Overall confirmatory execution gate: **NOT PASSED**. Neither attack nor
  training may start.

## 6. Amendment/deviation

The development tie-rate run is the requested completion of the frozen sample
size calculation for the approved comparator, not tuning. No privacy outcome
was used to change a comparator, threshold, margin, budget or seed. The only
new amendment records supervisor decisions already supplied in chat.

## 7. Bước tiếp theo được protocol cho phép

Supervisor may review the freeze manifest, RQ1 target checksum/overlap matrix
and RQ2 seed contract. Before final execution authorization, explicitly close
the RQ2 method-family multiplicity item or state that no family-wise claim is
intended and approve the corresponding rule. Until then, only integrity review
and documentation are allowed.
