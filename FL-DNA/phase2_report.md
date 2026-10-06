# Báo cáo nghiệm thu Phase 2

Báo cáo này thay thế kết luận bàn giao tạm thời trước đó. Run đã xác minh: `phase2_verified_v2`.

## Trạng thái

- Implementation: PASS trong phạm vi 13 tests và kiểm tra artifact đã chạy.
- Diagnostic: SUPPORTED_ON_PILOT ở protocol cố định 300 bước: baseline thắng nhất quán cả 3 restart trên 8/10 target khi so với từng control. Đây không phải bằng chứng hội tụ, tái dựng transaction hợp lệ, hay leakage dân số.
- Phase 2: COMPLETE_WITH_LIMITATIONS. Có thể bắt đầu đặc tả/validation Phase 3; chưa có bằng chứng full FedAvg inversion hoặc hiệu quả bảo vệ của DNA.

## Công việc và thuật toán

2A: đọc closure cũ, xuất bảng budget 300 riêng từng target/restart; xác minh 48 cặp initialization bằng torch.equal. Closure có 102 attack calls và 12 prior records; các budget là lời gọi độc lập dùng cùng seed, không phải checkpoint của một trajectory. Hai target development không phải hai nhóm dân số.

2B: lưu RobustScaler center/scale fit trên train và 5 categories của transaction type. MAE từng feature ở không gian scale và đơn vị gốc; transaction type chấm argmax. Strict one-hot yêu cầu mỗi phần tử gần 0/1 và đúng một phần tử gần 1 (atol=1e-6). Kiểm tra balance differences sau inverse transform. Không áp phương trình raw vào các feature có scale khác nhau.

2C: chỉ phát triển ứng viên bỏ L2 từ hai development targets. Giữ Adam lr=0.05, 300 bước, 3 restart. Với hai target này, L2=0 có MSE non-fraud dao động 0.1496--0.7493 giữa restart, trong khi L2=1e-4 gần 0.3185 hơn. Vì vậy L2=0 là lựa chọn development hẹp, không phải tham số đã tối ưu hay được xác nhận ổn định. Chưa triển khai cosine objective, softmax parameterization, TabLeak; không tuning DNA.

Objective: L_match = mean_p(mean((gradient_p(dummy)-observed_p)^2)); L_total=L_match+lambda*mean(dummy^2), lambda=0. Các parameter tensor có trọng số bằng nhau, không phải toàn bộ phần tử bằng nhau. Dummy N(0,1); initialization iteration 0 có tham gia best-so-far. Mỗi bước lưu match/regularization/total. Candidate và restart chọn bằng minimum total objective, không dùng ground truth; không so objective giữa baseline và zero để suy ra leakage.

2D: 50k-row stratified subset, một round warm-up FedAvg 3 client, MLP 128/64/32/1, focal loss và local Adam hiện tại. Gradient quan sát thuộc một mẫu known-label ở eval mode: BatchNorm dùng running buffers, Dropout tắt. Đây là checkpoint ít huấn luyện và gradient từng mẫu, chưa đại diện delta local training. 10 target có 5 fraud/5 non-fraud, không coi trung bình cân bằng lớp là leakage dân số.

## Provenance và tái lập

Source row IDs (zero-based data row, excluding header): `[2199318, 6362575, 508122, 4828616, 5989567, 2777959, 1030689, 6030597, 3838926, 3455665]`. Development IDs: `[5784126, 6040745]`. Đã assert không giao nhau. Mapping replay đúng stratified sampling/split bằng test dataframe và assert labels; CSV SHA256: `16910f90577b0d981bf8ff289714510bb89bc71bff7d3f220f024e287e4eea6b`.

Checkpoint đã lưu, reload trước capture; checksum trong JSON. Seed dữ liệu 20260907, attack root 731921. Scaler, validation indices, commit/worktree status trong JSON. Protocol lock ghi trước attack. Cấu hình giữ nguyên sau pilot trước; đây là lượt kiểm tra kỹ thuật, không được gọi là đánh giá mù mới độc lập.

## Kết quả từng target

Baseline/zero chọn restart riêng theo objective; prior trung bình 3 initialization. Bảng này mô tả reconstruction được chọn riêng từng method, nên không dùng để đếm thắng/thua. Không gọi các restart là mẫu độc lập.

| Target | Label | Baseline MSE | Zero MSE | Prior MSE | Baseline MAE | Type đúng | Best step |
|---|---|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.166920 | 14.720832 | 1.311376 | 0.335491 | 0 | 266 |
| 1 | 1 | 2187.787598 | 2244.239746 | 2272.315674 | 20.545664 | 1 | 300 |
| 2 | 0 | 0.135419 | 42.087044 | 1.514524 | 0.311745 | 1 | 294 |
| 3 | 0 | 10.568270 | 36.978672 | 12.494375 | 2.456160 | 0 | 232 |
| 4 | 1 | 1.215958 | 45.546471 | 4.371891 | 0.869308 | 0 | 299 |
| 5 | 0 | 0.304003 | 28.830824 | 1.657821 | 0.445265 | 1 | 300 |
| 6 | 1 | 16648.345703 | 16648.347656 | 17057.341146 | 46.542568 | 0 | 300 |
| 7 | 1 | 2.307348 | 56.863316 | 6.341699 | 1.291564 | 0 | 286 |
| 8 | 1 | 0.000606 | 60.079414 | 1.326761 | 0.019946 | 1 | 300 |
| 9 | 0 | 0.155514 | 32.664684 | 1.255551 | 0.290899 | 1 | 300 |

## So sánh theo lớp

Mean bị chi phối bởi target fraud 6. Bảng thêm median và range để cho thấy phân bố của 5 target mỗi lớp; không xem đây là ước lượng dân số.

| Label | Method | Mean MSE | Median MSE | MSE range | Mean MAE | Argmax type accuracy |
|---|---|---:|---:|---:|---:|---:|
| 0 | baseline | 2.266025 | 0.166920 | 0.135419--10.568270 | 0.767912 | 0.60 |
| 0 | zero_gradient | 31.056411 | 32.664684 | 14.720832--42.087044 | 4.737993 | 0.20 |
| 0 | prior | 3.646729 | 1.514524 | 1.255551--12.494375 | 1.218050 | 0.00 |
| 1 | baseline | 3767.931442 | 2.307348 | 0.000606--16648.345703 | 13.853810 | 0.40 |
| 1 | zero_gradient | 3811.015321 | 60.079414 | 45.546471--16648.347656 | 17.339023 | 0.20 |
| 1 | prior | 3868.339434 | 6.341699 | 1.326761--17057.341146 | 12.538536 | 0.20 |

## Nhận xét và cổng quyết định

Kết quả chính dùng target làm đơn vị: baseline thắng cả 3 restart trước zero-gradient trên 8/10 target và trước prior trên 8/10 target. Hai target còn lại là fraud target 1 và 6; baseline thắng zero 1/3 restart và prior 2/3 restart ở mỗi target đó. Con số 26/30 (zero) và 28/30 (prior) được giữ như chẩn đoán độ ổn định theo restart, không phải 26 hoặc 28 mẫu độc lập. Không báo p-value hoặc CI dân số.

Zero-gradient dùng cùng initialization, learning rate, budget và số restart với baseline; vì vậy phép so sánh tách tác động của observed gradient trong đúng thuật toán này. Nó không phải benchmark cho no-update attacker mạnh nhất. Prior hiện là đối chứng không tối ưu hóa; chưa có prior theo phân phối dữ liệu hoặc baseline no-update được tối ưu riêng. Bảng selected và bảng paired cùng restart là hai estimand khác nhau.

Năm trong mười baseline reconstruction đạt best ở bước 300. Do đó kết quả chỉ mô tả attack với ngân sách tối đa 300 bước; loss curve hiện có không chứng minh hội tụ hoặc mức độ attack mạnh nhất. Budget không được tăng sau khi xem evaluation. Lợi ích tương đối so với control cũng không đồng nghĩa tái dựng transaction thành công.

Fraud MSE có outlier lớn, đặc biệt target 6 (baseline 16648.35); target 1 cũng có MSE cao. Mean theo lớp vì thế dễ gây hiểu nhầm và phải đọc cùng median, range và bảng từng target. Không suy nguyên nhân của các outlier từ run này.

Type accuracy trong bảng là argmax trên vector continuous gồm 5 category, chỉ là tín hiệu exploratory. Strict one-hot validity bằng 0 cho baseline, zero-gradient và prior; vì vậy 40--60% argmax accuracy không phải bằng chứng tái dựng categorical transaction hợp lệ. Balance consistency là phép đo, chưa phải constraint trong optimizer. Không có threshold tái dựng thành công đã hiệu chỉnh (TBD).

## Kiểm tra và compute

13 tests pass. 120 vector reload checks kiểm tra shape/dtype/giá trị; 60 objective reload checks dùng checkpoint và observed signal, so match/reg/total với candidate. 30 pairing assertions trong evaluation. 60 optimization trajectories, 30 priors; 0 failed jobs.

Runtime: {"attack_seconds_including_baseline_and_zero_restarts": 67.66009470999998, "total_seconds_including_data_and_warmup": 78.045432333}. Timer attack bao gồm baseline+zero cùng restarts; total gồm setup/training/lưu file. Không đo peak RAM/VRAM. Không ngoại suy sang Adam multi-step inversion.

## Artifacts và lệnh

Run directory: `artifacts/phase2/phase2_verified_v2`. `heldout_diagnostic_results.json`, `heldout_trials.csv`, `paired_results.csv`, `feature_results.csv`, `convergence.csv`, `development_results.csv`, checkpoint, targets, observed gradients và vectors. Artifact cũ giữ nguyên.

```bash
DATALOADER_NUM_WORKERS=0 .venv-phase1/bin/python -m experiments.run_phase2_diagnostic
.venv-phase1/bin/python -m experiments.summarize_phase2 artifacts/phase2/phase2_verified_v2
.venv-phase1/bin/python -m pytest -q -p no:cacheprovider tests/test_phase1_invariants.py tests/test_phase2_metrics.py
```

## Bàn giao tiếp theo

- A: đặc tả pre-local checkpoint, Adam state, batch order, BN/Dropout của FedAvg thật.
- B: xác minh forward local update trước inversion; dùng Phase 2 làm diagnostic reference. Nếu cần nâng attack tabular, thử từng parameterization/objective trên development riêng, thêm prior không dùng update được tối ưu công bằng, và kiểm tra hội tụ với budget khóa trước evaluation.
- C: phân tích khả nghịch DNA và random realization trước adaptive attack; chưa xếp hạng defense.
- D: khóa protocol Phase 3, metric/compute budget; giữ đơn vị sample/restart/client rõ ràng.

Không triển khai Phase 3, không push/publish. Phase 2 hoàn thành với giới hạn đã mô tả: diagnostic sample-gradient đã được kiểm tra, còn full FedAvg update attack, adaptive DNA attacker, categorical-valid reconstruction và kết luận bảo vệ DNA đều là việc Phase 3.
