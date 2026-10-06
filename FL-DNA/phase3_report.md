# Báo cáo Phase 3: full-client Adam update inversion

## Kết luận nghiệm thu

Đã hoàn tất triển khai và chạy đánh giá trên toàn bộ local set của cả ba client
trong subset PaySim 500.000 dòng. Tín hiệu là model delta sau một local epoch Adam
thật. Replay, candidate và artifacts đều đạt các kiểm tra đã chạy.

**Trạng thái cuối: COMPLETE_WITH_NEGATIVE_RESULT. Cổng Phase 4: CHƯA ĐẠT.**
Kết quả full-client ban đầu chưa rõ, nên đã chạy thêm follow-up, bounded controls,
đạo hàm BatchNorm và Adam difficulty ladder. Confirmation cuối chỉ thắng prior
và zero-update ở 7/10 nhóm; mean MSE đều xấu hơn control. Full-client development
sau sửa cũng xấu hơn prior. Chưa có bằng chứng reconstruction đủ mạnh để xếp
hạng khả năng bảo vệ DNA.

Không tăng ngân sách sau khi xem lỗi reconstruction để tìm kết quả tích cực.
Không đặt ngưỡng thành công tùy ý. Đây là development evaluation của một
partition/checkpoint, chưa phải kết luận trên nhiều lần huấn luyện độc lập.

## 1. Đối chiếu PROJECT.md

| Hạng mục | Đã thực hiện | Kết quả |
|---|---|---|
| 3A: threat model | Checkpoint, labels, order, RNG, optimizer, BN, preprocessing | Đã đặc tả |
| 3B: update thật | Gọi train_local_model; lưu delta, capture và evaluator riêng | Đạt runtime |
| Replay trước inversion | Toàn bộ parameters và buffers, ba client | Max error = 0 |
| 3C mức 1/2 | Pilot Adam 1/2 bước; tests 1/2/5 bước và đạo hàm | Đạt cases đã chạy |
| 3C mức 3 | Unroll đủ 130/117/73 bước, batch 1024, một epoch | Đã chạy full local set |
| Controls | Baseline, zero-update, prior cùng initialization, 3 restarts | 18 trajectories, 9 priors |
| Reconstruction | MSE, median, MAE, lỗi feature scaled/raw, categorical theo lớp | Đã lưu |
| Candidate/provenance | Objective-only selection, reload, pairing, checksum | Đạt |
| Compute/convergence | Runtime, peak RSS, loss history và biểu đồ | Đã đo |
| Hiệu lực để so DNA | So với cả hai controls | Kết quả âm tính; gate không đạt |

“Full client” nghĩa là mọi bản ghi trong client của subset 500k. Không khẳng định
đã dùng toàn bộ 6,36 triệu dòng CSV, nhiều round quan sát, checkpoint 50-round,
unknown-label attack hoặc attacker thông thường không biết RNG.

## 2. Dữ liệu và cấu hình

- 500.000 dòng: train 325.000, validation 74.999, test 100.001.
- Giữ PaySim preprocessing, RobustScaler fit train, 13 features, MLP 128/64/32/1,
  BatchNorm và Dropout gốc. Focal loss alpha=0.95, gamma=2.
- Warm-up một round FedAvg bằng đúng split/scaler 500k rồi capture local epoch
  tiếp theo. Không dùng checkpoint Phase 2 với scaler khác cho kết quả chính.
- Native local Adam: lr=0.001, betas=(0.9,0.999), epsilon=1e-8, fresh moments,
  không weight decay. Optimizer mới mỗi client, đúng hàm training hiện tại.
- CPU, torch intra-op 1 thread/process, DataLoader zero-worker. Ba client chạy
  ở ba process. MPS chưa được kiểm chứng.
- Attack Adam lr=0.05, 100 iterations, 3 restarts, không regularization.
- Data seed 20260907; run seed 1157354082. Initialization seed derive theo
  client/restart; CPU RNG trước local training nằm trong capture.

| Client | Bản ghi | Fraud | Non-fraud | Local steps |
|---|---:|---:|---:|---:|
| 0 | 132.175 | 140 | 132.035 | 130 |
| 1 | 118.963 | 140 | 118.823 | 117 |
| 2 | 73.862 | 139 | 73.723 | 73 |

Các bản ghi được tái dựng đồng thời: 325.000 record errors không phải
325.000 attack experiments độc lập. Ba restart không phải ba dataset mới.

## 3. Thuật toán và threat model

Attacker được cấp pre-local model, labels theo thứ tự, batch sizes/order, scaler,
Adam config và RNG Dropout. Labels, order và RNG là giả định bổ sung mạnh.
Model luôn train mode: BatchNorm dùng thống kê batch, Dropout hoạt động.

1. Native local training tạo model cuối; lấy model cuối trừ model ban đầu.
2. Recorder lưu toàn bộ batches và row IDs trong client; assert mỗi record xuất
   hiện đúng một lần. Đây là vị trí dataset của client, không phải CSV row ID.
3. Simulator dùng functional_call, buffers riêng và Adam moments có graph đạo hàm.
   fork_rng phục hồi cùng dropout realization trong mỗi replay.
4. Replay dữ liệu thật phải khớp native training trước khi tối ưu dummy data.
5. Tạo một ma trận dummy cho toàn bộ client từ N(0,1). Mỗi attack iteration
   chạy lại toàn bộ local epoch với dummy data để tạo delta giả.
6. Objective là trung bình MSE từng parameter tensor so với delta quan sát.
   Mỗi tensor có trọng số bằng nhau; không phải MSE flatten toàn model.
7. Đạo hàm đi qua các bước Adam về dummy data. Labels/order cố định. Buffers
   được capture/replay nhưng chưa dùng trong objective.
8. Candidate có objective thấp nhất được giữ, kể cả iteration 0. Ground truth
   không chọn candidate, restart hay stopping.
9. Đọc candidate từ file và tính lại objective để đối chiếu best đã lưu.

Adam replay giữ đúng thứ tự float32 PyTorch: lerp cho moment thứ nhất,
addcmul cho moment thứ hai, sqrt trước chia bias correction, rồi addcdiv.
Chuyển bias correction vào trong sqrt từng gây sai lệch ở pilot trước.
Ở variance đúng bằng 0, simulator quy ước derivative sqrt bằng 0, giữ forward
sqrt(0)=0; đây không phải bảo đảm tối ưu hóa toàn cục.

Ground truth chỉ dùng cho forward validation và chấm kết quả. Chấm theo known
ordering; chưa có unordered-set reconstruction/Hungarian matching. Zero-update
là control đổi observed signal thành 0, không phải no-update attacker mạnh nhất.
Prior là initialization chưa tối ưu. Chi tiết: fedavg_threat_model.md và
update_capture_protocol.md.

## 4. Kết quả reconstruction

MSE càng thấp càng tốt. Mean ± std dưới đây tính trên 3 restarts/client,
population std ddof=0, chỉ mô tả độ dao động initialization.

| Client | Baseline MSE | Zero-update MSE | Prior mean MSE |
|---|---:|---:|---:|
| 0 | 149.121392 ± 0.004474 | 149.150487 ± 0.006113 | 149.128892 |
| 1 | 81.486367 ± 0.005070 | 81.501261 ± 0.005674 | 81.478432 |
| 2 | 117.470869 ± 0.023872 | 117.487986 ± 0.025679 | 117.475558 |

Baseline thắng zero 3/3 restarts ở cả ba client. Với prior: thắng 3/3 ở client 0,
0/3 ở client 1, 2/3 ở client 2. Chỉ **1/3 client** thắng prior nhất quán.
5/9 là chẩn đoán restart, không phải 5/9 mục tiêu độc lập. Không báo p-value
hoặc suy ra leakage dân số.

Bảng sau chọn restart baseline bằng objective, sau đó chấm từng lớp:

| Client | Lớp | Mean MSE | Median MSE | Type argmax accuracy |
|---|---|---:|---:|---:|
| 0 | Non-fraud | 142.0845 | 1.9028 | 20.05% |
| 0 | Fraud | 6788.3270 | 114.1594 | 17.14% |
| 1 | Non-fraud | 75.3176 | 2.0303 | 20.02% |
| 1 | Fraud | 5312.5409 | 206.4178 | 23.57% |
| 2 | Non-fraud | 106.2275 | 1.9638 | 19.63% |
| 2 | Fraud | 6084.9908 | 148.3489 | 23.02% |

Mean và median khác xa nhau; không chỉ đọc mean. Strict one-hot validity bằng
0 cho các reconstruction được chọn. Argmax trên 5 categories không chứng minh
khôi phục transaction hợp lệ. MAE từng feature scaled/raw nằm trong report.json.

6/9 baseline trajectories đạt best ở bước 100. Objective giảm nhưng chưa chứng
minh hội tụ. Objective thấp không đồng nghĩa record đúng; không so objective
baseline/zero như một thước đo privacy.

## 5. DNA capture và compute

Đọc lại captures, chạy native local training và assert raw delta chính xác.
DNAEncoder thật encode/decode mọi floating delta; dna_transform_state thật dùng
current config: mix=0.05, keep=0.90, shrink=0.50, block size=256.
Đây là kiểm tra capture/representation, chưa phải adaptive DNA attack.

| Client | Lossless bit-exact | Encode/decode | Transform | Transform seed |
|---|---|---:|---:|---:|
| 0 | PASS | 92.37 ms | 56.36 ms | 797480653 |
| 1 | PASS | 91.75 ms | 55.39 ms | 1431978332 |
| 2 | PASS | 92.45 ms | 55.93 ms | 2037897412 |

Defense run seed 1305222346. Transform replay bằng logged seed khớp state.
Các thời gian này là một lần đo/capture, không phải network latency.

| Client | Preprocessing | Warm-up | Local epoch | Tổng run gồm restarts | Peak process RSS |
|---|---:|---:|---:|---:|---:|
| 0 | 4.56 s | 2.82 s | 1.32 s | 990.92 s | 3.17 GB |
| 1 | 4.56 s | 2.77 s | 1.11 s | 890.54 s | 2.55 GB |
| 2 | 4.86 s | 2.85 s | 0.78 s | 620.99 s | 2.36 GB |

Ba process chạy đồng thời; không cộng ba dòng để gọi là wall time.
Lượt dài nhất khoảng 16,5 phút. RSS gồm preprocessing, không phải VRAM.
Chưa đo GPU, network traffic hay production SecureAgg.

## 6. Kiểm tra và lỗi development

- 17/17 regression tests Phase 1–3 pass. Bốn cases mới kiểm tra Adam replay,
  RNG/model isolation và finite-difference derivative trên mô hình trơn.
- 3/3 full-client replays max absolute error = 0, gồm parameters và buffers.
- 18/18 candidate reload checks; 9/9 initialization pairing checks.
- Đủ 18 histories, mỗi history 101 điểm; minimum khớp best metadata.
- Ba pre-local checkpoints và scaler files có checksum giống nhau.
- Summary kiểm tra checksum, thiếu/duplicate trials và protocol mismatch.
- Các run chính có 0 failed jobs.

Development có một lượt dừng vì checkpoint/scaler khác subset, và một lỗi
recorder do persistent workers giữ dataset hai trường. Đã sửa warm-up theo đúng
scaler và dùng zero-worker recorder. Artifacts cũ giữ riêng, không được đưa vào
summary chính. Không nới tolerance để bỏ qua replay fail.

## 7. Files và lệnh tái lập

Ba run chính dưới artifacts/phase3:
- full_20260908T143837588533Z: client 0.
- full_20260908T143843502722Z: client 1.
- full_20260908T143850567575Z: client 2.

Tổng hợp: artifacts/phase3/full_summary_20260908/ gồm report.json,
metrics_summary.csv, restart_summary.csv, convergence.png, defense captures,
environment.json, source_snapshot và manifest.json. Mỗi run có protocol lock,
preprocessing, pre-local checkpoint, compute profile, capture, evaluator data;
mỗi trajectory có initialization, best reconstruction và loss history.

~~~bash
.venv-phase1/bin/python -m pytest -q -p no:cacheprovider tests/test_phase1_invariants.py tests/test_phase2_metrics.py tests/test_phase3_updates.py
.venv-phase1/bin/python -m experiments.run_phase3_full_client --max-rows 500000 --iterations 100 --restarts 3 --clients 0 --seed 1157354082
.venv-phase1/bin/python -m experiments.run_phase3_full_client --max-rows 500000 --iterations 100 --restarts 3 --clients 1 --seed 1157354082
.venv-phase1/bin/python -m experiments.run_phase3_full_client --max-rows 500000 --iterations 100 --restarts 3 --clients 2 --seed 1157354082
.venv-phase1/bin/python -m experiments.summarize_phase3_full artifacts/phase3/full_20260908T143837588533Z artifacts/phase3/full_20260908T143843502722Z artifacts/phase3/full_20260908T143850567575Z --output artifacts/phase3/reproduced_summary
~~~

Lệnh summary đọc run đã lưu. Khi training lại, dùng thư mục mới runner in ra.
Summary tạo seed DNA capture mới và ghi lại. Để replay capture cũ, dùng
config/seed trong defense_capture_client_*.pt.

## 8. Quyết định tiếp theo

Thực thi Phase 3 đã chuyển sang full-client model delta. Hiệu lực reconstruction
chưa thuyết phục so với prior, categorical chưa hợp lệ, budget chưa chứng minh
hội tụ. **Chưa dùng attack này để kết luận DNA bảo vệ tốt.**

- A: giữ capture/split làm reference; chưa mở nhiều round/dataset.
- B: phát triển attack theo protocol development mới, kiểm tra parameterization
  tabular và thông tin buffer chưa khai thác; không mặc định tăng iterations là đủ.
- C: phân tích toán học DNA độc lập; chưa xếp hạng privacy từ kết quả này.
- D: khóa tiêu chí reconstruction/budget trước evaluation mới. Ngưỡng thành công
  chưa có căn cứ vẫn để TBD.

Không push GitHub, không thay utility artifacts 50-round cũ.

## 9. Đóng Phase 3: Adam difficulty ladder

Phase 3 được đóng ở trạng thái **COMPLETE_WITH_NEGATIVE_RESULT**. Tất cả sáu
đầu việc cuối đã thực hiện; effectiveness gate không đạt, nên `phase4_ready` là
false. Kết quả máy đọc được nằm tại
`artifacts/phase3_closure/adam_ladder_v2/phase3_closure.json`.

### 9.1. Mục tiêu tái dựng đã chốt

Mục tiêu chính là full-client update của cấu hình utility: local Adam, một epoch,
batch 1024, train-mode BatchNorm/Dropout. Hai mức nhỏ hơn chỉ dùng xác định điểm
attack bắt đầu mất hiệu lực:

- one_batch: 4 records, 1 fraud, một Adam step;
- four_batches: 16 records, 4 fraud, bốn Adam steps;
- full_client: 73862 records, 73 Adam steps trong development client 2.

Attacker biết labels, batch order, local RNG, optimizer và checkpoint. Quan sát
là individual floating model-state delta, gồm parameters và BatchNorm buffers.
Đây là giả định thuận lợi cho attacker và không đại diện Secure Aggregation.

### 9.2. Adam ladder trên development

Mỗi mức nhỏ dùng 4 nhóm development. Attack chạy 300 iterations và thử ba
learning rates. MSE delta âm nghĩa là baseline tốt hơn prior.

| Mức | lr=0.01 | lr=0.05 | lr=0.10 |
|---|---:|---:|---:|
| 1 batch, 4 records | -1.037220 | -1.471977 | -1.618191 |
| 4 batches, 16 records | -24.754767 | -10.316176 | -42.257900 |

Development cải thiện ở cả hai mức. Theo rule khóa trước, one_batch chọn lr=0.10.
Kết quả này cho thấy differentiable Adam attack có thể khai thác update trong
nhóm development nhỏ; chưa đủ chứng minh ổn định.

### 9.3. Confirmation đã khóa

Confirmation one_batch dùng 10 nhóm source-disjoint, 3 restarts, 300 iterations.
Baseline/zero tự chọn candidate bằng objective của attacker. Prior là trung bình
ba initializations. Hungarian matching chỉ dùng khi chấm reconstruction trong
từng label. Sign test loại ties và dùng group làm đơn vị.

| Control | Baseline thắng | Mean delta MSE | Median delta MSE | p một phía |
|---|---:|---:|---:|---:|
| Prior | 7/10 | +74.356646 | -10.676704 | 0.171875 |
| Zero-update | 7/10 | +49.654386 | -79.534573 | 0.171875 |

Median âm nhưng mean dương do các nhóm lỗi lớn. Gate yêu cầu đồng thời mean âm,
median âm và p<0.05 với cả hai controls; không điều kiện nào được nới sau khi xem
kết quả. Confirmation không đạt.

### 9.4. Full-client result

Full-client Adam development sau sửa differentiable BatchNorm vẫn âm tính:

| Method | Reconstruction MSE |
|---|---:|
| Prior | 117.099641 |
| Baseline observed update | 129.473152 |
| Zero-update | 129.398158 |

Baseline không vượt prior hoặc zero. Không chạy full-client confirmation vì
development không cho tín hiệu cải thiện.

### 9.5. Sáu đầu việc đã hoàn thành

1. Chốt rõ target full-client và vai trò diagnostic của các mức nhỏ.
2. Chạy Adam theo ladder một batch, nhiều batch và full client.
3. Sửa objective/buffer trên development; BatchNorm buffer có đạo hàm thật.
4. Khóa lr, iterations, restarts, controls, metrics và gate trước confirmation.
5. Chạy 10 nhóm confirmation mới và báo paired effect/sign test theo group.
6. Lưu protocol, seeds, source IDs, original/initial/reconstruction, histories,
   per-record CSV, runtime và closure JSON.

Runner v2 tạo 331 files, 8.1 MB, hoàn tất trong 203.11 giây. Run v1 dừng do
thiếu argument `batch_size` và không được dùng; v2 sửa lỗi và chạy trọn vẹn.
29 regression tests pass; `git diff --check` pass. Native Adam replay được kiểm
tra cho từng target trước attack; saved candidate được load và objective tính lại.

### 9.6. Quyết định

**Phase 3 execution: COMPLETE. Phase 3 scientific result: NEGATIVE. Phase 4:
NOT READY.** Baseline attack chưa tái dựng tốt hơn prior/control ổn định trên
confirmation, và full-client development xấu hơn controls. Vì vậy attack hiện
chưa đủ cơ sở để xếp hạng khả năng bảo vệ DNA. Đây là kết luận cuối của protocol
hiện tại, không phải tác vụ đang dang dở.

## 10. Full-client Adam: kiểm chứng bản sửa buffer

Đã mở rộng đạo hàm BatchNorm sang simulator Adam và kiểm tra hồi quy:
29 tests pass, gồm native Adam nhiều bước và finite difference của buffer.
Protocol: `phase3_adam_buffers_protocol.md`. Run:
`artifacts/phase3_followup/adam_buffers_v6/adam_buffers/`.

Dùng đúng client development cũ 73862 records, local Adam một epoch,
checkpoint/batches/RNG như reference. 100 attack iterations, một initialization
ghép cặp. Objective khớp tất cả floating state deltas bằng equal-tensor MSE.
Categorical softmax giữ như reference. Candidate chọn theo objective, kiểm tra
loss sau load; native replay đạt trước chạy. Cả hai candidate tốt nhất ở bước 100.

| Method | Reconstruction MSE |
|---|---:|
| Prior | 117.099641 |
| Baseline Adam + differentiable buffers | 129.473152 |
| Zero-update | 129.398158 |

Baseline xấu hơn prior 12.373511 và zero 0.074994. Development âm tính: sửa đạo
hàm không tự giải quyết reconstruction full-client. Không suy ra nguyên nhân
duy nhất từ kết quả này. Hai trajectories hoàn tất trong 213.87 giây (elapsed
cặp gồm evaluation/artifact); 0 failed jobs. Không chạy confirmation mới từ
cấu hình không cải thiện development. Artifacts trước được giữ nguyên.

Full Phase 3 vẫn NOT READY; chưa có bằng chứng đủ để đánh giá bảo vệ DNA.
Không thể biến yêu cầu hoàn thành thành cam kết đạt kết quả dương. Mọi vòng
phát triển tiếp phải có giả thuyết và budget xác định, không tự tăng số lượt
confirmation tới khi p-value đạt ngưỡng.

## 11. Confirmation fair_controls_v5

Đã hoàn tất development và confirmation sau sửa đối chứng. Run riêng:
`artifacts/phase3_bounded/fair_controls_v5/`. Prepare loại source IDs của tất cả
bounded runs có target artifacts và hai subset 500k cũ trước khi lấy nhóm mới.
4 nhóm development, 10 nhóm confirmation; mỗi nhóm 4 giao dịch, gồm 1 fraud.
Learning rate 0.01 được khóa trước confirmation; 300 bước, 3 restarts.

Prior được lấy trung bình ba initialization, không chọn bằng objective chứa
update. Zero-update dùng scale hằng 1 và chọn restart bằng objective của chính
nó. Baseline chọn objective thấp nhất. Ground truth chỉ dùng metric/matching.
Sign test loại exact ties; đơn vị là nhóm client, không phải restart hay record.

| So sánh MSE baseline trừ control | Thắng / nhóm | Mean | Median | p một phía |
|---|---:|---:|---:|---:|
| Prior | 7/10 | +0.278993 | -1.341396 | 0.171875 |
| Zero-update | 8/10 | -6.630079 | -7.141737 | 0.054688 |

Không ties. Cả hai gate đều false. Có cải thiện ở đa số nhóm, nhưng mean so
prior vẫn dương và chưa đạt quy tắc khóa trước. Không tăng n sau khi nhìn p
để tìm significance. Do nhiều vòng development, p-values là mô tả exploratory.

12 tests liên quan pass, gồm native SGD/Adam replay, buffer finite differences,
Hungarian và zero-control scale independence. Development có 24 trajectories;
confirmation có 60 trajectories, mỗi candidate được lưu/load và kiểm tra loss.
Các lệnh prepare/development/confirm/summarize của phase3_bounded_validation
đã chạy xong; kết quả JSON/CSV nằm trong run trên. git diff --check pass.

Giới hạn: capture bounded hiện dùng simulator đã unit-test against native;
chưa native replay riêng từng target. Đây là one-step SGD, class enrichment
25% fraud, train-mode BatchNorm, known labels/order/RNG và checkpoint cố định.
Không phải full-client Adam của utility experiments. Không so DNA trong lượt
này vì baseline còn chưa đạt yêu cầu. Full Phase 3 vẫn NOT READY.

## 12. Bounded development và sửa đạo hàm BatchNorm

Các run bounded tiếp theo chưa đóng cổng full-client Adam. SGD một bước với
4 records (1 fraud, 3 non-fraud) là diagnostic theo Level 1 của PROJECT.md.
Không tương đương batch 1024 hoặc client hàng chục nghìn records.

Confirmation trước sửa đạo hàm: SGD v1 thắng prior 2/10 nhóm; cosine v2 thắng
3/10; Hungarian v3 thắng 5/10. Các gate đều false. Không tiếp tục thay test set
chỉ để tìm một lượt đạt significance.

Runtime inspection phát hiện running_mean/running_var không có đạo hàm theo
dummy input trong native functional BatchNorm. Loss trước đó có chứa các buffer
này, nhưng đạo hàm tối ưu bỏ mất thành phần đó. Không thể quy mọi thất bại cũ
cho scale objective hoặc hoán vị. Đặc biệt known dropout RNG gắn với vị trí hàng,
nên matching là metric set-level bổ sung, không chứng minh thứ tự vô nghĩa.

simulate_sgd được bổ sung thống kê differentiable bằng pre-forward hooks:
running mean cập nhật từ batch mean; running variance từ unbiased batch variance,
dùng đúng momentum. Native forward/parameter update vẫn giữ nguyên. Hooks được
gỡ sau mỗi call. Test xác nhận native replay và đạo hàm buffer bằng central
finite difference trên float64. Tổng 11 tests liên quan pass.

Run `artifacts/phase3_bounded/bn_gradient_v4/` đã hoàn thành development:

| Attack learning rate | Mean MSE baseline trừ prior |
|---|---:|
| 0.01 | -31.594781 |
| 0.05 | -86.000240 |
| 0.10 | -141.356261 |

Số âm là cải thiện. Đây là 4 nhóm development, chưa chạy confirmation v4.
Không so trực tiếp độ lớn này với v3 vì targets khác nhau. Cần kiểm tra exclusion
với tất cả bounded runs trước khi mở confirmation; prepare hiện chỉ loại hai
subset 500k cũ và nhóm development của chính run đó.

Các vấn đề protocol phải giải quyết trước gate: zero-update đang dùng norm của
update thật làm scale, nên chưa là đối chứng không biết update; summary chọn
restart baseline bằng objective rồi lấy prior/control cùng restart, chưa so với
prior độc lập hay control tự chọn restart. Sign test hiện đếm ties trong n,
không phải exact sign test loại ties. Multiple development/confirmation cycles
đòi hỏi một protocol mới khóa trước, không coi p-value từng vòng là xác nhận cuối.

Trạng thái: BUG FIX VERIFIED; DEVELOPMENT PROMISING; FULL PHASE 3 NOT READY.
Không có kết luận khả năng bảo vệ DNA từ các bounded results này.

## 13. Năm bước bổ sung: nghiệm thu trên dữ liệu mới

Kết quả cập nhật thay cho quyết định tiếp theo ở mục 8: năm bước đã thực thi,
nhưng hiệu lực attack vẫn INCONCLUSIVE. Không mở Phase 4 để xếp hạng DNA.
Artifacts: `artifacts/phase3_followup/final_checks_v1/`.

### 9.1. Phân tích lỗi theo feature

`feature_analysis.json` tách fraud/non-fraud, client, restart và method.
Trên các baseline cũ, trung bình MSE feature qua client/restart của non-fraud
lớn nhất ở oldbalanceOrg (763.78), newbalanceOrig (438.47), balance_diff_orig
(116.06). Với fraud, balance_diff_orig là 76851.86, oldbalanceOrg 1314.69,
balance_diff_dest 228.14. Đây là lỗi trên thang chuẩn hóa, không phải tiền tệ.
Các feature balance chi phối lỗi; phân tích này không xác định nguyên nhân nhân quả.

### 9.2. Budget trên cùng trajectory

Dùng client development 2 cũ, cùng khởi tạo restart 0 và cùng checkpoint.
Chạy một trajectory 500 bước mỗi phương pháp, lưu best objective tại ba mốc;
không đổi initialization theo budget. Ground truth chỉ dùng chấm metric.

| Bước | Baseline MSE | Zero-update MSE | Prior MSE |
|---|---:|---:|---:|
| 100 | 117.478852 | 117.504945 | 117.489766 |
| 300 | 117.466498 | 117.504659 | 117.489766 |
| 500 | 117.461974 | 117.508792 | 117.489766 |

Baseline objective giảm từ 5.7467e-5 ở mốc 100 xuống 2.4062e-5 ở mốc 500,
nhưng mức cải thiện reconstruction so prior vẫn nhỏ. Không xem objective giảm
là bằng chứng tái dựng thành công. Mốc 100 tái lập kết quả raw cũ.

### 9.3. Thay một thành phần của attack

Mỗi ứng viên chạy 100 bước, một khởi tạo, baseline và zero-update ghép cặp.
Model, native Adam/focal loss và local-training mode không đổi.

| Attack | MSE baseline | MSE prior riêng | Baseline trừ prior |
|---|---:|---:|---:|
| Raw reference | 117.478852 | 117.489766 | -0.010914 |
| Objective chia learning-rate bình phương | 118.149699 | 117.489766 | +0.659933 |
| Softmax nhóm type | 117.088589 | 117.099641 | -0.011052 |
| Ràng buộc balance-derived features | 154.907934 | 149.655451 | +5.252483 |

Scaled chia objective cho local learning rate bình phương, áp dụng giống nhau
cho baseline/control. Categorical giữ numeric và dùng softmax cho năm cột type.
Balanced tính hai hiệu số balance trong đơn vị gốc rồi chuẩn hóa lại bằng scaler
cố định. Hai biến thể sau có prior được decode theo chính parameterization đó.
Không gán mức giảm tuyệt đối do đổi prior cho thông tin trong update.

Chọn categorical theo baseline-minus-own-prior thấp nhất ở development.
Chênh lệch với raw chỉ khoảng 0.000138, không chứng minh ưu thế ổn định.
Balanced được bổ sung sau khi hai ứng viên đầu đã có kết quả, trước freeze và
confirmation; ghi rõ trong `development_amendment.json`, không gọi toàn bộ
sweep là preregistered. `frozen.json` khóa categorical/100 bước/2 restarts trước
khi tạo tập mới. Softmax là relaxation liên tục, không phải one-hot hợp lệ.

### 9.4. Kiểm chứng source-disjoint

Loại toàn bộ source rows của subset 500k cũ, lấy một subset 500k khác với seed
472287461; train 325000 hàng, ba client 132118/119033/73849 hàng. Giữ scaler
và checkpoint cũ, không refit trên confirmation. Source IDs và provenance được
lưu. Đây không phải ba lần huấn luyện FL độc lập.

Mỗi client chạy baseline/zero-update với hai initialization ghép cặp, 100 bước.
Prior là initialization chưa tối ưu, được decode theo cùng categorical mapping.
Attack biết label, thứ tự batch và RNG local training; đây là giả định mạnh.
Tín hiệu là full-client local model delta, không phải một sample gradient.

| Client | Baseline MSE mean ± std | Prior mean | Zero-update mean ± std |
|---|---:|---:|---:|
| 0 | 147.611927 ± 0.022704 | 147.618218 | 147.627226 ± 0.004745 |
| 1 | 81.647775 ± 0.002782 | 81.648356 | 81.676032 ± 0.007727 |
| 2 | 115.581622 ± 0.000995 | 115.573771 | 115.585639 ± 0.010747 |

Std dùng ddof=0 qua hai restarts, không phải confidence interval.
Baseline thắng prior 2/6 cặp và zero-update 4/6 cặp; không client nào thắng
prior ở cả hai restarts. Bốn trong sáu baseline có best objective ở bước cuối,
nên chưa chứng minh hội tụ. Mean MSE lớn hơn median nhiều, phản ánh lỗi đuôi lớn.
Không dùng số lượng hàng lớn để coi sáu cặp này là bằng chứng độc lập mạnh.

### 9.5. Kiểm tra, compute và bàn giao

- 20 tests pass (Phase 1 invariants, Phase 2 metrics, Phase 3 update, follow-up).
- 20 trajectories mới: 8 development và 12 confirmation; 24 checkpoint records.
- 24 candidate được load lại và objective kiểm tra; 10 initialization pairing
  checks. Summary kiểm tra history minimum, budget, protocol hash và completion.
- Native local-update replay được kiểm tra trước các cặp attack. Không nới tolerance.
- Không có failed job trong follow-up hoàn tất.
- Development raw baseline+zero 500 bước: 769.35 giây; scaled 195.54 giây,
  categorical 193.54 giây, balanced 191.76 giây (mỗi cặp 100 bước).
- Confirmation mỗi cặp baseline+zero: client 0 là 364.39/345.81 giây,
  client 1 là 333.52/294.69 giây, client 2 là 202.67/209.45 giây.
  Đây là elapsed từng job gồm chấm/lưu, chạy đồng thời nên không cộng thành
  wall time toàn phiên. Không có số đo peak RAM/VRAM mới.

Runner: `experiments/phase3_followup.py`; summary:
`experiments/summarize_phase3_followup.py`; tests: `tests/test_phase3_followup.py`.
Outputs gồm original targets/source IDs, initialization, best reconstruction,
history, per-record metrics, `followup_report.json`, `confirmation_summary.csv`,
`development_convergence.png`, `manifest.json`.

Các lệnh đã thực thi (từ FL-DNA, output là thư mục kết quả ở trên):

~~~bash
.venv-phase1/bin/python -m experiments.phase3_followup prepare --output artifacts/phase3_followup/final_checks_v1
# dev chạy riêng cho raw, scaled, categorical, balanced
.venv-phase1/bin/python -m experiments.phase3_followup dev --variant raw --output artifacts/phase3_followup/final_checks_v1
.venv-phase1/bin/python -m experiments.phase3_followup freeze --output artifacts/phase3_followup/final_checks_v1
.venv-phase1/bin/python -m experiments.phase3_followup holdout --output artifacts/phase3_followup/final_checks_v1
# evaluate chạy riêng --client 0, 1, 2
.venv-phase1/bin/python -m experiments.phase3_followup evaluate --client 0 --output artifacts/phase3_followup/final_checks_v1
.venv-phase1/bin/python -m experiments.summarize_phase3_followup artifacts/phase3_followup/final_checks_v1
.venv-phase1/bin/python -m pytest -q -p no:cacheprovider tests/test_phase1_invariants.py tests/test_phase2_metrics.py tests/test_phase3_updates.py tests/test_phase3_followup.py
~~~

Khi chạy lại, dùng output mới để giữ evidence. Protocol chi tiết ở
`phase3_followup_protocol.md`. Development chỉ một client/restart; confirmation
chỉ một partition/checkpoint và budget 100, chưa kiểm chứng 500 trên tập mới.

**Kết luận:** thực thi năm bước COMPLETE; baseline reconstruction INCONCLUSIVE;
Phase 4 privacy ranking NOT READY. Attack chạy đúng tín hiệu local update nhưng
chưa chứng minh lợi ích tái dựng ổn định hơn prior. Không suy ra DNA bảo vệ tốt
từ việc baseline yếu. Bước phát triển sau cần protocol mới cho attack strength
và tiêu chí tái dựng; chưa có căn cứ đặt ngưỡng thành công nên vẫn TBD.

## 10. Attack redevelopment follow-up: bounded one-batch validation

Sau kết quả full-client âm tính, một nhánh phát triển attack hẹp hơn được chạy
để kiểm tra xem lỗi nằm ở implementation, parameterization hay độ khó của tín
hiệu local update. Nhánh này không thay thế full FedAvg attack. Nó chỉ dùng
một local update nhỏ: 4 records, gồm 1 fraud và 3 non-fraud, một Adam step, labels
và thứ tự record đã biết, local RNG được replay, model architecture/optimizer và
checkpoint trước local training đều được attacker biết.

Thay đổi thuật toán chính:

- Thêm PaySim manifold parameterization: attack tối ưu 6 numeric feature gốc và
  5 logits của transaction type; hai feature balance difference được suy ra từ
  `oldbalanceOrg/newbalanceOrig` và `newbalanceDest/oldbalanceDest`.
- Thêm objective `balanced_bn`: so khớp update theo từng tensor thay vì flatten
  toàn bộ, đồng thời đưa BatchNorm running buffers vào objective.
- Candidate được chọn bằng attacker objective; ground truth chỉ dùng để chấm
  reconstruction MSE sau khi attack xong.
- Mỗi candidate được lưu ra `.pt`, load lại, replay local update và kiểm tra lại
  objective để bảo đảm artifact khớp best candidate.
- Cổng xác nhận yêu cầu baseline reconstruction MSE tốt hơn cả prior và
  zero-update control với mean delta âm, median delta âm và one-sided exact sign
  p < 0.05.

### 10.1. Bounded one-batch confirmation

Frozen attack trước confirmation: `manifold_bn1`, lr=0.05, 300 iterations,
3 restarts. Confirmation dùng 12 nhóm source-disjoint với development.

| Control | Wins | Mean delta | Median delta | Sign p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 10/12 | -342.856957 | -70.820009 | 0.019287 | PASS |
| Zero-update | 11/12 | -450.825332 | -121.020073 | 0.003174 | PASS |

Delta = baseline MSE - control MSE, nên delta âm nghĩa là attack dùng update thật
tái dựng tốt hơn control. Kết quả này xác nhận attack bounded one-batch có tín
hiệu thật để khai thác trong phạm vi nhỏ này. Nó vẫn chưa chứng minh full-client
FedAvg update inversion.

### 10.2. Paired DNA evaluation under the validated bounded attack

Lossless DNA encode/decode được kiểm tra trên toàn bộ confirmation signals và
cho max abs error = 0.0. Vì vậy lossless DNA là transport comparator bit-exact:
nó không thay đổi tín hiệu attack trong protocol này.

DNA Transform conservative được đánh giá theo cùng targets, cùng seeds, cùng
ngân sách và cùng labels/order/RNG. Attacker biết transform config và seed, dùng
exact forward DNA transform với identity BPDA backward.

| Setting | Wins where DNA MSE > baseline | Mean DNA-baseline MSE | Median DNA-baseline MSE | Sign p |
|---|---:|---:|---:|---:|
| DNA Transform conservative + BPDA | 8/12 | -6.004947 | +10.610876 | 0.193848 |

MSE cao hơn nghĩa là reconstruction khó hơn. Median đi theo hướng có lợi cho
DNA Transform, nhưng mean lại hơi âm do outlier lớn và sign test không đạt. Do
đó kết luận đúng là **inconclusive**, không phải DNA Transform đã bảo vệ ổn định.
BPDA cũng chỉ là adaptive approximation cho transform rời rạc, không phải đạo
hàm chính xác của seed/permutation rule.

### 10.3. Scale check: 16 records, 4 Adam steps

Cấu hình attack được freeze từ one-batch run và áp dụng nguyên trạng, không
retune, sang tier khó hơn: 16 records, 4 fraud, 12 non-fraud, 4 Adam steps.

| Control | Wins | Mean delta | Median delta | Sign p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 9/12 | -199.871000 | -40.938482 | 0.072998 | FAIL |
| Zero-update | 11/12 | -248.595570 | -104.490508 | 0.003174 | PASS |

Attack vẫn tốt hơn zero-update nhưng không pass prior gate. Vì cổng baseline
chưa đạt ở tier này, DNA không được đánh giá ở scale check. Đây là quyết định
đúng về protocol: không dùng một attack chưa đủ mạnh để xếp hạng defense.

### 10.4. Files and commands

Artifacts chính:

- `artifacts/attack_redevelopment/run_20260908T180405538992Z/attack_redevelopment_report.json`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/dna_confirmation_report.json`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/attack_redevelopment_final_summary.json`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/attack_redevelopment_report.md`
- `artifacts/attack_redevelopment/scale_20260908T184713943293Z/attack_redevelopment_report.json`

Code chính:

- `attacks/tabular_parameterization.py`
- `attacks/adaptive_dna.py`
- `experiments/run_attack_redevelopment.py`
- `experiments/run_adaptive_dna_confirmation.py`
- `experiments/run_attack_scale_confirmation.py`
- `experiments/summarize_attack_redevelopment.py`
- `tests/test_attack_redevelopment.py`

Lệnh tái lập:

~~~bash
.venv-phase1/bin/python -m experiments.run_attack_redevelopment
.venv-phase1/bin/python -m experiments.run_adaptive_dna_confirmation artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.run_attack_scale_confirmation --source-run artifacts/attack_redevelopment/run_20260908T180405538992Z
.venv-phase1/bin/python -m experiments.summarize_attack_redevelopment artifacts/attack_redevelopment/run_20260908T180405538992Z artifacts/attack_redevelopment/scale_20260908T184713943293Z
~~~

### 10.5. Updated Phase 3 conclusion

Phase 3 có thêm một kết quả dương tính ở phạm vi bounded one-batch: baseline
attack đã tái dựng tốt hơn prior và zero-update một cách ổn định. Tuy nhiên
đây chưa phải full FedAvg client-update attack. DNA Transform trong phạm vi này
cho tín hiệu lẫn lộn và chưa đủ cơ sở để tuyên bố bảo vệ. Khi tăng lên 16
records/4 Adam steps, baseline không pass prior gate, nên chưa có evaluator đủ
mạnh cho defense ở scale lớn hơn.

Trạng thái sau follow-up:

- Bounded one-batch attack: READY cho diagnostic nhỏ.
- DNA Transform defense claim: NOT READY, kết quả inconclusive.
- 16-record/four-step attack: CONDITIONAL/NOT READY vì fail prior gate.
- Full FedAvg update attack: NOT VALIDATED.

## 11. Scale attack redevelopment: 16 records / 4 Adam steps

Sau khi one-batch attack đã pass, Phase 3 được mở rộng thêm một lần ở tier
16 records / 4 Adam steps. Mục tiêu là kiểm tra xem attack có thể mạnh hơn ở
điều kiện gần FL hơn không trước khi dùng nó để đánh giá DNA. Đây vẫn chưa phải
full-client FedAvg update attack.

Protocol:

- Development riêng: 3 groups, mỗi group 16 records gồm 4 fraud và 12 non-fraud.
- Confirmation riêng: 12 groups source-disjoint với development và artifacts cũ.
- Cấu hình thử: `manifold_bn1`, `manifold_bn3`; attack lr 0.03 và 0.05.
- Ngân sách: 600 iterations, confirmation 3 restarts.
- Selection: chọn cấu hình có max(mean delta vs prior, mean delta vs zero) thấp nhất.
- Gate giữ nguyên: cả prior và zero phải có mean delta âm, median delta âm và
  one-sided exact sign p < 0.05.

Development chọn `manifold_bn1`, lr=0.03, 600 iterations. Cấu hình này được
freeze trước confirmation.

| Stage | Variant | LR | Iterations | Prior mean delta | Zero mean delta |
|---|---|---:|---:|---:|---:|
| Development chosen | manifold_bn1 | 0.03 | 600 | -148.961696 | -246.722704 |

Confirmation:

| Control | Wins | Mean delta | Median delta | Sign p | Gate |
|---|---:|---:|---:|---:|---|
| Prior | 6/12 | -107.189003 | -2.559094 | 0.612793 | FAIL |
| Zero-update | 9/12 | -161.663003 | -61.009508 | 0.072998 | FAIL |

Mean delta vẫn âm, nghĩa là attack trung bình có cải thiện so với controls.
Nhưng wins không ổn định: chỉ 6/12 trước prior và 9/12 trước zero. P-value đều
không đạt cổng 0.05. Vì vậy tier 16-record vẫn **NOT READY** để đánh giá defense.
DNA Transform không được chạy ở tier này vì baseline evaluator chưa đủ tin cậy.

Run này mất 905.30 giây và lưu tại:

- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/attack_scale_redevelopment_report.json`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/development_summary.csv`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/confirmation_selected.csv`

Kết luận sau scale redevelopment: tăng iterations từ 300 lên 600 và chọn lại
cấu hình trên development không đủ để mở Phase 4. Phase 3 hiện có một attack
diagnostic nhỏ đã validate, nhưng chưa có attack đủ mạnh cho 16-record hoặc full
FedAvg client update.

## 12. Failure-pattern diagnostic: vì sao scale gate vẫn fail

Sau ba lần gặp cùng kiểu thất bại, Phase 3 bổ sung một diagnostic không đổi
protocol: đọc lại selected confirmation artifacts, tách reconstruction MSE theo
lớp fraud/non-fraud, và tính tensor-level matching loss cho selected baseline
candidates. Diagnostic này không chọn lại candidate, không chạy lại attack và
không thay đổi gate.

### 12.1. Tách theo lớp

Kết quả one-batch validated:

| Class | Control | Wins | Mean delta | Median delta |
|---|---|---:|---:|---:|
| Non-fraud | Prior | 8/12 | +23.313853 | -1.644455 |
| Non-fraud | Zero-update | 8/12 | -5.244579 | -35.450519 |
| Fraud | Prior | 10/12 | -1441.369387 | -483.239062 |
| Fraud | Zero-update | 11/12 | -1787.567590 | -596.438704 |

Kết quả scale redevelopment 16 records / 4 Adam steps:

| Class | Control | Wins | Mean delta | Median delta |
|---|---|---:|---:|---:|
| Non-fraud | Prior | 0/12 | +100.409617 | +81.648257 |
| Non-fraud | Zero-update | 2/12 | +74.163816 | +47.520288 |
| Fraud | Prior | 11/12 | -729.984863 | -189.804291 |
| Fraud | Zero-update | 12/12 | -869.143462 | -292.470469 |

Delta = baseline MSE - control MSE. Delta âm nghĩa là update thật giúp attack
tái dựng tốt hơn control. Diagnostic này cho thấy kết quả gộp đang che một
pattern quan trọng: attack khai thác tín hiệu fraud khá rõ, nhưng non-fraud lại
thua prior mạnh, nhất là ở tier 16-record. Vì mỗi group có 12 non-fraud và 4
fraud, lỗi non-fraud đủ lớn để làm fail gate tổng thể.

Diễn giải phù hợp là: attack hiện tại không scale đều qua lớp. Nó không hoàn
toàn mất tín hiệu; nó có tín hiệu tốt trên fraud, nhưng prior không dùng update
lại là đối chứng rất mạnh trên non-fraud. Do đó bước tiếp theo không nên là leo
ngay lên full-client, mà nên sửa cách objective/parameterization xử lý non-fraud
hoặc thiết kế control/prior phân phối rõ hơn.

### 12.2. Tensor-level diagnostic

Top tensor terms của tier 16-record:

| Tensor | Kind | Mean term | Mean cosine loss | Mean relative MSE |
|---|---|---:|---:|---:|
| `network.0.bias` | parameter | 1.192169 | 1.023003 | 1.691660 |
| `network.4.bias` | parameter | 1.154008 | 0.977549 | 1.764590 |
| `network.0.weight` | parameter | 1.127237 | 0.933180 | 1.940574 |
| `network.8.bias` | parameter | 1.073527 | 0.883622 | 1.899049 |
| `network.4.weight` | parameter | 1.053312 | 0.879291 | 1.740208 |

Các term lớn nhất vẫn nằm ở parameter tensors của các lớp đầu/giữa. BatchNorm
buffers có term thấp hơn trong selected reconstructions. Điều này gợi ý lỗi
không chỉ nằm ở BatchNorm buffer weighting; bài toán tối ưu đang không đưa
dummy inputs tới vùng non-fraud cụ thể hơn prior.

Artifacts:

- `artifacts/attack_redevelopment/run_20260908T180405538992Z/failure_pattern_diagnostics.md`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/failure_pattern_class_summary.csv`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/failure_pattern_tensor_summary.csv`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/failure_pattern_diagnostics.md`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/failure_pattern_class_summary.csv`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/failure_pattern_tensor_summary.csv`

Updated next step: giữ Phase 3 ở trạng thái **NOT READY for full FL defense
claims**. Hướng phát triển tiếp theo nên nhắm vào non-fraud reconstruction và
objective weighting trước khi tăng quy mô.

## 13. Class-gradient contribution diagnostic

Để phân biệt lỗi objective với đặc tính của update signal, Phase 3 bổ sung một
diagnostic không chạy inversion: với từng confirmation group, tính gradient
riêng từ fraud-only subset và non-fraud-only subset tại checkpoint trước local
training. Norm được báo ở hai dạng:

- Raw norm: norm của gradient trung bình trên riêng lớp đó.
- Weighted norm: raw norm nhân với tỷ lệ record của lớp trong group, tương ứng
  scale của mean-reduction loss.

Diagnostic này không thay đổi gate và không dùng để chọn candidate.

### 13.1. Scale 16 records / 4 Adam steps

Ở tier 16-record, mỗi group có 4 fraud và 12 non-fraud nên có thể đo gradient
theo lớp. Có ba mode:

- Eval: tắt BatchNorm batch statistics và Dropout.
- Train separate subsets: chạy riêng fraud-only và non-fraud-only subsets. Mode
  này đo độ lớn theo lớp nhưng dùng BatchNorm statistics khác với batch gộp.
- Train shared BN: chạy một full-batch train-mode forward, rồi tách per-sample
  loss trên cùng graph. Đây là diagnostic sát attack simulator nhất.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| Eval | 12 | 12/12 | 30.172757 | 30.542365 | 0.997142 | -0.380780 |
| Train separate subsets | 12 | 12/12 | 138.640140 | 106.837831 | 0.039169 | -0.171198 |
| Train shared BN | 12 | 12/12 | 124.319230 | 105.290608 | 0.999963 | 0.310647 |

Con số `Train separate subsets` có cosine fraud/full thấp vì BatchNorm nhìn từng
subset riêng, không cùng điều kiện với update thật. Khi đo bằng `Train shared BN`,
fraud contribution lại gần trùng hướng full gradient (cosine 0.999963). Sau khi
nhân tỷ lệ lớp, fraud vẫn lớn hơn non-fraud hơn 100 lần theo median. Vì vậy mâu
thuẫn eval/train trước đó là artifact của cách đo subset riêng, không phải bằng
chứng rằng train-mode làm mất tín hiệu fraud. Observed update ở tier này vẫn bị
fraud chi phối mạnh.

### 13.2. One-batch 4 records / 1 Adam step

Ở one-batch, mỗi group chỉ có 1 fraud và 3 non-fraud. Eval-mode vẫn đo được,
nhưng train-mode fraud-only bị skip vì BatchNorm train không hợp lệ với batch
size 1.

| Mode | Groups | Fraud weighted norm > non-fraud | Mean weighted ratio | Median weighted ratio | Mean fraud cosine/full | Mean non-fraud cosine/full |
|---|---:|---:|---:|---:|---:|---:|
| Eval | 12 | 10/12 | 35.791337 | 26.700656 | 0.830354 | 0.092965 |
| Train shared BN | 12 | 12/12 | 118.325667 | 118.498655 | 0.999658 | 0.394465 |

Train shared BN vẫn chạy được ở one-batch vì nó dùng full batch 4 records trước
khi tách loss theo lớp. Fraud contribution cũng gần trùng full gradient
(cosine 0.999658) và weighted norm lớn hơn non-fraud hơn 118 lần theo median.
One-batch pass gate chủ yếu nhờ fraud signal; non-fraud/prior vẫn có mean delta
dương (+23.313853), tức prior vẫn cạnh tranh tốt trên non-fraud.

### 13.3. Interpretation

Diagnostic này ủng hộ giả thuyết rằng failure pattern là đặc tính của tín hiệu
training trên PaySim/focal loss, không chỉ là một bug trong attack code. Focal
loss với alpha=0.95 làm fraud records tạo gradient lớn hơn rất nhiều so với
non-fraud. Khi update gộp nhiều record, tín hiệu fraud chi phối hướng update,
còn non-fraud dễ bị kéo lệch khỏi prior.

Vì vậy có hai hướng hợp lệ, tùy mục tiêu nghiên cứu:

- Nếu muốn đánh giá leakage tự nhiên của FL fraud detection, giữ attack hiện tại
  và báo cáo leakage tách lớp: fraud rò rỉ rõ hơn non-fraud.
- Nếu muốn xây attacker-favorable evaluator mạnh nhất, cần thiết kế một
  class-decomposed hoặc class-reweighted objective và ghi rõ đó là relaxation
  có lợi cho attacker, không phải server-side observation mặc định.

Không nên tăng tiếp lên full-client chỉ để tìm p-value. Nút thắt hiện tại là
class-skewed signal và non-fraud reconstruction, không phải số sample.

Artifacts:

- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/class_gradient_contribution.md`
- `artifacts/attack_redevelopment/scale_dev_20260908T191737994986Z/class_gradient_contribution_summary.csv`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/class_gradient_contribution.md`
- `artifacts/attack_redevelopment/run_20260908T180405538992Z/class_gradient_contribution_summary.csv`

## 14. Class-decomposed attacker diagnostic

Đã chạy thêm một diagnostic theo hướng attacker-favorable để kiểm tra nút thắt
non-fraud. Runner mới không dùng full local Adam update như tín hiệu server nhìn
thấy. Thay vào đó, nó cấp cho attacker hai tín hiệu oracle: fraud contribution
và non-fraud contribution, đều tính từ cùng một full-batch train-mode graph.
Objective khớp hai contribution này với trọng số lớp bằng nhau. Ground truth
không được dùng để chọn candidate; candidate vẫn chọn bằng attack objective và
được load lại để kiểm tra objective sau lưu.

Artifacts:
`artifacts/attack_redevelopment/class_decomp_20260908T200353687099Z/`.
Protocol: 16 records/group, 4 fraud + 12 non-fraud, 3 development groups,
12 confirmation groups, 100 iterations, 2 restarts. Development chọn
`manifold_bn1`, lr=0.05 trước confirmation. Runtime 39.85 giây.

| So sánh | Thắng / nhóm | Mean delta MSE | Median delta MSE | p một phía |
|---|---:|---:|---:|---:|
| Baseline vs prior | 11/12 | -226.194423 | -92.524752 | 0.003174 |
| Baseline vs zero-update | 11/12 | -134.849582 | -87.820915 | 0.003174 |
| Fraud vs prior | 12/12 | -911.952115 | -343.784590 | 0.000244 |
| Non-fraud vs prior | 5/12 | +2.391475 | +0.860072 | 0.806152 |

Kết quả này giải quyết câu hỏi chẩn đoán chính. Khi fraud/non-fraud signal được
tách và objective được cân bằng, attack tổng thể pass gate, nhưng phần cải thiện
đến gần như toàn bộ từ fraud records. Non-fraud vẫn không thắng prior: mean và
median delta đều dương, chỉ 5/12 nhóm tốt hơn prior. Vì vậy thất bại non-fraud
không chỉ do aggregate objective cũ chôn tín hiệu; với diagnostic oracle hiện
tại, non-fraud vẫn chưa cho thấy rò rỉ tái dựng ổn định.

Không dùng kết quả này để tuyên bố attack server-side đã sẵn sàng. Server FL
chuẩn không nhìn thấy class-separated gradients, và Secure Aggregation càng
không cung cấp individual update. Kết luận hợp lệ ở đây hẹp hơn: PaySim/focal
loss tạo leakage mạnh hơn trên fraud records, còn non-fraud reconstruction vẫn
yếu ngay cả trong kiểm tra có lợi cho attacker. Để đánh giá DNA trong full FL,
cần một attacker thích nghi trên đúng transmitted update, hoặc một protocol mới
khóa trước nếu muốn dùng class-decomposed diagnostic như upper-bound analysis.

## 15. Class feature-spread diagnostic

Để kiểm tra vì sao prior cạnh tranh tốt trên non-fraud, đã chạy thêm diagnostic
thống kê trên feature representation sau RobustScaler và one-hot type. Diagnostic
này không chạy attack và không đánh giá DNA; nó chỉ đo xem records của từng lớp
có nằm gần centroid lớp của chúng hay không.

Artifacts:
`artifacts/attack_redevelopment/class_feature_spread/`.

| Scope | Class | n | Mean centroid MSE | Median centroid MSE | Mean pairwise MSE | Mean feature variance |
|---|---|---:|---:|---:|---:|---:|
| train_reference_500k | non-fraud | 324581 | 100.779240 | 8.393242 | 299.863343 | 100.779240 |
| train_reference_500k | fraud | 419 | 4038.719079 | 1325.761277 | 8096.762172 | 4038.719079 |
| one-batch confirmation | non-fraud | 36 | 30.056516 | 4.235694 | 61.830547 | 30.056516 |
| one-batch confirmation | fraud | 12 | 2751.483347 | 932.430985 | 6003.236394 | 2751.483347 |
| scale 16-record confirmation | non-fraud | 144 | 76.543792 | 4.019673 | 154.158127 | 76.543792 |
| scale 16-record confirmation | fraud | 48 | 1055.848379 | 346.621365 | 2156.626476 | 1055.848379 |
| class-decomposed confirmation | non-fraud | 144 | 98.094265 | 7.236811 | 197.560478 | 98.094265 |
| class-decomposed confirmation | fraud | 48 | 3538.482891 | 1205.339318 | 7227.539522 | 3538.482891 |

Fraud/non-fraud spread ratio:

| Scope | Mean centroid ratio | Median centroid ratio | Mean pairwise ratio | Mean variance ratio |
|---|---:|---:|---:|---:|
| train_reference_500k | 40.074911 | 157.955794 | 27.001507 | 40.074911 |
| one-batch confirmation | 91.543655 | 220.136513 | 97.091756 | 91.543655 |
| scale 16-record confirmation | 13.794043 | 86.231225 | 13.989703 | 13.794043 |
| class-decomposed confirmation | 36.072271 | 166.556684 | 36.583934 | 36.072271 |

Fraud records are much more dispersed than non-fraud records in all checked
scopes. On the 500k train reference, fraud mean centroid MSE is 40.07 times
higher and median centroid MSE is 157.96 times higher. The same pattern appears
inside the attack target sets. The largest fraud variance is consistently tied
to `balance_diff_orig`, while non-fraud records have much smaller IQRs around
their typical balance/amount values.

This supports the current interpretation: non-fraud is difficult to evaluate by
reconstruction MSE because a prior-like guess is already close to many
non-fraud samples. Fraud records are both gradient-dominant and statistically
more spread out, so successful attack signal is easier to observe there. This
does not prove that non-fraud is private; it only explains why the current
attack/control protocol sees little reconstruction gain over prior for that
class.

## 16. Objective audit and balanced-tensor rerun

Đã kiểm tra lại đúng câu hỏi objective: một số đường Phase 3 không giống nhau.
Full-client runner ban đầu dùng trung bình MSE theo parameter tensor, nhưng
bounded Adam ladder dùng `cosine_magnitude`, trong đó candidate và observed
state được flatten thành một vector dài. Vì observed signal gồm cả BatchNorm
running buffers, đường này có nguy cơ bị buffer lớn chi phối.

### 16.1. Tensor norm audit

Kiểm tra trên confirmation group 0 của `adam_ladder_v2` cho thấy vấn đề thật:

| Thành phần | Norm |
|---|---:|
| `network.1.running_var` | 391.298035 |
| Tất cả running buffers | 391.403415 |
| Parameter-like tensors | 0.100341 |
| Running-buffer / parameter-like ratio | 3900.714456 |

Vì running buffer lớn hơn parameter tensors hàng nghìn lần, flatten objective
không phải phép đo cân bằng giữa các tensor. Đây là lỗi trong bounded/ladder
objective, không phải trong mọi runner Phase 3.

### 16.2. Sửa objective

Đã thêm mode `balanced_tensor` trong `phase3_bounded_validation.update_objective`.
Mode này tính term riêng cho từng tensor, chuẩn hóa theo năng lượng tensor tham
chiếu, rồi lấy trung bình các tensor. Đường cũ `cosine_magnitude` vẫn giữ để
đọc lại artifact cũ và so sánh protocol cũ.

Regression test mới kiểm tra rằng một BatchNorm buffer cực lớn không được phép
làm mất tín hiệu của tensor nhỏ khi dùng `balanced_tensor`.

### 16.3. Rerun 4-record / 1-step Adam trên targets cũ

Rerun dùng đúng `adam_ladder_v2/confirmation_targets.pt`: 10 groups, mỗi group
4 records gồm 1 fraud và 3 non-fraud, 300 iterations, 3 restarts, lr=0.10.
Artifacts:
`artifacts/phase3_closure/adam_ladder_balanced_objective_20260909T060144815783Z/`.

| Objective | Control | Thắng / nhóm | Mean delta MSE | Median delta MSE | p một phía | Gate |
|---|---|---:|---:|---:|---:|---|
| Old flatten `cosine_magnitude` | Prior | 7/10 | +74.356646 | -10.676704 | 0.171875 | FAIL |
| Old flatten `cosine_magnitude` | Zero-update | 7/10 | +49.654386 | -79.534573 | 0.171875 | FAIL |
| New `balanced_tensor` | Prior | 9/10 | -12.647373 | -5.431931 | 0.010742 | PASS |
| New `balanced_tensor` | Zero-update | 6/10 | -12.656906 | -1.973405 | 0.376953 | FAIL |

Sửa objective giúp baseline vượt prior rõ hơn ở scope nhỏ nhất, nhưng chưa vượt
zero-update ổn định. Vì gate yêu cầu thắng cả prior và zero-update, run này vẫn
**BALANCED_OBJECTIVE_GATE_FAILED**. Do đó chưa chạy lại 16-record hoặc full-client
theo objective mới.

Hiện tượng prior tốt lên nhưng zero-update không tốt lên cần được đọc cẩn thận.
Objective cân bằng làm attack bớt bị `running_var` chi phối và khớp tốt hơn các
tensor parameter nhỏ, nên reconstruction vượt prior ở nhiều group hơn. Nhưng
zero-update cũng được tối ưu bằng cùng objective mới và cùng initialization; nó
có thể tìm được một dummy update nhỏ/điển hình khá cạnh tranh trong scope rất
nhỏ này. Nói cách khác, objective mới cải thiện khả năng vượt một prior tĩnh,
nhưng chưa chứng minh observed update chứa đủ thông tin riêng biệt để vượt
đối chứng zero-update ổn định.

### 16.4. Tách lớp sau khi sửa objective

Diagnostic theo lớp trên run mới:

| Class | Control | Thắng / nhóm | Mean delta | Median delta |
|---|---|---:|---:|---:|
| Non-fraud | Prior | 5/10 | -2.385704 | -0.437396 |
| Non-fraud | Zero-update | 5/10 | +0.361362 | +0.173888 |
| Fraud | Prior | 9/10 | -43.432379 | -12.966417 |
| Fraud | Zero-update | 8/10 | -51.711708 | -22.042208 |

Fraud vẫn là nơi có tín hiệu mạnh hơn. Non-fraud chỉ cải thiện nhẹ so với prior
và không vượt zero-update. Vì vậy sửa objective làm kết quả nhỏ tốt hơn, nhưng
không đảo kết luận class-wise.

Gradient contribution trên cùng targets:

| Mode | Groups | Fraud weighted norm > non-fraud | Median weighted ratio | Fraud cosine/full |
|---|---:|---:|---:|---:|
| Eval | 10 | 8/10 | 22.356134 | 0.766478 |
| Train shared BN | 10 | 10/10 | 114.022751 | 0.999962 |

Khoảng cách fraud/non-fraud không giảm sau khi sửa objective; ở train shared BN
nó vẫn khoảng 114 lần theo median. Vì vậy con số khoảng 105 lần trước đó không
phải artifact của objective flatten. Nó là thuộc tính của observed gradient trên
targets này.

Eval mode thay đổi nhiều hơn vì đó là diagnostic phụ: nó tắt BatchNorm batch
statistics và Dropout, nên nhạy với target set cụ thể và không phản ánh đúng
attack simulator train-mode. Kết luận chính lấy từ `Train shared BN`, vì mode
này dùng một full-batch train graph rồi mới tách loss theo lớp. Theo mode chính
này, fraud vẫn gần trùng hướng full gradient và vẫn áp đảo non-fraud.

### 16.5. Bậc tự do thô

Mô hình có 13,057 floating state observations: 12,609 parameters và 448 floating
BatchNorm buffers. Số ẩn nếu tái dựng input là `records x 13 features`, hoặc
`records x 11 latent variables` với PaySim manifold.

| Scope | Records | Unknowns 13-feature | Unknowns manifold | Obs / 13-feature unknowns |
|---|---:|---:|---:|---:|
| 4-record | 4 | 52 | 44 | 251.096154 |
| 16-record | 16 | 208 | 176 | 62.774038 |
| Client 0 | 132175 | 1718275 | 1453925 | 0.007599 |
| Client 1 | 118963 | 1546519 | 1308593 | 0.008443 |
| Client 2 | 73862 | 960206 | 812482 | 0.013598 |

Đây chỉ là ước lượng bậc tự do thô. Nó không tính ràng buộc dữ liệu, one-hot
transaction type, quan hệ balance-derived features, optimizer dynamics hoặc
prior. Vì vậy không dùng nó như chứng minh bất khả thi; nó chỉ giải thích vì
sao full-client inversion khó hơn rất nhiều so với bounded groups.

Ở chiều ngược lại, scope 4-record có số quan sát nhiều hơn số ẩn rất lớn
(`Obs / unknown = 251.096154`) nhưng vẫn chưa pass zero-update gate. Điều này
cho thấy thất bại ở scope nhỏ không phải do thiếu số phương trình thô. Nút thắt
ở đây nằm ở chất lượng tín hiệu dùng để phân biệt record cụ thể: non-fraud gần
prior hơn, còn fraud mới là phần gradient rõ nhất. Với full-client, cả hai yếu
tố cùng tồn tại: tín hiệu từng record bị pha loãng và số ẩn lớn hơn số quan sát
rất nhiều.

### 16.6. Quyết định sau checklist

Kết quả không rơi vào Nhánh A. Objective flatten đúng là lỗi trong bounded ladder
và đã sửa, nhưng scope 4-record vẫn không pass đủ hai controls, còn fraud/non-fraud
gradient gap vẫn trên 100 lần trong train shared BN. Kết luận Phase 3 vì vậy
không bị đảo ngược: baseline attack có tín hiệu ở fraud, yếu ở non-fraud, và
chưa đủ cơ sở để đánh giá khả năng bảo vệ DNA trong full FL. Hướng tiếp theo
nếu mở Phase 4 vẫn nên là attack chính thức trên đúng transmitted update, với
fraud-focused analysis được khóa protocol trước.
