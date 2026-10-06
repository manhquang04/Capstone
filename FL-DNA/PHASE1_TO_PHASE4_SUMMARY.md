# FL-DNA — Tổng hợp dễ hiểu Phase 1 → Phase 4

Tài liệu này giải thích lại toàn bộ quá trình nghiên cứu bằng ngôn ngữ đơn
giản, kể cả những thử nghiệm đã thử, tại sao thử, và kết quả ra sao — kể cả
khi thử nghiệm đó thất bại hoặc phải sửa lại nhiều lần. Nguồn gốc số liệu:
`PROJECT.md`, `PHASE1_AUDIT.md`, `phase2_report.md`, `phase3_report.md`,
`phase4_protocol.md`, `phase4_report.md`, `phase4_dna_transform_spec.md`,
`fl_dna_overleaf_paper.tex`, và các artifact JSON/CSV trong `artifacts/`.

## 0. Bài toán bằng một câu

Có một hệ thống Federated Learning (FL) phát hiện gian lận giao dịch: nhiều
"client" (ví dụ nhiều ngân hàng/chi nhánh) tự huấn luyện mô hình trên dữ liệu
riêng của mình, rồi chỉ gửi *update* (thay đổi trọng số mô hình) lên server để
gộp lại (FedAvg) — không gửi dữ liệu gốc. Câu hỏi: server (hoặc kẻ tấn công
đứng ở vị trí server) có thể nhìn update đó và **đoán ngược lại** giao dịch gốc
hay không? Và nếu có một cách "xáo trộn" update lấy cảm hứng từ mã hóa DNA
sinh học (gọi là **DNA Transform**) thì nó có thực sự làm việc đoán ngược khó
hơn không, hay chỉ là làm phức tạp hóa vô ích?

### Ba câu hỏi nghiên cứu chuẩn (RQ chính thức của đề tài)

Đây là bộ câu hỏi nghiên cứu chính thức, chuẩn, dùng làm khung tham chiếu cho
toàn bộ tài liệu này (thay cho các bản RQ khác từng xuất hiện rải rác trong
`PROJECT.md` hoặc bản thảo `fl_dna_overleaf_paper.tex` — những bản đó là các
cách chia nhỏ/soạn thảo nội bộ ở các thời điểm khác nhau, không phải bộ RQ
chuẩn cuối cùng):

- **RQ1:** DNA encoding có bảo vệ gradient khỏi gradient inversion attack
  hiệu quả hơn Differential Privacy (DP) không, đo định lượng bằng PSNR/SSIM
  của dữ liệu tái dựng được?
  *(Nguyên văn: "Does DNA encoding protect gradients against gradient
  inversion more effectively than Differential Privacy, measured
  quantitatively (PSNR/SSIM of the reconstructed data)?")*
- **RQ2:** Việc áp dụng DNA encoding ảnh hưởng thế nào đến độ chính xác cuối
  cùng của mô hình toàn cục (đo bằng F1-score và AUC-ROC)?
  *(Nguyên văn: "How does applying DNA encoding affect the final accuracy of
  the global model (measured by F1-score and AUC-ROC)?")*
- **RQ3:** Chi phí tính toán và băng thông của DNA encoding có nằm trong
  phạm vi chấp nhận được để triển khai thực tế không?
  *(Nguyên văn: "Are the computational and bandwidth costs of DNA encoding
  within an acceptable range for real-world deployment?")*

### RQ1/RQ2/RQ3 được trả lời ở đâu trong toàn bộ dự án

Cần phân biệt rõ hai mảng công việc khác nhau trong repo, vì chúng trả lời
RQ1/RQ2/RQ3 bằng hai cách đo khác nhau:

- **Mảng 50-round utility/attack chính thức** (mô tả trong `README.md`, không
  thuộc phạm vi Phase 1–4 nói ở tài liệu này): đây là nơi trả lời RQ1 đúng
  nghĩa đen (so PSNR/SSIM của DNA Transform/DNA lossless với DP, bảng "Accepted
  Gradient Inversion Results"), RQ2 (bảng F1/ROC-AUC 50-round, "Accepted
  50-Round Utility Results"), và một phần RQ3 (thời gian DNA Transform theo
  ms/round ở "Experiment 2: Client Scalability Test", và bảng
  `privacy_utility_tradeoff.csv`). Attack ở mảng này là attack gradient một
  mẫu đơn giản (chưa phải update FedAvg thật), nên PSNR/SSIM ở đây chỉ mang
  tính chỉ báo ban đầu, chưa được kiểm định chặt bằng đối chứng/thống kê.
- **Mảng Phase 1 → Phase 4** (nội dung chính của tài liệu này): đây là nơi
  làm sâu và **kiểm định chặt chẽ hơn nhiều** phần lõi của RQ1 — tức "DNA có
  thực sự bảo vệ tốt hơn không bảo vệ/đối chứng đơn giản không" — bằng cách
  xây một attacker được kiểm định thống kê (đối chứng Prior/Zero-update,
  p-value, tập dữ liệu tách biệt dev/eval) và áp dụng đúng lên **update FedAvg
  thật** (không chỉ gradient một mẫu), thay vì chỉ dùng PSNR/SSIM ở gradient
  đơn giản. Nói cách khác, Phase 1–4 là phần "làm cho RQ1 đáng tin cậy hơn",
  dùng metric MSE/cosine/relative-L2 thay vì PSNR/SSIM, và có so sánh thêm với
  Differential Privacy-style (clipping/noise) như một trong các đối chứng.
  Phase 1–4 **không** trực tiếp đo RQ2 (utility F1/AUC-ROC) hay RQ3 (chi phí
  tính toán/băng thông) ở đúng quy mô nhỏ (4 record/1 bước) mà nó dùng để đánh
  giá attack — hai câu hỏi này vẫn dựa vào bảng 50-round trong `README.md`
  (xem "Việc còn dang dở" ở mục 6).

Vì vậy, đọc tài liệu này (mục 1 → 4) như phần "đào sâu và kiểm định lại RQ1"
bằng một phương pháp khoa học chặt hơn PSNR/SSIM một lần chạy, chứ không phải
một bộ RQ khác. Ba câu hỏi con dưới đây (RQ1a/RQ1b/RQ1c) là cách nội bộ để chia
nhỏ RQ1 thành các phần kiểm chứng được, tương ứng với 3 thứ đội nghiên cứu cần
làm rõ trước khi trả lời "có bảo vệ tốt hơn không" một cách có căn cứ:

- **RQ1a — có giảm reconstruction không?** DNA Transform có làm giảm khả năng
  bị tái dựng dữ liệu (so với hoàn toàn không bảo vệ) không?
- **RQ1b — lợi ích có phải "hàng thật" không?** Nếu có giảm, lợi ích đó có
  đến từ chính cơ chế DNA, hay chỉ vì nó làm update bị méo/nhiễu đi theo cách
  mà một phép làm nhiễu đơn giản khác (random retention, top-k, clipping/
  noise — các phương án cùng họ với DP) cũng làm được y hệt?
- **RQ1c — có ổn định khi mở rộng không?** Nếu DNA thực sự có ích, lợi ích đó
  có ổn định khi mở rộng ra nhiều client hơn, dữ liệu khác hơn, hệ thống thật
  hơn (full-client, nhiều round) không?

Toàn bộ Phase 1–4 chỉ đủ sức trả lời RQ1a và một phần RQ1b, trong một phạm vi
rất hẹp và được kiểm soát chặt (chưa chạm tới RQ1c — đó vốn là phạm vi của
Phase 5, chưa từng bắt đầu). RQ2 và RQ3 của đề tài vẫn dựa vào kết quả
50-round có sẵn, chưa được đo lại ở đúng quy mô attack mà Phase 4 dùng.

## 1. Phase 1 — Dựng nền, kiểm tra máy móc có chạy đúng không

**Mục tiêu:** Trước khi kết luận bất cứ điều gì về bảo mật, phải chắc chắn
code chạy đúng: pipeline dữ liệu, mô hình, DNA lossless encode/decode, DNA
Transform, tấn công gradient inversion đều hoạt động như mô tả.

**Đã làm:**

- Chuẩn hóa dữ liệu PaySim (giao dịch giả lập gian lận): chọn cột, tạo 2 đặc
  trưng phái sinh (`balance_diff_orig`, `balance_diff_dest`), chia
  train/val/test theo tỷ lệ nhãn, mô phỏng 3 client với phân bố dữ liệu hơi
  lệch nhau (non-IID nhẹ).
- Xây mô hình MLP 3 lớp ẩn (128→64→32→1) với BatchNorm + Dropout, dùng focal
  loss (một loại loss ưu tiên nhận diện lớp hiếm — ở đây là fraud, vì fraud
  chỉ chiếm ~0.1–0.2% dữ liệu).
- Kiểm tra DNA lossless: mã hóa update thành chuỗi "DNA" rồi giải mã lại, phải
  ra đúng bit-for-bit như ban đầu (invariant test) — đây là đường truyền, KHÔNG
  phải cơ chế bảo vệ.
- Chạy thử một attack đơn giản: tấn công gradient của **một mẫu dữ liệu**
  (chưa phải update sau khi train), dùng nhãn biết trước, tối ưu một "dữ liệu
  giả" (dummy data) sao cho gradient của nó khớp với gradient quan sát được.
- Viết 5 bài kiểm tra "vệ sinh" (validation checks): kết quả có lặp lại được
  không, artifact lưu ra rồi đọc lại có khớp không, tăng ngân sách tối ưu có
  ảnh hưởng gì, so với "không có update thật" (zero-gradient) thế nào, và so
  DNA lossless với DNA Transform trên cùng target.

**Kết quả:** Code chạy đúng, không phát hiện lỗi implementation nghiêm trọng.
Nhưng thử nghiệm attack chỉ có **2 mục tiêu** (1 fraud, 1 non-fraud) — quá ít
để kết luận bất cứ điều gì về hiệu quả tấn công.

**Trạng thái:** Implementation PASS, hiệu quả tấn công INCONCLUSIVE (chưa đủ
dữ liệu để nói attack có "ăn" thật hay không). Không dùng làm bằng chứng khoa
học, chỉ là bước dọn đường.

## 2. Phase 2 — Kiểm tra attack có thực sự khai thác được tín hiệu thật không

**Vấn đề cần trả lời:** Ở Phase 1 mẫu quá ít. Phase 2 mở rộng lên **10 mục
tiêu** (5 fraud, 5 non-fraud), và so sánh attack "thật" (dùng gradient quan
sát được) với 2 đối chứng:

- **Zero-gradient control**: giả vờ gradient quan sát được = 0, xem attack tối
  ưu ra cái gì (nếu vẫn ra kết quả gần đúng thì chứng tỏ tín hiệu không đến từ
  gradient thật mà chỉ từ mô hình/khởi tạo).
- **Prior control**: đoán mù bằng giá trị trung bình dữ liệu training, không
  tối ưu gì cả — đại diện cho "không cần tấn công, chỉ cần đoán theo phân bố
  chung".

**Thử nghiệm phụ đã thử ở Phase 2:**

- **Thử bỏ regularization (L2=0) so với giữ L2=1e-4**: chỉ thử trên 2 mục
  tiêu phát triển (không phải mục tiêu đánh giá). Kết quả: L2=0 cho MSE dao
  động mạnh giữa các lần chạy lại (0.15–0.75) — không ổn định, nên **không**
  được chọn làm cấu hình chính thức, chỉ ghi nhận là một lựa chọn phát triển
  hẹp, chưa xác nhận.
- Chuẩn hóa lại cách đo lỗi: tách MAE theo từng đặc trưng, kiểm tra tính hợp
  lệ của vector one-hot (loại giao dịch) bằng "strict validity" (mỗi phần tử
  gần 0/1, đúng 1 phần tử gần 1) — phát hiện ra rằng **không có candidate nào**
  (kể cả baseline) đạt vector one-hot hợp lệ tuyệt đối, nên độ chính xác
  "đoán đúng loại giao dịch" (40–60% bằng argmax) chỉ mang tính tham khảo, chứ
  chưa chứng minh tái dựng đúng giao dịch thật.

**Kết quả chính:** Tính theo "thắng cả 3 lần chạy lại" trên từng mục tiêu, attack
thật thắng zero-gradient ở 8/10 mục tiêu và thắng prior ở 8/10 mục tiêu. Hai
mục tiêu còn lại là 2 giao dịch fraud có MSE rất lớn (outlier, ví dụ MSE lên
tới 16648) — dấu hiệu sớm cho thấy fraud khó/dễ tái dựng rất khác nhau tùy
từng giao dịch cụ thể (manh mối này sẽ trở thành phát hiện lớn ở Phase 3).

**Trạng thái:** SUPPORTED_ON_PILOT — đủ cơ sở để bước sang Phase 3, nhưng đây
vẫn chỉ là gradient của **một mẫu đơn lẻ**, chưa phải update FedAvg thật (vốn
là kết quả của nhiều bước huấn luyện cục bộ trên nhiều dòng dữ liệu).

## 3. Phase 3 — Tấn công vào update thật sau khi client huấn luyện xong

Đây là bước quan trọng: thay vì tấn công gradient của 1 mẫu, giờ tấn công
**update thật** mà một client gửi lên server sau khi đã huấn luyện cục bộ
(nhiều bước Adam, có BatchNorm, có nhiều dòng dữ liệu).

### 3.1. Cú sốc đầu tiên: attack thất bại hoàn toàn ở quy mô đầy đủ

Chạy thẳng attack lên update của **cả client** (73.862–132.175 dòng dữ liệu,
sau 1 vòng FedAvg thật): attack thua cả 2 đối chứng — thậm chí thua cả
"zero-update" (giả vờ không có update nào được gửi lên)!

| Method | MSE |
|---|---:|
| Prior (đoán mù) | 117.10 |
| Baseline (dùng update thật) | 129.47 |
| Zero-update (giả vờ update = 0) | 129.40 |

Đây là dấu hiệu bất thường nghiêm trọng — cần chẩn đoán tại sao.

### 3.2. Chuỗi chẩn đoán từng bước (10 vòng thử nghiệm)

1. **Chạy thẳng full-client** → thất bại (như trên).
2. **Kiểm tra lại công thức đạo hàm BatchNorm** trong hàm mô phỏng huấn luyện
   cục bộ — phát hiện một lỗi thật trong công thức tính gradient của
   `running_mean`/`running_var`. Sửa lỗi này là đúng đắn, nhưng sau khi sửa,
   vấn đề chính vẫn còn nguyên → chứng tỏ đây không phải nguyên nhân gốc.
3. **Thu nhỏ quy mô dần dần** để cô lập vấn đề: thử ở quy mô rất nhỏ trước
   (1 batch/4 dòng dữ liệu/1 bước Adam), rồi tăng dần (4 batch/16 dòng/4
   bước), rồi mới thử full-client. Đây là chiến lược "leo thang có kiểm soát"
   thay vì đâm đầu vào bài toán khó nhất ngay từ đầu.
   - Ở quy mô nhỏ nhất (4 dòng/1 bước): có dấu hiệu tích cực (thắng 7/10
     control) nhưng p=0.172 — chưa đạt ngưỡng ý nghĩa thống kê p<0.05.
   - Ở quy mô 16 dòng/4 bước: **tệ đi rõ rệt**, chỉ thắng prior 6/12.
4. **Thử 4 công thức "objective" khác nhau** (cách đo độ khớp giữa update giả
   và update thật): dùng sai số thô, chia theo bình phương learning rate, đổi
   cách tham số hóa phần loại giao dịch (softmax), thêm ràng buộc balance-diff.
   → Không cải thiện được gì đáng kể.
5. **Đổi sang bộ dữ liệu 500k dòng hoàn toàn khác** (source-disjoint) để loại
   trừ khả năng "chẳng qua chọn nhầm phải bộ dữ liệu khó". → Kết quả giống hệt
   → loại trừ giả thuyết này.
6. **Mở rộng kiến trúc attack** (thử thêm 2 biến thể `manifold_bn1`,
   `manifold_bn3`), tăng gấp đôi số vòng lặp tối ưu (lên 600) → vẫn thất bại
   theo cùng một kiểu.
7. **Bước ngoặt — tách kết quả theo lớp fraud/non-fraud** (thay vì chỉ nhìn
   trung bình gộp cả 2 lớp): phát hiện ra kết quả "tệ" ở các bước trước thực
   ra là **hai bức tranh hoàn toàn trái ngược bị trộn lẫn với nhau**:
   - Trên **fraud**: attack thắng áp đảo (11–12/12 nhóm).
   - Trên **non-fraud**: attack gần như thua tuyệt đối (0–2/12 nhóm).
   
   Vì mỗi nhóm chỉ có 4 fraud/12 non-fraud (tỷ lệ mất cân bằng phản ánh đúng
   dữ liệu thật), phần thắng khổng lồ trên fraud và phần thua trên non-fraud
   "trung hòa" nhau khi gộp trung bình, khiến ta tưởng attack yếu, trong khi
   thực ra nó rất mạnh trên một lớp và rất yếu trên lớp kia.
8. **Đo trực tiếp mức độ "gradient bị chi phối bởi fraud"**: gradient của các
   giao dịch fraud có độ lớn gấp ~105–121 lần gradient của non-fraud (median
   ratio). Cosine similarity giữa gradient-fraud-riêng và gradient-tổng gần
   bằng 1 (0.9999) — tức là gradient tổng gần như *chính là* gradient của
   fraud, non-fraud gần như "vô hình" trong tín hiệu quan sát được.
   
   *Lưu ý kỹ thuật quan trọng*: lần đo đầu tiên bị sai vì tính thống kê
   BatchNorm riêng cho từng nhóm nhỏ (subset) thay vì tính trên batch gộp thật
   — đã sửa bằng cách dùng đúng chế độ `train_shared_bn` (BatchNorm nhìn đúng
   batch gộp như khi huấn luyện thật).
9. **Kiểm tra giả thuyết "chỉ vì objective bị lỗi"**: xây một phiên bản attack
   "oracle" được cấp thẳng nhãn fraud/non-fraud riêng biệt (chỉ dùng để chẩn
   đoán, không phải attacker chính thức vì đây là thông tin mà attacker thật
   không có quyền biết). Kể cả khi được ưu ái tối đa như vậy, non-fraud vẫn
   thua prior (5/12, p=0.806), trong khi fraud thắng tuyệt đối (12/12,
   p=0.0002). → Bác bỏ giả thuyết "chỉ là lỗi objective"; đây là đặc tính thật
   của dữ liệu/mô hình.
10. **Đo độ "phân tán" của dữ liệu trong không gian đặc trưng**: fraud phân
    tán gấp 150–220 lần so với non-fraud (đo bằng khoảng cách tới centroid).
    Diễn giải: non-fraud rất giống nhau (gần trung bình chung), nên chỉ cần
    "đoán theo trung bình" (prior) đã gần đúng sẵn rồi — attack không có nhiều
    "chỗ" để cải thiện thêm. Fraud thì đa dạng/phân tán hơn nhiều, nên đoán mù
    theo trung bình sẽ sai nhiều, và gradient mạnh của fraud lại cho attack đủ
    tín hiệu để tái dựng tốt.

### 3.3. Một lỗi objective khác được phát hiện muộn hơn

Khi soát lại kỹ hơn, phát hiện objective "bounded Adam ladder" đang so sánh
toàn bộ trạng thái nổi (floating state) của Adam theo kiểu phẳng
(`cosine_magnitude`), trong đó các buffer chạy của BatchNorm (`running_var`)
có độ lớn gấp gần 3900 lần các tensor tham số thật — nghĩa là objective đang
gần như chỉ tối ưu để khớp buffer BatchNorm, chứ không phải khớp tham số mô
hình thật. Đã thêm objective mới `balanced_tensor` (cân bằng trọng số giữa các
loại tensor) và chạy lại. Kết quả cải thiện đôi chút (thắng prior 9/10 thay vì
7/10) nhưng thắng zero-update chỉ 6/10 → gate vẫn fail. Bài học: một lỗi
objective có thể làm kết quả prior "tốt giả tạo", nhưng sửa nó không đảo
ngược kết luận chính của Phase 3.

### 3.4. Kết luận Phase 3

- Attack **tái dựng rất tốt** phần fraud, **gần như không tái dựng được**
  phần non-fraud — đây là đặc tính thật của dữ liệu (focal loss khuếch đại
  gradient của lớp hiếm là fraud, khiến gradient tổng gần như chỉ phản ánh
  fraud).
- Attack "chính thức" (không có oracle nhãn) **chưa đạt ngưỡng thống kê**
  cần thiết để coi là đáng tin cậy ở quy mô full-client (chỉ thắng 7/10 cả 2
  đối chứng, p=0.172).
- **Trạng thái: COMPLETE_WITH_NEGATIVE_RESULT.** Chưa đủ điều kiện để đánh giá
  DNA Transform ở Phase 4 nếu dùng đúng baseline attack này ở quy mô lớn —
  cần một baseline nhỏ hơn, không dùng oracle, và tập trung vào fraud.

## 4. Phase 4 — Đánh giá DNA Transform có thực sự bảo vệ hay không

### 4.1. Khóa "luật chơi" trước khi code (`phase4_protocol.md`)

Trước khi viết bất kỳ dòng code nào, protocol định nghĩa rõ:

- **Threat model chính thức**: attacker biết kiến trúc mô hình, checkpoint
  trước khi client huấn luyện, cấu hình optimizer, và quan sát được update đã
  truyền đi. Attacker **không** được biết nhãn fraud/non-fraud riêng biệt của
  từng bản ghi (không oracle).
- **"Same scope"**: DNA chỉ được so sánh với baseline trong đúng cùng một
  phạm vi (cùng nguồn update, cùng số bước optimizer, cùng nhóm lớp, cùng
  checkpoint, cùng mức hiểu biết của attacker, cùng bộ đối chứng, cùng ngân
  sách thí nghiệm).
- **Gate A (attack có hiệu lực không)**: phải thắng **CẢ HAI** đối chứng
  (Prior và Zero-update), với delta trung bình âm, delta trung vị âm, và
  p<0.05 (kiểm định dấu một phía). Chỉ thắng 1/2 đối chứng = **FAIL**, không
  có khái niệm "pass một nửa".
- **Gate B (kiểm tra tính toàn vẹn)**: chạy lại (replay) phải khớp tuyệt đối,
  candidate lưu ra đọc lại phải tái tạo đúng objective, mọi seed phải được
  ghi log, không được âm thầm bỏ qua job lỗi.
- **Quy tắc dừng**: nếu thắng đa số nhóm nhưng vẫn thiếu p-value, hoặc chỉ cải
  thiện ở mức gộp nhưng fail khi tách theo lớp, hoặc attack thích nghi thất
  bại trong khi baseline cùng phạm vi cũng thất bại → **dừng lại để chẩn
  đoán**, không tăng số mẫu chỉ để "ráng có ý nghĩa thống kê".
- **Bộ đối chứng tối thiểu**: FL Baseline (không bảo vệ), DNA lossless, DNA
  Transform, random retention (giữ ngẫu nhiên một phần update), top-k
  retention (chỉ giữ các phần tử lớn nhất), clipping+noise (kiểu DP truyền
  thống).
- **3 mức độ hiểu biết của attacker về DNA (adaptive attacker)**:
  - Level 0: không biết/không dùng DNA gì cả.
  - Level 1: biết thuật toán DNA Transform nhưng **không biết** seed/số
    ngẫu nhiên cụ thể đã dùng.
  - Level 2: biết **cả** seed/số ngẫu nhiên cụ thể (trường hợp cực đoan có
    lợi cho attacker, dùng để đo giới hạn trên).

### 4.2. Xây dựng baseline attacker hợp lệ (không DNA) — chuỗi debug dài

Baseline ban đầu (không có DNA) ở quy mô nhỏ (4 dòng/1 bước Adam) vẫn **FAIL**
cả 2 đối chứng (Prior p=0.19, Zero p=0.07). Cần chẩn đoán tiếp:

- **Đo gradient contribution lại**: vẫn thấy fraud chi phối mạnh (~121–148
  lần) → không phải do thiếu tín hiệu.
- **Phát hiện "candidate-selection instability"**: so sánh objective (thứ
  attack dùng để tự chọn "kết quả tốt nhất") với sai số thật (fraud-MSE) —
  chỉ tương quan Pearson ~0.52. Nghĩa là objective **chọn nhầm** candidate ở
  4/12 nhóm — có candidate tốt hơn tồn tại nhưng attack không nhận ra.
- **Thử tăng số lần chạy lại (restart) từ 3 lên 10** cho 4 nhóm khó: tìm được
  candidate tốt hơn thật, nhưng objective **vẫn không chọn đúng nó** ở 3/4
  nhóm → không phải do thiếu số lần thử.
- **Phát hiện quan trọng — kiểm tra "tính hợp lý" (plausibility)**: candidate
  mà objective chọn (với "điểm khớp gradient" thấp nhất) lại chứa giá trị
  **phi vật lý**: số tiền giao dịch âm tới -1.080.477, số dư âm. Tức là
  objective "khớp gradient tốt" nhưng lại chọn một nghiệm hoàn toàn vô nghĩa
  về mặt dữ liệu thật.
- **Kiểm tra ràng buộc dữ liệu thật**: hai đặc trưng `balance_diff_*` (hiệu số
  dư trước/sau giao dịch) trên thực tế **luôn bằng đúng công thức tính từ 4
  cột số dư gốc** (residual = 0 tuyệt đối, kiểm tra trên 6.36 triệu dòng thật)
  — đây là một ràng buộc toán học cứng của dữ liệu, không phải "mẹo" đặc thù
  cho lớp fraud/non-fraud.
- **Sửa 1 — Hard reparameterization**: không cho attack tự do tối ưu 2 đặc
  trưng `balance_diff_*` nữa, mà **bắt buộc tính chúng đúng công thức** từ 4
  cột số dư đang được tối ưu.
- **Sửa 2 — Non-negative penalty**: phạt thêm khi amount/balance ra giá trị
  âm, với trọng số phạt λ (thử các giá trị 0, 0.001, 0.01, 0.1, 1, 10).
- **Sự cố quy trình #1 (tự phát hiện và tự sửa)**: nhận ra mình đang chọn λ
  trực tiếp trên chính 4 nhóm sẽ dùng để đánh giá cuối cùng → vi phạm nguyên
  tắc tách dev/eval. Đã sửa: tách riêng `development_gate_targets.pt` để chọn
  λ, chỉ sau đó mới đánh giá trên 12 nhóm chính thức.
- **Sự cố quy trình #2**: nhận ra 12 nhóm "chính thức" ban đầu cũng đã bị
  "nhìn thấy" trong quá trình chẩn đoán trước đó (rủi ro rò rỉ thông tin, dù
  vô tình). Đã tạo hẳn một bộ mục tiêu hoàn toàn mới
  (`fresh_final_targets.pt`, không trùng với bất kỳ target cũ nào) và chạy
  **một lần duy nhất** (single-shot), không được chỉnh sửa sau khi thấy kết
  quả.

**Kết quả Gate A cuối cùng (chạy một lần trên bộ target hoàn toàn mới):**

| Đối chứng | Thắng | p |
|---|---:|---:|
| Prior | 10/12 | 0.0193 |
| Zero-update | 12/12 | 0.0002 |

→ **PASS.** Gate B (kiểm tra toàn vẹn) cũng pass tuyệt đối: 240/240 artifact
đọc lại khớp đúng, sai số tối đa = 0.0.

### 4.3. Kiểm tra bản chất toán học của DNA Transform

Đọc trực tiếp code thật (`transform_defense.py`): với mỗi block dữ liệu
(kích thước 256), nếu biết seed/số ngẫu nhiên cụ thể, phép biến đổi là một
**phép tuyến tính** (nhân ma trận), có hạng đầy đủ (full-rank), số điều kiện
(condition number) chỉ ~1.19 — nghĩa là về mặt toán học, phép này **rất dễ
đảo ngược** nếu biết seed. Ngoài ra, `block_seed` lại phụ thuộc vào chính nội
dung của update gốc (rolling hash) — thứ mà attacker không biết trước — nên
không tồn tại "không gian seed nhỏ" để dò mò/brute-force.

### 4.4. Tấn công DNA ở từng mức độ hiểu biết

**Level 2 (biết chính xác seed) — "direct inversion":**

Chỉ cần đảo ngược trực tiếp phép tuyến tính của từng block (768 block trên 12
nhóm). Sai số khôi phục cực nhỏ: sai số tương đối trung bình ≈ 3.3×10⁻⁸ (chỉ
là sai số làm tròn số thực dấu phẩy động). **Kết luận: nếu bị lộ seed, DNA
Transform gần như trong suốt hoàn toàn — không có tác dụng bảo vệ gì.**

**Level 1 (biết thuật toán, KHÔNG biết seed) — quá trình xây công thức đúng
mất 3 lần sửa:**

1. Lần 1: chọn "candidate tốt nhất mỗi nhóm" bằng cách so với **update gốc
   thật** → đây là cách làm oracle (dùng thông tin mà attacker thật không có),
   **không hợp lệ**.
2. Lần 2: công thức tham chiếu lại tính ngược từ chính update đã quan sát
   (`M_r⁻¹ · update_quan_sát`) → tự tham chiếu chính nó (self-referential),
   không có ý nghĩa kiểm định thật.
3. Lần 3 (đúng): `argmin (r, c) của khoảng cách giữa M_r · delta(c) và
   update_quan_sát` — nghĩa là thử nhiều "kịch bản seed có thể" (r) và nhiều
   "dữ liệu giả" (c) cùng lúc, chọn ra cặp nào biến đổi ra gần giống nhất với
   update thật sự quan sát được — không dùng update gốc ở bất kỳ bước nào.

Sau khi có công thức đúng, còn phải:

- Thử lại với pool candidate cũ (tái sử dụng từ baseline) → kết quả gần sát
  mức "oracle" nhưng **không hợp lệ** vì chưa phải attack thật đầy đủ.
- Xây "full Level 1 adaptive attack" thật (tối ưu đồng thời cả kịch bản seed
  và dữ liệu giả cùng lúc): thử lần đầu với ngân sách nhỏ (2 lần chạy lại) →
  **FAIL** cả 2 đối chứng.
- Tăng ngân sách lên 8 lần chạy lại: **PASS** trên tập phát triển (7/8, cả
  2 đối chứng, p=0.035).
- Chạy một lần duy nhất trên tập mục tiêu hoàn toàn mới (12 nhóm,
  `level1_fresh_final_targets.pt`): **PASS** (Prior 10/12 p=0.019, Zero 11/12
  p=0.003).

**So sánh trung tâm của cả Phase 4 — DNA vs Raw update, cùng attacker, cùng
target:**

| Nhánh | vs Prior | vs Zero-update |
|---|---|---|
| Raw update (không có DNA) | 10/12, p=0.019 | 12/12, p=0.0002 |
| DNA Level 1 | 10/12, p=0.019 | 11/12, p=0.003 |

**DNA có khó tái dựng hơn raw không?** 6/12 nhóm — chỉ hơn một nửa, mean
delta +245.72, median delta chỉ +1.45, **p=0.61 — không có ý nghĩa thống kê**.
→ **Không có bằng chứng DNA bảo vệ tốt hơn việc hoàn toàn không bảo vệ gì.**

- Kiểm tra lại các nhóm có chênh lệch cực đoan nhất (outlier check): không
  tìm thấy lỗi kỹ thuật nào (không phải lỗi artifact, lỗi replay, hay chọn
  sai bước cuối) → giữ nguyên số liệu.
- Thử cấu hình DNA "mạnh hơn" (`stronger`: mix=0.12, keep=0.82, shrink=0.35)
  thay vì `conservative` (mặc định): **kết quả còn tệ hơn** — chỉ 4/12,
  p=0.93. Giả thuyết: biến đổi càng mạnh theo một công thức công khai thì
  càng dễ bị attacker "bù trừ ngược" khi biết đúng thuật toán.

### 4.5. So sánh với 4 phương án bảo vệ đơn giản khác (trả lời RQ1b)

| Phương án | Attack riêng của nó có vượt qua đối chứng không? | Có khó hơn raw không? |
|---|---|---|
| DNA Transform (conservative) | Có | Không (6/12, p=0.61) |
| DNA Transform (stronger) | Có | Không, còn tệ hơn (4/12, p=0.93) |
| Random retention (giữ ngẫu nhiên ~10% độ lớn) | Có | Không (7/12, p=0.39) |
| Clipping + nhiễu Gaussian (kiểu DP, tấn công trực tiếp) | Không (fail chính đối chứng của nó) | Không đánh giá được |
| Clipping + nhiễu Gaussian (tấn công kiểu Monte Carlo, trên bộ target hợp lệ) | Có | Không (4/12, p=0.93) |
| Top-k retention (chỉ giữ phần tử lớn nhất) | Không ổn định — pass ở tập phát triển nhưng fail ở tập mới | Chưa kết luận được |

**Chi tiết đáng chú ý:**

- **Clipping+noise, lần thử đầu tiên thất bại** vì attack cố đoán đúng **một**
  hiện thực nhiễu cụ thể — vô nghĩa với nhiễu ngẫu nhiên độc lập (mỗi lần
  nhiễu ra một giá trị khác nhau, không có "đáp án đúng" cố định để đoán).
  → Sửa bằng cách đổi sang tối ưu theo **kỳ vọng Monte Carlo** (mô phỏng
  N=10/50/100 mẫu nhiễu giả định rồi lấy trung bình) — sau khi sửa, attack
  pass được ở tập phát triển.
- **Khi áp dụng lên tập mục tiêu mới đầu tiên**: bất ngờ **baseline raw** (đối
  chứng, không phải DNA) lại **fail** trên chính tập mới đó (9/12 cả 2 đối
  chứng, p=0.073) — điều tra kỹ cho thấy không phải lỗi tạo tập dữ liệu
  (checksum, không trùng lặp đều đúng), mà là attack raw ở lần chọn candidate
  này không ổn định trên đúng draw ngẫu nhiên này (oracle-candidate vẫn pass
  11/12, chứng tỏ vấn đề nằm ở khâu tự-chọn-candidate, không phải bản chất dữ
  liệu không thể tái dựng). → Tạo thêm **một tập thứ hai** (fresh2) để kiểm
  tra độ ổn định — lần này cả raw và clipping/noise đều pass, và so sánh hợp
  lệ cho kết quả: **clipping/noise không khó hơn raw có ý nghĩa** (4/12,
  p=0.93).
- **Top-k, thử 3 lần**: (1) top-k "cứng" (giữ đúng top-k phần tử) không khả vi
  tại ranh giới, khó tối ưu bằng gradient. (2) Đổi sang "soft top-k" (làm mềm
  bằng hàm sigmoid có nhiệt độ điều chỉnh), thử với 8 lần chạy lại: pass được
  đối chứng zero nhưng fail đối chứng prior. (3) Tăng lên 16 lần chạy lại:
  pass cả 2 đối chứng **trên tập phát triển**, nhưng khi áp dụng đúng cấu
  hình đó lên **tập mục tiêu mới** (chưa từng thấy trước) thì **fail cả 2**
  đối chứng (7/12, p=0.39). → Top-k **không tổng quát hóa được** từ tập phát
  triển sang tập mới, nên **không thể kết luận** nó có bảo vệ hay không —
  vấn đề nằm ở chính công cụ đo (attacker chưa đủ tin cậy), không phải bằng
  chứng ủng hộ top-k.

### 4.6. Thử nới lỏng hiểu biết của attacker — attacker không biết kiến trúc mô hình

**Ý tưởng:** Nếu attacker biết ít hơn (không biết chính xác kiến trúc mô hình:
số lớp, số neuron mỗi lớp), liệu DNA có tỏ ra bảo vệ tốt hơn không?

**Vấn đề kỹ thuật phải giải quyết trước:** nếu cho attacker nhìn thẳng
`state_dict` (tên tensor, shape) để "đoán" kiến trúc, thì kiến trúc đã lộ sẵn
qua chính tên/shape của tensor rồi — không có ý nghĩa gì để thử nghiệm.
→ Thiết kế lại kênh quan sát: chiếu (project) toàn bộ update về một không
gian chung cố định 4096 chiều bằng một ma trận chiếu công khai
(`signed_hadamard_jl`), và mỗi kiến trúc ứng viên được khởi tạo độc lập (không
copy từ checkpoint thật) — để đảm bảo kiến trúc thật không bị lộ qua bất kỳ
kênh phụ nào.

- **Kiểm tra ma trận chiếu có làm lộ thông tin không**: lần đầu dùng
  `signed_sparse_jl` bị phát hiện làm lộ kiến trúc qua độ thưa/độ lớn khác
  nhau giữa các kiến trúc → đổi sang `signed_hadamard_jl` kèm chuẩn hóa L2 →
  kiểm tra lại thấy sạch (không còn khác biệt giữa các kiến trúc).

**Mức B (lưới rộng, 6 kiến trúc khác nhau × 4 learning rate × 4 loss = 96 tổ
hợp):** chỉ cần kiểm tra sơ bộ xem kiến trúc thật có lọt top-3 (theo objective
đã điều chỉnh theo độ phức tạp) hay không:

- Lần đầu (ngân sách nhỏ): kiến trúc thật xếp hạng 5/6 — **FAIL**.
- Tăng ngân sách một lần duy nhất (theo đúng luật protocol cho phép tăng 1
  lần): xếp hạng 4/6 — **vẫn FAIL**.
- → Dừng theo đúng luật đã khóa trước, **không chạy hết 96 tổ hợp**.

**Mức B' (lưới hẹp hơn, chỉ dao động ±20% quanh đúng kiến trúc thật quanh 3
lớp 128/64/32, 27 tổ hợp):**

- Kiến trúc thật xếp hạng **23/27** — gần như bét bảng!
- Kiểm tra loại trừ nguyên nhân "phạt độ phức tạp" (complexity penalty): thứ
  hạng trước và sau khi áp dụng phạt độ phức tạp **giống hệt nhau** (23/27 cả
  hai), và mức phạt chỉ chênh ~0.12% → **không phải do cách tính phạt**.

**Diễn giải:** Đây **không phải** là lỗi của riêng DNA Transform. Đây là giới
hạn của chính phương pháp "khớp gradient" khi phải đồng thời tối ưu cả kiến
trúc lẫn dữ liệu giả: kiến trúc nhỏ hơn có ít bậc tự do hơn, nên dễ "khớp giả"
với gần như bất kỳ tín hiệu nào — kể cả tín hiệu sai — với loss thấp hơn cả
kiến trúc đúng kích thước thật. Vì bước sàng lọc sơ bộ đã fail (ngay cả khi
chưa có bất kỳ cơ chế bảo vệ nào tham gia), nhánh này **không dùng được** để
xếp hạng DNA hay bất kỳ phương án bảo vệ nào khác. Dừng lại ở đây, không chạy
tiếp bước phát triển/đánh giá đầy đủ cho nhánh "attacker yếu hơn" này.

### 4.7. Kết luận Phase 4

**Phạm vi đã kiểm chứng:** 4 dòng dữ liệu/nhóm, 1 giao dịch fraud/nhóm, 1 bước
Adam cục bộ, attacker biết kiến trúc/tiền xử lý/loss, không có oracle nhãn.
Đây chính là phần "kiểm định chặt lại RQ1" (đối chiếu DNA với đối chứng, kể cả
một đối chứng kiểu DP là clipping/noise) — nhưng đo bằng MSE/relative-L2 thay
vì PSNR/SSIM, và ở update FedAvg thật thay vì gradient một mẫu như bảng
PSNR/SSIM 50-round trong `README.md`. RQ2 (F1/AUC-ROC) và RQ3 (chi phí tính
toán/băng thông) của đề tài **không** được đo lại ở đúng quy mô 4-record này;
chúng vẫn dựa vào bảng 50-round có sẵn trong `README.md` ("Accepted 50-Round
Utility Results" cho RQ2, "Experiment 2: Client Scalability Test" cho một
phần RQ3), xem thêm "Việc còn dang dở" trong `PROJECT.md`.

- **RQ1a — DNA có giảm rủi ro tái dựng không?** **Không có bằng chứng.** Ở
  mức attacker biết chính xác seed, DNA gần như trong suốt (sai số ~10⁻⁸). Ở
  mức attacker chỉ biết thuật toán không biết seed, DNA không khó hơn raw có
  ý nghĩa thống kê (6/12, p=0.61); bản mạnh hơn còn tệ hơn (4/12, p=0.93).
- **RQ1b — lợi ích có đến từ chính DNA không?** Không có phương án nào trong 4
  phương án đơn giản đối chứng (DNA×2, random retention, clipping/noise MC —
  clipping/noise chính là cách xấp xỉ DP-style noise ở quy mô nhỏ này) cho
  thấy khó hơn raw có ý nghĩa thống kê ở cùng cường độ làm méo update. Top-k
  không thể kết luận vì chính công cụ đo (attacker) không ổn định giữa tập
  phát triển và tập mới.
- **RQ1c — có ổn định khi mở rộng thực tế không?** **Chưa làm.** Theo đúng
  luật quyết định đã khóa trước, khi không phát hiện lợi ích ở quy mô nhỏ
  nhất đã kiểm chứng kỹ nhất, bước tiếp theo không phải là mở rộng lên quy mô
  lớn hơn (nhiều client hơn, full-client, nhiều round hơn) mà là **dừng lại
  và thiết kế lại DNA Transform**, hoặc chấp nhận đây là kết quả âm tính.
- **Giới hạn phương pháp phát hiện thêm:** khi thử làm yếu hiểu biết của
  attacker về kiến trúc mô hình (Level B/B'), chính công cụ đo (gradient
  matching) đã hỏng trước khi kịp đánh giá bất kỳ cơ chế bảo vệ nào — đây là
  giới hạn của công cụ, cần ghi nhận riêng, không lẫn vào kết luận RQ1a/RQ1b.

**Trả lời RQ1 (câu hỏi nghiên cứu chuẩn) trong phạm vi đã kiểm chứng ở Phase
1–4:** Ở mức đo chặt hơn PSNR/SSIM một lần chạy (có đối chứng thống kê, dùng
đúng update FedAvg), DNA encoding/Transform **không cho thấy bảo vệ tốt hơn
Differential-Privacy-style clipping/noise, và cũng không tốt hơn việc hoàn
toàn không bảo vệ.** Kết quả PSNR/SSIM ở bảng 50-round trong `README.md`
(DNA Transform nhỉnh hơn DP một chút ở phía "khó tái dựng hơn") cần được đọc
cùng với phát hiện này: đó là kết quả ở một attack gradient-đơn-mẫu chưa qua
kiểm định thống kê chặt (không đối chứng Prior/Zero-update, không p-value,
không tập dev/eval tách biệt), nên **không nên coi là câu trả lời cuối cùng**
cho RQ1 — câu trả lời đáng tin cậy hơn, dựa trên toàn bộ Phase 1–4, là
**CHƯA có bằng chứng** DNA bảo vệ tốt hơn DP hay tốt hơn không bảo vệ, trong
phạm vi hẹp đã kiểm chứng.

**Quyết định cuối:** `STOP_OR_REDESIGN_DNA_FOR_THIS_SCOPE` — dừng mở rộng cấu
hình DNA hiện tại, không ngoại suy sang full-client FedAvg. Xem chi tiết đầy
đủ (số liệu, bảng, artifact path) tại phần "Conclusion" trong
[`phase4_report.md`](phase4_report.md).

## 5. Bài học phương pháp luận xuyên suốt cả 4 phase

Những nguyên tắc này được áp dụng lặp đi lặp lại và là lý do các kết luận
trên đáng tin cậy hơn một lần chạy đơn giản:

1. **Không bao giờ tin kết quả gộp (aggregate) mà không tách theo lớp** — phát
   hiện fraud/non-fraud trái ngược nhau ở Phase 3 chỉ lộ ra khi tách riêng.
2. **Không dùng dữ liệu đã "nhìn thấy" trong lúc chẩn đoán để làm bộ đánh giá
   cuối cùng** — phải tạo tập mục tiêu hoàn toàn mới (source-disjoint,
   fresh) trước khi tuyên bố kết quả chính thức, và chỉ chạy **một lần**
   (single-shot) trên tập đó.
3. **Không dùng oracle/ground-truth để chọn candidate của attacker** — mọi
   phiên bản có oracle chỉ được gọi là "chẩn đoán", không phải attack chính
   thức.
4. **Khi kết quả bất thường (attack thua cả đối chứng), phải tìm nguyên nhân
   gốc rễ trước khi tăng ngân sách** — ví dụ vụ "objective bị BatchNorm
   running_var chi phối", vụ "balance_diff không được ràng buộc đúng công
   thức", vụ "candidate objective-best chứa giá trị âm phi vật lý".
5. **Gate (tiêu chí đạt/không đạt) phải khóa trước khi chạy, không được nới
   sau khi thấy kết quả** — kể cả khi kết quả "gần đạt" (ví dụ p=0.07), vẫn
   tính là FAIL.
6. **Chấp nhận kết luận âm tính** — nhiều nhánh nghiên cứu (Phase 3 full-client,
   Level B/B', top-k) đều dừng lại ở trạng thái "không đạt" hoặc "chưa kết
   luận được", và được báo cáo đúng như vậy thay vì bị ép thành kết quả tích
   cực.
7. **Một lần tăng ngân sách thử nghiệm là được phép nếu đã định trước, nhưng
   không được lặp lại nhiều lần chỉ để "ráng ra ý nghĩa thống kê"** — ví dụ
   Level B chỉ được tăng ngân sách đúng 1 lần rồi dừng theo luật đã khóa.
