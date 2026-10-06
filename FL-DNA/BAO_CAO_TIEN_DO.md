# Báo cáo tiến độ — FL-DNA (bản dễ hiểu, không cần kiến thức nền)

## 1. Đang nghiên cứu cái gì?

Tưởng tượng nhiều ngân hàng muốn cùng nhau xây một hệ thống AI phát hiện gian
lận, nhưng không ai muốn chia sẻ dữ liệu giao dịch của khách hàng mình cho
ngân hàng khác hay cho một bên thứ ba. Giải pháp là: mỗi ngân hàng tự huấn
luyện AI trên dữ liệu của mình, rồi chỉ gửi đi "những gì AI đã học được"
(gọi là *update* — có thể hiểu như một bản tóm tắt các thay đổi trong bộ não
AI) cho một máy chủ trung tâm để gộp lại thành một AI chung, mạnh hơn. Đây
gọi là **Federated Learning (FL)**.

Nghe có vẻ an toàn vì dữ liệu gốc không đi đâu cả. Nhưng câu hỏi đặt ra là:
**liệu chỉ nhìn vào "bản tóm tắt" (update) đó, có ai đó có thể đoán ngược lại
được thông tin giao dịch gốc không?** Nếu có, thì FL không an toàn như tưởng.

Nhóm nghiên cứu đề xuất một cách "xáo trộn" bản tóm tắt đó trước khi gửi đi,
lấy cảm hứng từ cách mã hóa thông tin của DNA sinh học, gọi là **DNA
Transform**. Câu hỏi lớn của cả đề tài là: **cách xáo trộn này có thực sự làm
cho việc đoán ngược khó hơn không, mà không làm AI kém thông minh đi, và
không tốn quá nhiều tài nguyên máy tính?**

## 2. Ba câu hỏi cần trả lời

| # | Câu hỏi (nói theo cách đơn giản) |
|---|---|
| **1** | DNA Transform có chống lại việc bị "đoán ngược dữ liệu" tốt hơn cách bảo vệ truyền thống (thêm nhiễu ngẫu nhiên, gọi là Differential Privacy) không? |
| **2** | Sau khi áp dụng DNA Transform, AI có còn phát hiện gian lận chính xác như trước không, hay bị giảm chất lượng? |
| **3** | Việc áp dụng DNA Transform có tốn quá nhiều thời gian xử lý và dung lượng truyền tải không, để có thể dùng thật ngoài đời không? |

## 3. Đã làm được những gì

### 3.1. Một bộ thử nghiệm lớn đã có từ trước

Nhóm đã chạy một hệ thống FL đầy đủ (3 "ngân hàng" giả lập, chạy 50 vòng học,
dùng 500.000 giao dịch) và so sánh nhiều cách bảo vệ khác nhau. Từ đó đã có
sẵn câu trả lời ban đầu cho cả 3 câu hỏi:

- **Câu hỏi 2 (AI có còn thông minh không?)**: Có bảng đo độ chính xác đầy đủ.
  DNA Transform (bản "an toàn vừa phải") giữ được độ chính xác khá tốt, chỉ
  giảm nhẹ so với không bảo vệ gì. Ngược lại, nếu thêm nhiễu ngẫu nhiên quá
  mạnh (cách truyền thống), AI gần như mất khả năng phát hiện gian lận.
- **Câu hỏi 3 (có tốn tài nguyên không?)**: Đã đo được: DNA Transform tốn
  thêm khoảng 0.15–0.6 giây xử lý mỗi vòng học tùy số lượng "ngân hàng" tham
  gia — mức chi phí này khá nhỏ.
- **Câu hỏi 1 (có chống đoán ngược tốt hơn không?)**: Đã có một phép thử ban
  đầu, kết quả cho thấy DNA Transform có vẻ khó bị đoán ngược hơn một chút so
  với không bảo vệ gì. **Nhưng phép thử này rất đơn giản và sơ sài** — giống
  như chỉ thử đoán mò một lần rồi kết luận luôn, chưa kiểm tra xem kết quả đó
  có phải do may mắn hay không. Vì vậy nhóm quyết định làm lại phép thử này
  một cách nghiêm túc và chặt chẽ hơn — đó chính là công việc của 4 giai đoạn
  (Phase 1 đến Phase 4) mô tả dưới đây.

### 3.2. Bốn giai đoạn làm lại câu hỏi 1 cho chắc chắn

Mục tiêu của 4 giai đoạn này là xây một "người thử đoán ngược" (gọi là
**attacker** — kẻ tấn công giả lập, do chính nhóm nghiên cứu tạo ra để kiểm
tra độ an toàn) thật sự giỏi và đáng tin cậy, rồi dùng nó để kiểm tra DNA
Transform một cách công bằng.

**Giai đoạn 1 — Kiểm tra máy móc có chạy đúng không.**
Trước khi tin bất kỳ kết quả nào, phải chắc chắn mọi thứ hoạt động đúng: dữ
liệu được xử lý đúng, AI học đúng, cách mã hóa DNA không làm mất dữ liệu khi
mã hóa/giải mã. Kết quả: mọi thứ chạy đúng, sẵn sàng cho bước tiếp theo.

**Giai đoạn 2 — Kiểm tra xem "kẻ tấn công thử" có thực sự đoán được gì không.**
Thử cho attacker đoán ngược 10 giao dịch, so sánh với việc "đoán mù" (chỉ
đoán theo giá trị trung bình chung, không cần thông tin gì). Kết quả: attacker
đoán đúng hơn đoán mù ở 8/10 trường hợp — nghĩa là nó có "năng lực" thật, chứ
không phải chỉ đoán may rủi. Nhưng ở bước này attacker mới chỉ thử đoán từ
một mẩu thông tin rất nhỏ (một phép tính đơn của một giao dịch), chưa phải là
loại thông tin thật mà một ngân hàng thực sự gửi đi trong FL.

**Giai đoạn 3 — Cho attacker thử với đúng loại thông tin thật (sau khi ngân
hàng huấn luyện xong, gộp lại thành một "bản tóm tắt" hoàn chỉnh).**
Đây là bước quan trọng và cũng là nơi có phát hiện lớn nhất. Khi thử ở quy mô
đầy đủ (một ngân hàng thật với hàng chục nghìn giao dịch), attacker **thất
bại hoàn toàn** — tệ đến mức thua cả việc "giả vờ không có thông tin gì để
đoán". Sau khi tìm hiểu kỹ, nhóm phát hiện ra nguyên nhân: vì giao dịch gian
lận rất hiếm (chưa tới 0.2% tổng số), nên "bản tóm tắt" mà AI gửi đi gần như
chỉ phản ánh những giao dịch gian lận đó, còn thông tin về các giao dịch bình
thường gần như "biến mất", không còn dấu vết rõ để đoán ngược. Kết quả: **có
thể đoán khá tốt phần giao dịch gian lận, nhưng gần như không đoán được gì về
giao dịch bình thường.**

**Giai đoạn 4 — Thu nhỏ phạm vi lại cho vừa sức, rồi mới so sánh DNA
Transform với các cách khác.**
Rút kinh nghiệm từ giai đoạn 3, nhóm thu nhỏ bài toán lại: chỉ thử với một
nhóm rất nhỏ dữ liệu (4 giao dịch, trong đó có 1 giao dịch gian lận), và chỉ
sau đúng 1 bước học của ngân hàng. Trong phạm vi nhỏ này, cuối cùng cũng xây
được một attacker đủ mạnh và đáng tin cậy (đoán đúng hơn hẳn cách đoán mù,
một cách chắc chắn chứ không phải may rủi). Dùng attacker này để kiểm tra:

- DNA Transform có làm việc đoán ngược khó hơn so với **hoàn toàn không bảo
  vệ gì** không? → **Không thấy sự khác biệt đáng tin cậy.**
- DNA Transform có làm việc đoán ngược khó hơn so với **cách thêm nhiễu ngẫu
  nhiên kiểu truyền thống (Differential Privacy)** không? → **Cũng không
  thấy sự khác biệt đáng tin cậy.**
- Nhóm còn thử thêm vài cách "xáo trộn" đơn giản khác để so sánh (ví dụ: chỉ
  giữ lại ngẫu nhiên một phần thông tin, hoặc chỉ giữ phần thông tin lớn
  nhất). Không cách nào trong số này chứng minh được là bảo vệ tốt hơn hẳn
  việc không làm gì cả.

## 4. Đang gặp vấn đề gì?

### 4.1. Kết quả cũ và kết quả mới đang "vênh" nhau

Phép thử ban đầu (mục 3.1) nói DNA Transform bảo vệ tốt hơn nhiễu ngẫu nhiên
một chút. Phép thử mới, chặt chẽ hơn (4 giai đoạn) lại nói không có khác biệt
gì đáng kể. Hai kết quả này **chưa được đối chiếu và giải thích rõ ràng** với
nhau — cần làm rõ kết quả nào đáng tin hơn trước khi đưa ra kết luận chính
thức cho câu hỏi 1.

### 4.2. "Người thử đoán ngược" chỉ giỏi khi bài toán rất nhỏ — vấn đề cốt lõi

Đây là vấn đề quan trọng nhất. Attacker chỉ chứng minh được năng lực thật sự
(đoán đúng hơn đoán mù một cách chắc chắn) khi bài toán bị thu nhỏ xuống còn
**4 giao dịch, 1 bước học** — nhỏ hơn rất nhiều so với một ngân hàng thật (có
thể tới hàng chục nghìn hoặc hàng trăm nghìn giao dịch mỗi lần gửi update).

Ở quy mô càng gần với thực tế, attacker càng đoán tệ đi rất nhanh — thậm chí
ở quy mô đầy đủ (giai đoạn 3), nó thua cả việc "không có thông tin gì để
đoán". Ví dụ cụ thể:

| Cách đoán | Mức độ sai (càng cao càng an toàn) |
|---|---:|
| Đoán mù (không dùng thông tin gì) | 117 |
| Attacker dùng thông tin thật để đoán | 129 |
| Giả vờ không có thông tin gì (đối chứng) | 129 |

(Attacker "dùng thông tin thật" lại sai gần bằng — thậm chí hơi tệ hơn — so
với việc "không dùng thông tin gì", nghĩa là nó gần như không khai thác được
gì thêm ở quy mô này.)

**Điều này có nghĩa là gì?** Mọi kết luận từ giai đoạn 4 (kể cả kết luận "DNA
Transform không bảo vệ tốt hơn") **chỉ chắc chắn đúng ở quy mô rất nhỏ đã thử
nghiệm**. Chưa có đủ cơ sở để khẳng định điều tương tự sẽ đúng khi áp dụng ở
quy mô ngân hàng thật, với dữ liệu đầy đủ.

### 4.3. Một vài cách bảo vệ khác vẫn chưa đánh giá được, không phải vì chúng an toàn

Khi thử so sánh DNA với một cách bảo vệ khác (chỉ giữ lại các thông tin "lớn
nhất" trong update, bỏ phần nhỏ), attacker dùng để kiểm tra cách này lại
không ổn định: nó vượt qua được bài kiểm tra trên bộ dữ liệu dùng để luyện
tập, nhưng khi đưa sang một bộ dữ liệu hoàn toàn mới thì lại thất bại. Nói
cách khác: **chưa có công cụ đủ tốt để đánh giá cách bảo vệ này**, chứ không
phải là bằng chứng nó thực sự an toàn.

### 4.4. Thử với "kẻ tấn công biết ít thông tin hơn" nhưng chưa thực hiện được

Có một kịch bản thực tế hơn: nếu kẻ tấn công không biết chính xác cấu trúc
bên trong của AI (bao nhiêu lớp, bao nhiêu "nơ-ron" mỗi lớp) thì sao? Nhóm đã
thử dựng kịch bản này, nhưng ngay từ bước kiểm tra sơ bộ, công cụ đo đã cho
kết quả vô lý: khi cho nhiều "cấu trúc AI khả nghi" cạnh tranh nhau xem cái
nào giống thật nhất, **cấu trúc đúng lại bị xếp gần cuối bảng** — tức là công
cụ đo tự nó không đáng tin ở kịch bản này (không liên quan gì đến DNA
Transform). Vì vậy nhánh nghiên cứu này phải tạm dừng, chưa đánh giá được gì
thêm.

### 4.5. Câu hỏi 2 và câu hỏi 3 chưa được đo lại ở đúng quy mô nhỏ mà 4 giai đoạn dùng

Câu trả lời hiện có cho "AI có còn chính xác không" và "có tốn tài nguyên
không" đều đến từ bộ thử nghiệm quy mô lớn (mục 3.1), còn phần đánh giá bảo
mật đáng tin cậy nhất lại ở quy mô rất nhỏ (mục 3.2). Ba câu hỏi hiện đang
được trả lời bằng ba bộ thử nghiệm khác nhau, chưa cùng một "sân chơi" — cần
làm lại để cả ba câu hỏi được đo trên cùng một cấu hình, thì kết luận cuối
cùng mới thực sự thuyết phục.

## 5. Danh sách những cách đã thử (để không thử lại những hướng đã biết không hiệu quả)

### 5.1. Các cách "xáo trộn" update đã đưa vào so sánh

| Cách bảo vệ | Ý tưởng | Đã biết được gì |
|---|---|---|
| Không bảo vệ gì | Gửi thẳng update gốc | Dùng làm mốc so sánh |
| Mã hóa DNA không mất dữ liệu | Mã hóa rồi giải mã lại đúng y hệt | Chỉ là cách "đóng gói" để truyền đi, không hề làm khó việc đoán ngược |
| DNA Transform (mức bảo vệ vừa phải — đang dùng chính) | Xáo trộn + làm mờ một phần + trộn lẫn dữ liệu theo quy tắc lấy cảm hứng từ DNA | Chưa tìm được bằng chứng nó khó bị đoán ngược hơn không bảo vệ gì, ở quy mô nhỏ đã thử |
| DNA Transform (mức bảo vệ mạnh hơn) | Xáo trộn/làm mờ mạnh tay hơn | Kết quả còn **tệ hơn** mức vừa phải, không giúp ích thêm |
| Thêm nhiễu ngẫu nhiên (Differential Privacy) | Giới hạn độ lớn rồi cộng nhiễu ngẫu nhiên | Nếu nhiễu mạnh thì AI gần như mất khả năng phát hiện gian lận |
| Che giấu bằng "mặt nạ" giữa các ngân hàng (Secure Aggregation) | Mỗi ngân hàng cộng thêm một số ngẫu nhiên bí mật, các số này tự triệt tiêu khi cộng dồn | Server chỉ thấy tổng gộp, không thấy từng ngân hàng riêng lẻ; giữ AI chính xác tốt |
| Kết hợp DNA Transform + mặt nạ | Dùng cả hai cùng lúc | Về lý thuyết là phương án an toàn nhất, nhưng chưa được kiểm tra bằng attacker riêng |
| Giữ lại ngẫu nhiên một phần thông tin | Bỏ ngẫu nhiên phần còn lại về 0 | Không chứng minh được khó bị đoán ngược hơn |
| Chỉ giữ phần thông tin "lớn nhất" | Bỏ các phần nhỏ về 0 | Công cụ kiểm tra chưa đủ tin cậy, chưa kết luận được |

### 5.2. Các phiên bản "kẻ tấn công thử" đã xây dựng qua từng giai đoạn

| Phiên bản | Kết quả |
|---|---|
| Đoán từ một mẩu thông tin nhỏ (giai đoạn 1–2) | Đoán đúng hơn đoán mù ở 8/10 trường hợp thử |
| Đoán từ "bản tóm tắt" đầy đủ, quy mô một ngân hàng thật (giai đoạn 3) | **Thất bại**, thua cả việc giả vờ không có thông tin gì |
| Sửa vài lỗi tính toán kỹ thuật trong công cụ đoán | Có cải thiện nhẹ nhưng vẫn không đủ để tin cậy ở quy mô lớn |
| Thử thêm vài kiểu "kiến trúc đoán" khác | Không cải thiện, vẫn thất bại theo cùng kiểu cũ |
| Cho công cụ đoán biết trước đâu là gian lận, đâu không (chỉ để tìm hiểu, không phải cách làm chính thức) | Đoán rất tốt phần gian lận, vẫn đoán kém phần bình thường — xác nhận đây là đặc điểm của dữ liệu, không phải lỗi công cụ |
| Thu nhỏ bài toán xuống 4 giao dịch/1 bước học, sửa lại vài quy tắc tính toán cho hợp lý hơn | Cuối cùng đạt được: đoán đúng hơn hẳn đoán mù một cách chắc chắn (không phải may rủi) |
| Kẻ tấn công biết cách DNA Transform hoạt động nhưng không biết "chìa khóa" bí mật cụ thể | Vẫn đoán được kha khá, nhưng không khó hơn việc tấn công update không được bảo vệ |
| Kẻ tấn công biết cả "chìa khóa" bí mật | Gần như đoán lại được y hệt dữ liệu trước khi bị DNA Transform xáo trộn — tức là nếu lộ chìa khóa, DNA Transform gần như vô dụng |
| Kẻ tấn công không biết cấu trúc bên trong của AI | Ngay từ bước kiểm tra sơ bộ đã cho kết quả vô lý — công cụ này chưa dùng được |

### 5.3. Các bước kiểm tra/tìm nguyên nhân đã làm khi kết quả bất thường

- Rà lại toàn bộ công thức tính toán bên trong (đặc biệt là một phần liên
  quan tới cách AI "chuẩn hóa" dữ liệu nội bộ) — tìm và sửa được 1 lỗi thật,
  nhưng sau khi sửa vấn đề chính vẫn còn, chứng tỏ đó không phải nguyên nhân
  gốc.
- Đo xem "sức nặng thông tin" của giao dịch gian lận so với giao dịch bình
  thường trong bản tóm tắt gửi đi — phát hiện: gian lận chiếm phần áp đảo,
  đây chính là lý do đoán được gian lận nhưng không đoán được giao dịch bình
  thường.
- Đo xem các giao dịch gian lận có "giống nhau" hay khác nhau nhiều — phát
  hiện: gian lận rất đa dạng, khác nhau nhiều, còn giao dịch bình thường khá
  giống nhau — nên "đoán mù theo trung bình" đã gần đúng sẵn với giao dịch
  bình thường rồi, không cần tấn công cũng đoán gần đúng.
- Kiểm tra xem "cách chọn đáp án tốt nhất" của công cụ đoán có tự chọn nhầm
  không — phát hiện: đúng là có chọn nhầm ở một số trường hợp, dù có đáp án
  tốt hơn tồn tại nhưng công cụ không nhận ra.
- Kiểm tra xem đáp án mà công cụ chọn có "hợp lý" về mặt thực tế không (ví dụ
  số tiền giao dịch có bị âm không) — phát hiện: có, một số đáp án được chọn
  chứa số tiền âm vô lý dù về mặt tính toán trông có vẻ "khớp tốt".
- Kiểm tra các quy tắc bắt buộc thật của dữ liệu (ví dụ: số dư sau giao dịch
  phải khớp đúng công thức với số dư trước đó) — xác nhận đây là quy tắc
  tuyệt đối, cần bắt công cụ đoán phải tuân theo thay vì để nó tự do đoán sai
  quy tắc.
- Thử để công cụ đoán "suy nghĩ" nhiều lần hơn (nhiều lần thử lại) xem có
  giúp tự sửa được lỗi chọn nhầm không — với hầu hết trường hợp là không tự
  sửa được, phải sửa trực tiếp cách công cụ hoạt động.
- Kiểm tra kỹ các trường hợp có kết quả bất thường nhất, xem có phải do lỗi
  lưu trữ/xử lý dữ liệu không — không tìm thấy lỗi, giữ nguyên kết quả.
- Khi công cụ đoán bất ngờ thất bại trên một bộ dữ liệu thử nghiệm mới, kiểm
  tra kỹ xem bộ dữ liệu đó có vấn đề gì không — không có vấn đề gì với dữ
  liệu, chỉ là bản thân công cụ đoán không ổn định trên đúng lần thử đó, nên
  phải thử lại với một bộ dữ liệu khác để chắc chắn hơn.
- Kiểm tra xem cách "che giấu" cấu trúc AI khỏi kẻ tấn công (dùng trong mục
  4.4) có vô tình để lộ thông tin qua đường khác không — phát hiện có, đã
  phải đổi cách che giấu khác để không bị lộ.
- Kiểm tra xem việc "phạt" các cấu trúc AI phức tạp hơn có phải là lý do
  khiến cấu trúc đúng bị xếp hạng thấp không (mục 4.4) — không phải, đây
  không phải nguyên nhân.

### 5.4. Các thử nghiệm khác ở quy mô đầy đủ (ngoài phạm vi 4 giai đoạn)

- Thử tăng dần mức độ "xáo trộn" của DNA Transform, xem độ chính xác AI và
  mức độ khó đoán ngược thay đổi thế nào.
- Thử với nhiều "ngân hàng" tham gia hơn (3, 5, 10 ngân hàng), đo độ chính
  xác và thời gian xử lý.
- Vẽ biểu đồ đánh đổi giữa độ chính xác AI và mức độ khó bị đoán ngược cho
  tất cả các cách bảo vệ.
- Kiểm tra việc đổi "chìa khóa" bí mật của DNA Transform mỗi lần chạy (thay
  vì dùng một chìa khóa cố định) có hoạt động đúng và có thể lặp lại khi cần
  không — xác nhận hoạt động đúng.
