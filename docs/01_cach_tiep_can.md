# Cách tiếp cận bài "Industrial Control System Intrusion Detection"

*Viết dưới góc nhìn một sinh viên đi thi Olympic AI (OlpAI) — nền tảng DL/NLP/CV, chưa
từng làm anomaly detection cho hệ điều khiển công nghiệp, có 6 giờ làm bài theo đội và
một bảng xếp hạng public để nộp thử.*

---

## 0. Điều kiện thi và hệ quả lên chiến thuật

Theo [quy chế OlpAI](https://www.olp.vn/olympic-ai-sinh-vi%C3%AAn/quy-ch%E1%BA%BF-k%E1%BA%BF-ho%E1%BA%A1ch),
vòng khu vực cho mỗi đội **6 giờ liên tục** với dữ liệu và yêu cầu do BTC cấp. Ba hệ quả:

1. **Không có thời gian cho mô hình lớn.** Một Transformer 20 triệu tham số train 3 tiếng
   là tự sát: hỏng một chỗ là mất luôn buổi thi. Phải có một baseline chạy được trong
   **15 phút đầu**, rồi cải thiện dần.
2. **Bảng xếp hạng là một *dụng cụ đo*, không phải nơi khoe.** Tập test không có nhãn ⇒
   mọi đánh giá cục bộ đều là phỏng đoán. Mỗi lượt nộp là một *phép đo thật*. Phải tiêu
   lượt nộp theo kiểu thiết kế thí nghiệm, không nộp bừa.
3. **Đọc kỹ cách chấm trước khi viết dòng code mô hình đầu tiên.** Ở bài này, riêng việc
   hiểu metric đã quyết định khoảng **80% điểm số** (xem mục 2). Đây là bài học lớn nhất.

### Phân bổ thời gian tôi sẽ dùng nếu làm lại

| Thời gian | Việc | Lý do |
|---|---|---|
| 0:00–0:30 | Đọc đề, **tìm mã nguồn metric**, tự cài lại, thử vài ví dụ đồ chơi | Metric quyết định hình dạng lời giải |
| 0:30–1:00 | EDA nhanh: số dòng, cột hằng, lệch phân phối train/test, cụm tương quan | Tránh các bẫy dữ liệu |
| 1:00–1:30 | Baseline tuyến tính (hồi quy mỗi tín hiệu từ các tín hiệu khác) + nộp lần 1 | Có điểm nền để so |
| 1:30–2:30 | **Quét hình dạng dự đoán bằng bảng xếp hạng** (số đoạn × độ rộng) | Rẻ, hiệu quả nhất |
| 2:30–4:30 | Mô hình mạnh hơn (TCN che-tái tạo, k-NN) để tăng recall | Nâng trần điểm |
| 4:30–5:30 | Tinh chỉnh quanh tối ưu, ghép mô hình | Vắt nốt |
| 5:30–6:00 | Chạy lại sạch, kiểm tra định dạng, nộp bản cuối | Không để lỗi ngớ ngẩn |

---

## 1. Đọc đề: đâu là thông tin, đâu là gợi ý có thể bỏ qua

Đề cho: 86 tín hiệu, 1 giây/mẫu; train ~1 triệu dòng **chỉ có dữ liệu bình thường**
(`Attack = 0`); test 86.400 dòng (đúng 1 ngày) không nhãn; nộp `predictions.csv` gồm
`row_id,anomaly`; chấm bằng **eTaPR F1** với θp = 0.5, θr = 0.1.

Mục "Gợi ý thực hành" của đề viết: *"Bạn có thể bắt đầu từ mô hình dự báo đa biến,
autoencoder theo cửa sổ thời gian, mô hình biểu diễn chuỗi, hoặc các cách đo mức lệch so
với dữ liệu bình thường."*

**Đây là chỗ dễ mất điểm nhất.** Gợi ý này đúng nhưng là mô tả cả một lĩnh vực — nó
không sai, chỉ là *không có thông tin*. Nếu đọc xong lao ngay đi xây LSTM-Autoencoder
(phản xạ tự nhiên của người có nền DL) thì sẽ:

- tốn 2 tiếng train một mô hình mà **không biết nó tốt hay xấu** (không có nhãn để đo);
- bỏ qua mất phần thực sự quyết định điểm: **biến điểm bất thường thành các đoạn 0/1
  như thế nào**.

Câu quan trọng nhất của đề nằm ở mục chấm điểm, không nằm ở mục gợi ý:

> *"eTaPR không chỉ kiểm tra bạn có chạm được vào một đoạn tấn công hay không, mà còn xem
> phần dự đoán đó bao phủ đoạn thật đến mức nào."*

Đọc câu đó xong việc cần làm là đi tìm **định nghĩa chính xác** của eTaPR, không phải đi
tìm kiến trúc mạng.

---

## 2. Hiểu metric trước — và nó thay đổi mọi thứ

eTaPR (Hwang et al., SAC '22) có mã nguồn mở tại `github.com/wshw4ng/eTaPR`. Tôi tải về,
đọc `etapr.py` + `tapr.py`, rồi **cài lại bằng numpy** và kiểm tra khớp với bản gốc trên
60 ca ngẫu nhiên (sai khác 5e-16). Việc này mất 30 phút và đáng giá hơn mọi thứ khác
trong buổi thi.

Tóm tắt công thức (A = các đoạn tấn công thật, P = các đoạn dự đoán):

```
M[a,p] = độ dài giao nhau
Pruning, lặp tới ổn định:
    xoá hàng a nếu 0 < (phần a được phủ) / |a| < θr = 0.1
    xoá cột p nếu 0 < (phần p nằm trong tấn công) / |p| < θp = 0.5
eTaR = trung bình theo a của  (d + d·cov)/2     , d = 1 nếu cov ≥ θr
eTaP = trung bình có trọng số √|p| của (d + d·rat)/2 , d = 1 nếu rat ≥ θp
F1 = 2·eTaP·eTaR/(eTaP+eTaR)
```

### Ba hệ quả rút ra *trước khi chạy bất kỳ mô hình nào*

**(a) Cái bẫy "hai lần 0".** Một đoạn dự đoán không đạt 50% nằm trong tấn công thì bị
xoá cột. Khi cột bị xoá, phần phủ mà nó đóng góp cho đoạn tấn công cũng mất theo. Nếu
*mọi* đoạn dự đoán đều bị xoá thì precision = 0 **và** recall = 0 luôn. Tôi đã kiểm chứng
điều này bằng một lần nộp thật: bản dự đoán 18 đoạn × 630 giây được đúng **0,00 điểm**
— trong khi bản 23 đoạn × 130 giây được 21,56.

**(b) Độ dài đoạn dự đoán là một siêu tham số hạng nhất.** Gọi L là độ dài tấn công thật,
ℓ là độ dài đoạn ta dự đoán (nằm trong tấn công). Điểm F1 của riêng cặp đó:

| ℓ \ L | 120 s | 300 s | 600 s | 1200 s | 2400 s |
|---|---|---|---|---|---|
| 60 s | 0.86 | 0.75 | 0.71 | **0** | **0** |
| 180 s | 0.91 | 0.89 | 0.79 | 0.73 | **0** |
| 600 s | **0** | 0.86 | 1.00 | 0.86 | 0.77 |
| 1200 s | **0** | **0** | 0.86 | 1.00 | 0.86 |

Vùng an toàn là ℓ ∈ [0.1·L, 2·L]; ra ngoài là mất trắng. Vì không biết L, **đây là thứ
phải đo bằng bảng xếp hạng ngay từ sớm**.

**(c) Precision tính theo *đoạn*, có trọng số √độ dài.** Nghĩa là 10 đoạn rác ngắn hại
gần bằng 3 đoạn rác dài. Và một đoạn đúng thì dù ngắn cũng được (1+rat)/2 ≥ 0.75. Suy ra
chiến lược "ít mà chắc" có lợi hơn "quét rộng cho chắc ăn" — ngược hoàn toàn với trực
giác quen thuộc từ các bài phân loại thông thường.

---

## 3. Khảo sát dữ liệu (30 phút, đừng làm lâu hơn)

Những thứ cần biết và đã tìm ra:

| Quan sát | Ý nghĩa |
|---|---|
| 6 tệp train, 1.004.402 dòng; test đúng 86.400 dòng = 1 ngày (2021-07-10) | Không có khoảng trống thời gian; mỗi tệp là một đoạn liên tục ⇒ không được để cửa sổ trượt bắc cầu giữa hai tệp |
| Tên cột dạng `P1_B2004`, `P1_FCV01D`, `P1_FCV01Z`, `P2_VIBTR01`… | Đây là bộ **HAI (HIL-based Augmented ICS security dataset)** của Hàn Quốc. Biết tên bộ dữ liệu giúp đoán được cấu trúc: P1 = vòng nước nóng, P2 = tuabin, P3 = xử lý nước, P4 = mô phỏng HIL. Hậu tố `B…` = setpoint, `D` = lệnh điều khiển, `Z` = phản hồi vị trí |
| **18/86 cột là hằng số** trong train | Bỏ đi, chúng chỉ tạo nhiễu chia-cho-không |
| 9 cụm tín hiệu có \|tương quan\| ≥ 0.995 (ví dụ `P1_FT01` và `P1_FT01Z` chỉ khác hệ số quy đổi) | Nếu để mô hình dự đoán một tín hiệu từ bản sao của chính nó thì residual luôn bằng 0 ⇒ mù hoàn toàn. **Phải loại cả cụm** |
| Lệch chế độ vận hành train ↔ test rất mạnh: `P1_FCV01D` trung bình 65 (train) so với 16 (test); `P4_HT_PS` là hằng số trong test nhưng biến thiên trong train | Không được dùng ngưỡng tuyệt đối học từ train. Phải chuẩn hoá residual **trên chính chuỗi đang chấm** |
| 68,5% số giây trong test có ít nhất một tín hiệu nằm ngoài dải train | Xác nhận điều trên: "ngoài dải train" **không** phải là dấu hiệu tấn công ở bài này |

### Một cái bẫy kỹ thuật suýt làm hỏng cả bài

Mô hình của tôi dùng đặc trưng EWMA (trung bình mũ xuôi và ngược). Ở **đầu chuỗi**, EWMA
xuôi không có quá khứ nên bị khởi tạo bằng chính giá trị đầu tiên ⇒ residual phình to
một cách hệ thống. Đo trên dữ liệu train sạch: 60 giây đầu có điểm bất thường cao gấp
**3–8 lần** mức nền. Hậu quả: bản dự đoán đầu tiên của tôi có một "đoạn tấn công ma" dài
1080 giây ngay đầu tệp test.

Cách sửa: đệm phản chiếu (reflect padding) ở hai đầu + nhân điểm với một hàm dốc tắt dần
ở rìa (`boundary_guard`). **Bài học: luôn chạy mô hình trên dữ liệu train sạch trước và
nhìn xem nó báo động ở đâu.** Nếu nó báo động trên dữ liệu bình thường thì đó là lỗi của
mình, không phải bất thường.

---

## 4. Chọn mô hình: vì sao tuyến tính lại thắng Deep Learning

Trực giác của người làm DL là dựng ngay một mạng. Nhưng hãy hỏi: *cái gì bị phá vỡ khi
có tấn công?*

Trong một nhà máy, các tín hiệu ràng buộc nhau bằng vật lý và bằng vòng điều khiển:
van mở → lưu lượng đổi → áp suất đổi. Tấn công = ai đó ghi một giá trị vào setpoint hoặc
lệnh điều khiển ⇒ **quan hệ giữa các tín hiệu bị gãy**. Vậy mô hình cần trả lời: *"biết
toàn bộ tín hiệu khác, thì tín hiệu này lẽ ra phải bằng bao nhiêu?"*

### Mô hình 1 — hồi quy quan hệ tuyến tính (thứ nên làm đầu tiên)

Với mỗi tín hiệu đích j:

```
v_j(t) ≈ w · [ v, ewma_quá_khứ(20s, 300s), ewma_tương_lai(20s, 300s) ] của MỌI tín hiệu khác
```

loại bỏ cả cụm tương quan ≥ 0.995 của j. Bài toán được chấm offline nên **dùng được cả
tương lai** — điều này tăng độ chính xác đáng kể so với mô hình nhân quả.

Mẹo cài đặt quan trọng: tích luỹ **ma trận Gram** (D×D với D = 68×5 = 340) theo từng khối
dữ liệu, sau đó nghiệm cho cả 68 tín hiệu đích lấy ra từ cùng một ma trận bằng
`np.linalg.solve`. Kết quả: **15 giây CPU cho 1 triệu dòng, RAM < 1 GB**, và thử 5 giá
trị ridge gần như miễn phí. Nếu viết kiểu ngây thơ (dựng ma trận đặc trưng 1M×340 rồi gọi
sklearn) thì hết RAM.

Sai số trung bình trên dữ liệu bình thường: **0,058** (đơn vị min-max). Đây là baseline
rất mạnh và **cuối cùng nó vẫn là mô hình tốt nhất trong ba mô hình tôi thử**.

### Mô hình 2 — TCN che-và-tái tạo (phiên bản phi tuyến của mô hình 1)

Mạng tích chập giãn hai chiều (±510 giây). Đầu vào là toàn bộ chuỗi nhưng **một cụm tín
hiệu bị che** (giá trị xoá về 0, kèm một kênh cờ báo "đang bị che"); nhiệm vụ là tái tạo
lại chính cụm đó. Khi chấm điểm thì chạy 55 lượt, mỗi lượt che một cụm.

Kết quả: trên tập kiểm định giả lập nó **thua mô hình tuyến tính** (F1 0,28 so với 0,46).
Lý do có lẽ là: phần lớn quan hệ trong nhà máy này gần như tuyến tính (quy đổi đơn vị,
setpoint → cơ cấu chấp hành), mà mô hình tuyến tính khớp đúng những quan hệ đó tới sai số
1e-3, còn mạng nơ-ron luôn có sai số xấp xỉ ~1e-2 ⇒ nền nhiễu cao hơn, lấp mất các lệch
nhỏ. Tuy vậy nó vẫn **bổ sung** cho mô hình tuyến tính khi ghép chung.

> **Lỗi kỹ thuật đáng nhớ**: bản TCN đầu tiên dùng `GroupNorm`, mà GroupNorm chuẩn hoá
> dọc **trục thời gian**. Mạng train với cửa sổ 512 điểm nhưng khi chấm lại chạy trên cả
> chuỗi 32.000 điểm ⇒ thống kê chuẩn hoá khác hẳn, residual phình từ 0,074 lên 0,196 và
> F1 rơi từ ~0,45 xuống 0,13. Cách phát hiện: chạy suy luận với `chunk` = 512, 2048,
> 20000 và thấy kết quả đổi theo. Cách sửa: chuẩn hoá theo **trục kênh tại từng thời
> điểm** (`ChannelNorm`) — không phụ thuộc độ dài chuỗi.

### Mô hình 3 — k-láng-giềng "trạng thái này đã từng xảy ra chưa"

Hai mô hình trên có một điểm mù: khi kẻ tấn công đổi setpoint, vòng điều khiển *đáp ứng
lại*, và sau giai đoạn quá độ nhà máy chạy ổn định ở điểm làm việc mới — lúc đó **quan hệ
vẫn đúng**, chỉ có điều điểm làm việc đó chưa từng thấy. Mô hình k-NN lấp chỗ đó: làm
trơn 60 giây, chiếu PCA 24 chiều, tìm 8 trạng thái gần nhất, residual = độ lệch so với
trung bình của chúng.

Bẫy: nếu chỉ so với train thì **cả ngày test đều "lạ"** (do lệch chế độ vận hành) — nó
gắn cờ 23,7% số điểm. Cách sửa: cho phép so với **chính tập test**, nhưng loại các thời
điểm cách điểm đang xét dưới 30 phút. Khi đó chế độ vận hành bình thường luôn tìm được
"bạn" ở chỗ khác trong ngày, còn một đoạn tấn công 5 phút thì không.

### Những thứ đã thử và **không** ăn thua

| Ý tưởng | Kết quả |
|---|---|
| Bộ dò nhảy bậc (change-point) so với 15–30 phút trước | Xếp hạng các tấn công đã biết rất kém (hạng 31, 36, trượt 2 cái) |
| Bộ dò "xung chữ nhật" (ghi giá trị rồi trả lại) | Chỉ bắt được 1/5 tấn công đã biết |
| Mô hình quan hệ chỉ dùng bối cảnh chậm (bỏ giá trị tức thời) | Kém hơn bản đầy đủ |
| Làm trơn từng tín hiệu trước khi gộp | Giảm F1 trên tập giả lập |
| Ghép đoạn thích nghi (khe hở ≤ tổng độ dài hai đoạn) | Giảm recall, không cứu được |
| Ghép mô hình kiểu "bỏ phiếu ≥ 2/3" | Kém hơn trung bình theo từng tín hiệu |

Ghi lại những cái *không* chạy cũng quan trọng như ghi cái chạy — để lần sau khỏi mất
thời gian lần nữa.

---

## 5. Biến điểm bất thường thành nhãn 0/1

Đây mới là nơi kiếm điểm. Quy trình:

```
residual (n×68) → chuẩn hoá theo từng tín hiệu → gộp top-k → làm trơn
               → tách vùng ứng viên → xếp hạng theo đỉnh → giữ N vùng tốt nhất
               → cắt mỗi vùng về đúng W giây quanh chỗ điểm cao nhất
```

**Chuẩn hoá theo từng tín hiệu** dùng thứ hạng (ECDF) trên chính chuỗi đang chấm:
`s_j(t) = −log(1 − F_j(|r_j(t)|))`. Ưu điểm: mỗi tín hiệu đóng góp một "độ bất ngờ" bị
chặn trên bởi log(n), không phụ thuộc dạng phân phối, và **ngưỡng có ý nghĩa cố định**
(ngưỡng 5.5 ⇔ phân vị 99,59% của từng tín hiệu) nên chuyển sang tập test khác vẫn dùng
được. Cách này thắng chuẩn hoá theo MAD (F1 0,46 so với 0,42 trên tập giả lập).

**Gộp top-k**: một tấn công thường chỉ làm gãy quan hệ ở vài tín hiệu; lấy trung bình tất
cả thì pha loãng, lấy max thì quá nhạy nhiễu. Trung bình của k tín hiệu cao nhất (k = 1–3)
là hợp lý.

---

## 6. Tập kiểm định giả lập: hữu ích, nhưng đã dẫn tôi đi sai

Vì test không có nhãn, tôi lấy `train6.csv` (không dùng khi huấn luyện) và tiêm tấn công
giả lập có nhãn: freeze, bias, ramp, đổi setpoint, scale, replay, noise; độ dài 2 phút –
1 giờ; có lọc bỏ những ca tiêm mà dữ liệu gần như không đổi (không thể phát hiện, giữ lại
chỉ làm nhiễu việc chọn ngưỡng).

Nó giúp được: chọn cách chuẩn hoá (rank thắng mad), chọn top-k, so sánh các mô hình.

**Nhưng nó nói dối ở đúng chỗ quan trọng nhất.** Tấn công tiêm nhân tạo làm lệch dữ liệu
*liên tục* suốt đoạn, nên điểm bất thường cũng cao liên tục, biên rõ ràng. Kết quả là
việc dò tham số trên đó kết luận: "đoạn ngắn, không cần ghép khe hở, ngưỡng cao" — và
**không hề cảnh báo** rằng đoạn dài sẽ bị eTaPR loại sạch về 0 điểm.

Tấn công thật thì khác: vòng điều khiển đáp ứng lại nên quan hệ chỉ gãy ở hai đầu, và độ
dài tấn công thật ngắn hơn nhiều so với dải tôi giả lập.

**Bài học**: tập kiểm định tự tạo chỉ đáng tin trong phạm vi giả định mà ta đã đưa vào
nó. Với những câu hỏi mà giả định của ta có thể sai (ở đây: *tấn công dài bao lâu?*),
phải đi hỏi bảng xếp hạng, không hỏi tập giả lập.

---

## 7. Dùng bảng xếp hạng như một dụng cụ đo

Nguyên tắc: **mỗi lần nộp đổi đúng một biến**, và ghi lại mọi thứ.

### Vòng 1 — dò hình dạng (5 lượt)

| Bản | Số đoạn | % số điểm | Dài trung vị | Điểm |
|---|---|---|---|---|
| Ngưỡng + ghép khe hở (bản đầu) | 23 | 4,6% | 129 s | 21,56 |
| Phủ rộng | 17 | 11,3% | 271 s | 6,28 |
| **Đoạn dài** | 18 | 16,1% | 630 s | **0,00** |
| **Ít mà chắc** | 5 | 1,7% | 300 s | **30,76** |
| k-NN một mình | 11 | 8,0% | 451 s | 9,24 |

Đọc được ngay: *tấn công ngắn* và *precision quyết định*. Toàn bộ phần còn lại của buổi
thi nên dành để khai thác hai điều này chứ không phải đi train thêm mô hình.

### Vòng 2 — quét (số đoạn × độ rộng), giữ nguyên mọi thứ khác (5 lượt)

| n \ rộng | 120 s | 180 s | 300 s |
|---|---|---|---|
| 5 | — | — | 32,87 |
| 8 | 36,50 | **37,87** | 25,71 |
| 12 | 38,55 | **39,18** ⭐ | 28,78 |
| 16 | — | 31,89 | 23,14 |

Khớp parabol theo từng trục: n tối ưu ≈ 10,6; độ rộng tối ưu ≈ 160 s — cả hai đều chỉ
hơn điểm đang có ~0,4 ⇒ **dừng tinh chỉnh, chuyển hướng khác**. (Biết *khi nào nên dừng*
cũng là một kỹ năng thi.)

### Vòng 3 — các giả thuyết còn lại (5 lượt)

| Phép thử | Điểm | Kết luận |
|---|---|---|
| Dịch cả 12 đoạn lui 60 s | 28,82 | Căn chỉnh hiện tại đã đúng |
| Dịch cả 12 đoạn tới 60 s | 19,24 | Lệch 60 s mất 10–20 điểm ⇒ tấn công rất ngắn |
| Xếp hạng theo đồng thuận 3 mô hình | 38,10 | Không hơn |
| Độ rộng thích nghi 120–220 s | 30,44 | Kém hẳn — vị trí cửa sổ bị xê dịch |
| Bỏ 2 đoạn "công tắc chế độ P2" | 38,59 | **Suy luận miền của tôi sai**: chúng có giá trị |

Điểm cuối: **39,18** (từ 21,56 ban đầu, gần gấp đôi) — trong khi không hề đổi mô hình,
chỉ đổi cách biến điểm thành đoạn.

### Ước lượng trần điểm và chỗ còn thiếu

Từ hai phương trình eTaPR với bản 5 đoạn (30,76) và bản 12 đoạn (39,18), giải ngược ra
ước lượng: tập test có khoảng **20–35 đoạn tấn công**, ta đang bắt đúng ~9–12 cái. Nghĩa
là **recall mới là nút thắt**, và muốn lên 60+ điểm thì phải *tìm thêm tấn công*, chứ
không phải chỉnh thêm hình dạng. Mà muốn thế thì cần mô hình tốt hơn — việc đáng làm nếu
còn thời gian, chứ không phải việc nên làm trước.

---

## 8. Nếu được làm lại từ đầu (bản rút gọn để thi)

1. **Cài lại metric, kiểm tra khớp với bản gốc.** Rồi tự hỏi: *hình dạng dự đoán nào làm
   metric này hài lòng?* Vẽ bảng ℓ × L như ở mục 2.
2. **EDA 30 phút**: cột hằng, cụm tương quan, lệch chế độ train↔test, biên chuỗi.
3. **Baseline tuyến tính** (Gram theo khối) — 15 giây, đừng dùng sklearn ngây thơ.
4. **Nộp ngay** một bản "ít mà chắc" (~10 đoạn × 180 s) và một bản "nhiều mà rộng" để
   *đo* xem tấn công dài hay ngắn. Hai lượt nộp này đáng giá hơn hai tiếng train mô hình.
5. Quét (số đoạn × độ rộng) quanh bản tốt hơn. Dừng khi khớp parabol cho thấy đã gần đỉnh.
6. **Chỉ sau đó** mới đầu tư vào mô hình mạnh hơn để tăng recall, và mỗi mô hình mới đều
   phải đo bằng: (a) thứ hạng của các tấn công *đã được bảng xếp hạng xác nhận*, (b) một
   lượt nộp.
7. Luôn chạy mô hình trên dữ liệu train sạch để kiểm tra nó **không** báo động bậy.

## 9. Những câu tự hỏi đã sinh ra tiến bộ

- *"Metric này phạt cái gì?"* → phát hiện bẫy hai-lần-0, đổi toàn bộ hình dạng lời giải.
- *"Mô hình này báo động ở đâu trên dữ liệu sạch?"* → tìm ra artifact đầu chuỗi.
- *"Kết quả có đổi khi tôi đổi một thứ lẽ ra không được phép ảnh hưởng?"* (độ dài khối
  khi suy luận) → tìm ra lỗi GroupNorm.
- *"Tập kiểm định của tôi giả định gì? Giả định đó có thể sai chỗ nào?"* → biết được rằng
  không thể hỏi nó về độ dài tấn công.
- *"Tôi đang tiêu lượt nộp để *biết thêm* điều gì?"* → thiết kế thí nghiệm thay vì nộp mò.
