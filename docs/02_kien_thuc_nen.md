# Kiến thức nền cho bài phát hiện xâm nhập hệ điều khiển công nghiệp

*Tài liệu tham khảo: những gì cần biết về miền bài toán, họ mô hình, cách xử lý dữ liệu
và họ metric — dành cho người có nền DL nhưng lần đầu gặp dữ liệu ICS.*

---

## 1. Hệ điều khiển công nghiệp và bộ dữ liệu HAI

### 1.1 Kiến trúc một hệ ICS

```
Cảm biến  →  PLC / DCS  →  Cơ cấu chấp hành (van, bơm)
              ↑    ↓
          SCADA / HMI (người vận hành đặt setpoint)
```

Vòng điều khiển kín (PID) liên tục so **giá trị đặt** (setpoint) với **giá trị đo**
(process value) rồi xuất **lệnh điều khiển** (control output) tới van/bơm. Đặc điểm quan
trọng nhất với người làm ML: **các tín hiệu không độc lập** — chúng bị ràng buộc bởi
phương trình vật lý và bởi luật điều khiển. Đó chính là cấu trúc để khai thác.

### 1.2 Quy ước đặt tên trong HAI

Bộ dữ liệu của bài này là **HAI (HIL-based Augmented ICS Security Dataset)** do Viện
nghiên cứu an ninh quốc gia Hàn Quốc phát hành (bản 22.04 có 86 tín hiệu). Hệ gồm 4 quá
trình ghép lại bằng mô phỏng HIL (Hardware-In-the-Loop):

| Tiền tố | Quá trình |
|---|---|
| `P1_` | Vòng nước nóng (boiler): bồn, bơm, van điều khiển lưu lượng/mức/áp suất |
| `P2_` | Tuabin hơi: tốc độ, rung, công tắc chế độ auto/manual |
| `P3_` | Xử lý/cấp nước: mức bồn, bơm, lưu lượng |
| `P4_` | Mô phỏng HIL nhà máy điện: phụ tải, công suất, nhiệt độ hơi |

Hậu tố cho biết vai trò của tín hiệu — **rất quan trọng khi diễn giải kết quả**:

| Hậu tố / dạng tên | Ý nghĩa | Ví dụ |
|---|---|---|
| `B####` | Giá trị đặt (setpoint) do bộ điều khiển/người vận hành ghi | `P1_B2004` (áp suất đặt), `P1_B3004` (mức nước đặt) |
| `…D` | Lệnh điều khiển gửi tới cơ cấu chấp hành | `P1_FCV01D` (lệnh mở van FCV01) |
| `…Z` | Phản hồi vị trí thực tế của cơ cấu chấp hành | `P1_FCV01Z` |
| `FT`, `LIT`, `PIT`, `TIT` | Cảm biến lưu lượng / mức / áp suất / nhiệt độ | `P1_FT01`, `P1_LIT01` |
| `PP`, `SOL` | Bơm, van điện từ | `P1_PP01AD` |
| `…Z` sau cảm biến | Giá trị đã quy đổi thang đo | `P1_FT01Z = k · P1_FT01` |

Hệ quả thực tế: cặp `D`/`Z` và cặp `FT`/`FTZ` gần như trùng nhau (tương quan > 0,999).
Nếu để mô hình dự đoán `P1_FT01` từ `P1_FT01Z` thì residual luôn ~0 ⇒ mù. **Phải gom cụm
tương quan cao và loại cả cụm ra khỏi tập biến giải thích.**

### 1.3 Tấn công trong HAI trông như thế nào

Kịch bản tấn công là ghi giá trị giả vào một điểm điều khiển (setpoint, lệnh, hoặc giá
trị cảm biến báo về). Ba dạng chính:

1. **Đổi setpoint** — vòng điều khiển *đáp ứng lại*: van chạy, áp suất/mức đổi theo. Sau
   quá độ, nhà máy chạy ổn định ở điểm làm việc mới. Quan hệ giữa các tín hiệu lúc này
   trông *vẫn hợp lệ* ⇒ mô hình quan hệ chỉ thấy bất thường ở **hai đầu**.
2. **Đổi lệnh điều khiển trực tiếp** (bỏ qua PID) — mâu thuẫn giữa sai số điều khiển và
   lệnh xuất ra kéo dài suốt thời gian tấn công ⇒ dễ phát hiện hơn.
3. **Giả mạo giá trị cảm biến** (có khi kèm "che giấu": phát lại giá trị cũ để HMI trông
   bình thường) — mâu thuẫn giữa cảm biến bị giả và các cảm biến khác.

Độ dài: **rất ngắn**, thường vài phút. Với bài này, phản hồi từ bảng xếp hạng cho thấy
tấn công chỉ cỡ **150–250 giây** (xem `01_cach_tiep_can.md`, mục 7).

---

## 2. Bài toán học máy: học nửa giám sát trên chuỗi đa biến

Điều kiện: **chỉ có dữ liệu bình thường để huấn luyện** (semi-supervised / one-class).
Không có ví dụ tấn công nào. Do đó:

- Không dùng được phân loại có giám sát.
- Mọi mô hình đều theo mô thức: *học cái "bình thường" → đo mức lệch → đặt ngưỡng*.
- Phần "đặt ngưỡng" không có nhãn để tối ưu ⇒ đây luôn là điểm yếu nhất của toàn bộ
  pipeline, và là chỗ nên đầu tư công sức nhất.

---

## 3. Các họ mô hình và độ phù hợp với ICS

### 3.1 Dự báo (forecasting)

Học `x(t) ≈ f(x(t−W..t−1))`; residual = sai số dự báo.

- Kiến trúc hay dùng: LSTM/GRU xếp chồng (baseline của cuộc thi HAICon), TCN, Transformer.
- **Ưu**: đơn giản, bắt tốt các thay đổi đột ngột.
- **Nhược nghiêm trọng với ICS**: mô hình nhìn thấy *quá khứ đã bị tấn công*, nên sau vài
  giây nó "bám theo" giá trị bị sửa và residual về lại bình thường ⇒ chỉ phát hiện được
  **thời điểm chuyển tiếp**, không phủ được cả đoạn tấn công. Với metric dạng đoạn
  (eTaPR) thì đây là hạn chế lớn.

### 3.2 Tái tạo (reconstruction)

Học nén–giải nén một cửa sổ: AE, VAE, LSTM-AE, USAD (hai AE đối kháng), Anomaly
Transformer. Residual = sai số tái tạo.

- **Ưu**: không cần nhãn, dễ cài.
- **Nhược**: AE đủ mạnh sẽ tái tạo tốt cả dữ liệu bất thường ("identity shortcut"); phải
  bóp nút thắt cổ chai, thêm nhiễu, hoặc dùng masking.

### 3.3 Quan hệ giữa các biến (relation / graph) — **phù hợp nhất cho ICS**

Học `x_j(t) ≈ f(mọi tín hiệu khác)`. Đại diện: GDN (Graph Deviation Network), MTAD-GAT,
và bản đơn giản nhất là **hồi quy tuyến tính từng biến một** (chính là cái thắng ở bài này).

- **Ưu**: đúng với bản chất ICS (vật lý ràng buộc các biến); residual **kéo dài suốt**
  thời gian quan hệ bị phá vỡ; rất rẻ nếu cài bằng ma trận Gram.
- **Lưu ý bắt buộc**: phải loại biến đích *và các bản sao của nó* khỏi đầu vào.
- Biến thể mạnh: **masked reconstruction** — che một nhóm biến trên toàn cửa sổ rồi tái
  tạo (bản phi tuyến của ý tưởng trên; là mô hình TCN trong repo này).

### 3.4 Khoảng cách / mật độ

k-NN, LOF, Isolation Forest, Mahalanobis, PCA + thống kê Q (SPE).

- **Ưu**: trả lời câu hỏi khác hẳn — *"trạng thái này đã từng xảy ra chưa?"* — nên bù
  được điểm mù "sau quá độ" của mô hình quan hệ. Rất rẻ nếu chiếu PCA trước.
- **Nhược**: cực nhạy với dịch chuyển chế độ vận hành giữa train và test. Mẹo hoá giải:
  cho phép so khớp với **chính chuỗi đang chấm**, loại bỏ vùng lân cận thời gian (±30
  phút) để một đoạn tấn công không tự tìm thấy "bạn" là chính nó.

### 3.5 Điểm thay đổi và biểu đồ kiểm soát

CUSUM, EWMA control chart, so sánh cửa sổ trước/sau. Rẻ và dễ hiểu, nhưng kêu ở mọi thao
tác vận hành bình thường ⇒ chỉ nên dùng như *bằng chứng bổ sung*, không dùng một mình.

### 3.6 Bất biến vật lý (invariant-based)

Cách tiếp cận kinh điển cho SWaT/WADI: trích các bất biến dạng `nếu van mở thì lưu lượng
> 0`, `mức bồn tăng khi bơm vào chạy`. Chính xác rất cao nhưng tốn công miền và thời
gian — không hợp với 6 giờ thi.

### 3.7 Bảng tóm tắt lựa chọn

| Họ | Chi phí | Phủ hết đoạn tấn công? | Hợp với 6 giờ thi? |
|---|---|---|---|
| Hồi quy quan hệ tuyến tính | rất thấp (15 s CPU) | một phần | **Nên làm đầu tiên** |
| Masked TCN / GDN | trung bình (20–40 phút GPU) | một phần | Có, nếu còn thời gian |
| Dự báo LSTM/GRU | trung bình | kém (chỉ bắt quá độ) | Không ưu tiên |
| Autoencoder cửa sổ | trung bình | trung bình | Không ưu tiên |
| k-NN/PCA novelty | thấp (nếu chiếu PCA) | **tốt** | Có |
| Change-point | rất thấp | tốt | Chỉ làm bằng chứng phụ |

---

## 4. Xử lý dữ liệu

### 4.1 Chuẩn hoá

- **Min-max theo train** là lựa chọn mặc định cho ICS (các tín hiệu có đơn vị rất khác
  nhau: 0–1 với cờ nhị phân, 0–27000 với nhiệt độ quy đổi). Nhớ **cắt** giá trị test vào
  khoảng như `[-0.5, 1.5]` để một điểm ngoài dải train không phá huỷ mô hình.
- Bỏ cột hằng số (ở HAI 22.04: 18/86 cột).
- Gom cụm tương quan ≥ 0,995 bằng union-find.

### 4.2 Đặc trưng thời gian rẻ mà hiệu quả

Với mỗi tín hiệu, thêm EWMA **quá khứ** và **tương lai** ở vài thang thời gian
(ví dụ nửa đời 20 s và 300 s). Bài chấm offline nên dùng tương lai là hợp lệ và giúp
nhiều. Cài bằng `scipy.signal.lfilter` (IIR bậc 1) — O(n).

> **Bẫy biên**: EWMA ở đầu/cuối chuỗi không có bối cảnh thật. Nếu khởi tạo bằng giá trị
> đầu tiên thì residual phình to một cách hệ thống và tạo ra "đoạn bất thường ma". Xử lý:
> đệm phản chiếu (reflect) độ dài ≥ 6 lần nửa đời, cộng thêm một hàm dốc dập điểm ở rìa.

### 4.3 Mẹo tính toán cho dữ liệu lớn với RAM nhỏ

- **Ma trận Gram theo khối**: muốn hồi quy 68 biến đích trên 1 triệu dòng × 340 đặc
  trưng, đừng dựng ma trận đầy đủ (1,4 GB). Hãy tích luỹ `G += Zᵀ Z` (340×340) theo từng
  khối 20 000 dòng, rồi giải nghiệm cho mọi biến đích từ cùng một `G`. RAM < 1 GB, 15 giây.
- **Trung bình trượt bằng cumsum**: O(n) thay vì O(n·w).
- **Suy luận chia khối có chồng lấn** cho mạng tích chập: phần chồng lấn ≥ receptive field.
- Đọc thẳng CSV trong `.zip` bằng `polars` (nhanh gấp ~5 lần `pandas`), không cần giải nén.

---

## 5. Từ residual đến điểm bất thường

### 5.1 Chuẩn hoá theo từng tín hiệu (bắt buộc)

Mỗi tín hiệu có mức nhiễu khác nhau vài bậc. Hai cách:

| Cách | Công thức | Ghi chú |
|---|---|---|
| Robust z-score | `(|r| − median) / (1.4826·MAD)` | Cần **sàn** cho MAD, nếu không tín hiệu gần như xác định sẽ cho z khổng lồ |
| **Thứ hạng (ECDF)** | `−log(1 − F(|r|))` | Bị chặn trên bởi log(n), không phụ thuộc dạng phân phối, ngưỡng có ý nghĩa phân vị cố định ⇒ **chuyển tập test vẫn dùng được** |

Tính thống kê **trên chính chuỗi đang chấm** (transductive): tấn công chỉ chiếm vài phần
trăm số điểm nên median/ECDF gần như không bị ảnh hưởng, trong khi mọi sai khác do đổi
chế độ vận hành thì bị trừ hết. Đây là cách hoá giải dịch chuyển phân phối train↔test.

### 5.2 Gộp nhiều tín hiệu

`max` quá nhạy, `mean` pha loãng ⇒ dùng **trung bình top-k** (k = 1–3). Với nhiều mô
hình: ghép ở mức *từng tín hiệu* (trung bình các z rồi mới gộp top-k) thường tốt hơn ghép
ở mức điểm cuối.

### 5.3 Đặt ngưỡng khi không có nhãn

- **Phân vị cố định** (gắn cờ p% số điểm) — đơn giản, cần đoán p.
- **POT / lý thuyết giá trị cực trị** (SPOT, DSPOT): khớp phân phối Pareto tổng quát cho
  phần đuôi, chọn ngưỡng theo xác suất rủi ro q. Bài bản hơn nhưng vẫn phải chọn q.
- **Tấn công giả lập**: tự tiêm bất thường vào dữ liệu bình thường để có nhãn mà dò. Rất
  hữu ích, nhưng **chỉ đúng trong phạm vi giả định đã tiêm** (xem bài học ở tài liệu 01).
- **Bảng xếp hạng**: nếu cuộc thi cho nộp nhiều lần, đây là nguồn nhãn duy nhất *thật*.
  Hãy tiêu lượt nộp như tiêu ngân sách thí nghiệm.

---

## 6. Metric cho phát hiện bất thường dạng đoạn

Đây là phần hay bị coi nhẹ nhất mà lại quyết định nhất.

### 6.1 Point-wise F1

Tính trên từng điểm. Nhược: một tấn công dài 1 giờ áp đảo mười tấn công 1 phút.

### 6.2 Point-adjust (PA)

Nếu chạm được *một điểm* trong đoạn tấn công thì coi như phát hiện **cả đoạn**. Rất phổ
biến trong các bài báo 2019–2021 nhưng đã bị chứng minh là **thổi phồng**: một bộ sinh
ngẫu nhiên cũng đạt F1 cao. Đừng tin các con số PA.

### 6.3 TaPR / eTaPR (dùng trong bài này)

Tách điểm thành hai phần cho cả precision và recall: **phát hiện được hay không**
(detection) và **phủ được bao nhiêu** (portion).

```
M[a,p] = |a ∩ p|
Pruning lặp:  xoá hàng a nếu 0 < Σ_p M[a,p]/|a| < θr
              xoá cột p nếu 0 < Σ_a M[a,p]/|p| < θp
cov[a] = Σ_p M[a,p]/|a| ;  rat[p] = Σ_a M[a,p]/|p|
eTaR = mean_a ( d_a + d_a·min(cov,1) )/2 ,  d_a = 1{cov ≥ θr}
eTaP = Σ_p √|p| ( d_p + d_p·rat )/2  /  Σ_p √|p| ,  d_p = 1{rat ≥ θp}
```

Ba điều cần khắc cốt ghi tâm:

1. **Đoạn dự đoán phải nằm ≥ θp (50%) trong tấn công**, nếu không nó bị xoá — và phần phủ
   mà nó đóng góp cho đoạn tấn công cũng mất luôn. Dự đoán quá dài ⇒ precision **và**
   recall cùng về 0.
2. **Chỉ cần phủ ≥ θr (10%) là tính "phát hiện được"**, phủ thêm thì được điểm portion.
   Suy ra: vài đoạn ngắn *đặt đúng chỗ* tốt hơn một đoạn dài tràn ra ngoài.
3. **Precision có trọng số √độ dài** và tính theo *đoạn*, không theo điểm ⇒ nhiều đoạn
   rác ngắn vẫn giết điểm.

Vùng an toàn cho độ dài đoạn dự đoán ℓ so với độ dài tấn công thật L: **ℓ ∈ [0.1·L, 2·L]**,
tối ưu ở ℓ ≈ L. Không biết L thì đây là siêu tham số **phải đo**.

### 6.4 Họ metric khác nên biết

- **Affiliation-based precision/recall** (Huet et al., KDD'22): dựa trên khoảng cách
  trung bình tới đoạn gần nhất, không có ngưỡng cứng.
- **VUS-ROC / VUS-PR** (Paparrizos et al., VLDB'22): tích phân theo bán kính dung sai,
  tránh việc chọn ngưỡng.
- **NAB score**: cửa sổ có trọng số giảm dần, thưởng phát hiện sớm.

Ý chung: **luôn đọc mã nguồn metric của cuộc thi trước khi thiết kế hậu xử lý.**

---

## 7. Hậu xử lý: biến điểm số thành các đoạn

Các phép biến đổi cơ bản và tác dụng lên eTaPR:

| Phép | Tác dụng |
|---|---|
| Làm trơn (trung bình trượt w giây) | Bỏ gai nhiễu 1–2 giây; w quá lớn thì làm nhoè biên |
| Ngưỡng trễ (hysteresis) th_hi/th_lo | Bắt đầu ở đỉnh chắc chắn, kéo dài tới chân dốc |
| Ghép khe hở ≤ g | Nối hai quá độ đầu/cuối của cùng một tấn công — **nhưng nếu ghép nhầm hai tấn công khác nhau thì đoạn ghép có thể tụt dưới θp và mất cả hai** |
| Bỏ đoạn < min_len | Giảm báo động giả rời rạc |
| **Cắt về độ rộng cố định quanh đỉnh** | Cách an toàn nhất khi tấn công ngắn: kiểm soát trực tiếp ℓ |
| Nới đoạn (dilate) | Chỉ dùng khi tin chắc tấn công dài hơn phần bắt được |

Với eTaPR và tấn công ngắn, tham số hoá tốt nhất mà tôi tìm được là **hai số**:
*số đoạn giữ lại trên mỗi 24 giờ* và *độ rộng mỗi đoạn* — vì chúng tách bạch đúng hai
thứ metric quan tâm (precision ↔ recall, và điều kiện θp).

---

## 8. Kỹ thuật chạy được trong phòng thi

- **Đặt số đoạn theo tỉ lệ/24 giờ**, không đặt số tuyệt đối — tập private có thể dài ngắn
  khác public.
- Ghi lại **mọi lần nộp** kèm tham số vào một bảng. Cuối buổi nhìn lại sẽ thấy xu hướng.
- Mỗi lần nộp chỉ đổi **một** biến.
- Khớp parabol qua 3 điểm để biết còn cách đỉnh bao xa — nếu chỉ còn < 1 điểm thì dừng,
  chuyển hướng khác.
- Kiểm tra định dạng nộp bằng script (đủ số dòng, `row_id` liên tục từ 0, chỉ có 0/1).
- Chạy lại toàn bộ pipeline từ đầu trong 30 phút cuối để chắc chắn kết quả tái lập được.

---

## 9. Tài liệu tham khảo

**Bộ dữ liệu**
- HAI Security Dataset — <https://github.com/icsdataset/hai>
- SWaT / WADI (iTrust, SUTD) — các bộ tương đương hay dùng để so sánh

**Metric**
- Hwang et al., *"Do you know existing accuracy metrics overrate time-series anomaly
  detections?"*, SAC 2022 — eTaPR. Mã nguồn: <https://github.com/wshw4ng/eTaPR>
- Kim et al., *"Towards a rigorous evaluation of time-series anomaly detection"*, AAAI
  2022 — phê phán point-adjust
- Huet et al., *"Local evaluation of time series anomaly detection algorithms"*, KDD 2022
  — affiliation metrics

**Mô hình**
- Deng & Hooi, *"Graph Neural Network-Based Anomaly Detection in Multivariate Time
  Series"*, AAAI 2021 — GDN
- Audibert et al., *"USAD: UnSupervised Anomaly Detection on Multivariate Time Series"*,
  KDD 2020
- Bai et al., *"An Empirical Evaluation of Generic Convolutional and Recurrent Networks"*,
  2018 — TCN
- Siffer et al., *"Anomaly Detection in Streams with Extreme Value Theory"*, KDD 2017 —
  POT/SPOT

**Cuộc thi tham khảo**
- HAICon 2020/2021 (Dacon, Hàn Quốc) — cùng bộ dữ liệu HAI, chấm bằng TaPR; baseline công
  khai là GRU xếp chồng dự báo bước kế tiếp
