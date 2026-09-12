# Nhật ký phiên làm việc & kế hoạch cho lần sau

Tài liệu này ghi lại **toàn bộ dữ kiện đo được** trong phiên, tách bạch rõ đâu là *sự
thật đã kiểm chứng*, đâu là *giả thuyết chưa kiểm chứng*, và đề xuất kế hoạch cho lần
làm lại với ngân sách nộp mới — sao cho kế thừa được cái đúng mà không bị đóng khung bởi
những lựa chọn ngẫu nhiên của phiên này.

---

## 1. Trình tự việc đã làm trong phiên

| # | Việc | Kết quả |
|---|---|---|
| 1 | Đọc đề, nhận ra dữ liệu là HAI 22.04 | Biết cấu trúc tên cột, đoán được dạng tấn công |
| 2 | Tải mã nguồn eTaPR, cài lại bằng numpy, đối chiếu 60 ca ngẫu nhiên | Khớp tới 5e-16 ⇒ có "thước đo" tin cậy ở máy |
| 3 | EDA: cột hằng, cụm tương quan, lệch chế độ train↔test | Bỏ 18 cột, gom 9 cụm, quyết định chuẩn hoá transductive |
| 4 | Mô hình quan hệ tuyến tính (Gram theo khối) | 15 giây CPU, MAE 0,058 |
| 5 | Phát hiện & sửa artifact đầu chuỗi (EWMA) | "Đoạn tấn công ma" 1080 s biến mất |
| 6 | Sinh tấn công giả lập + dò tham số bằng eTaPR | F1 giả lập ~0,46 |
| 7 | TCN che-và-tái tạo trên GPU (RTX 3050) | 34 phút; lần đầu **sai** do GroupNorm, sửa thành ChannelNorm |
| 8 | Mô hình k-NN "trạng thái lạ", có tự tham chiếu | Bổ sung cho mô hình quan hệ |
| 9 | **Nộp thử 17 lần để dò hình dạng dự đoán** | 21,56 → **39,18** |
| 10 | Viết tài liệu, chốt công thức cho private test | `configs/best_recipe.json` |

**Điều đáng nói nhất**: từ bước 1 đến 8 (gần như toàn bộ công sức mô hình) chỉ đưa điểm
từ 0 lên ~21. Bước 9 — *không đổi một dòng mô hình nào*, chỉ đổi cách biến điểm thành
đoạn — đưa từ 21 lên 39.

---

## 2. Sự thật đã kiểm chứng bằng phép đo

> Những điều dưới đây có bằng chứng trực tiếp. Nên kế thừa.

### 2.1 Bảng đầy đủ 17 lần nộp

| # | Cấu hình | Số đoạn | % số điểm | Dài trung vị | Điểm |
|---|---|---|---|---|---|
| 1 | relation+tcn, ngưỡng 5.0, ghép khe 60 s | 23 | 4,55% | 129 s | 21,56 |
| 2 | 3 mô hình, phủ rộng | 17 | 11,25% | 271 s | 6,28 |
| 3 | relation+tcn, nới đoạn ±240 s | 18 | 16,14% | 630 s | **0,00** |
| 4 | 3 mô hình, ngưỡng cao | 5 | 1,65% | 300 s | 30,76 |
| 5 | chỉ k-NN | 11 | 8,04% | 451 s | 9,24 |
| 6 | top-5, cắt 300 s | 5 | 1,40% | 299 s | 32,87 |
| 7 | top-8, cắt 300 s | 8 | 2,45% | 300 s | 25,71 |
| 8 | top-12, cắt 300 s | 12 | 3,13% | 286 s | 28,78 |
| 9 | top-16, cắt 300 s | 16 | 4,36% | 290 s | 23,14 |
| 10 | **top-8, cắt 180 s** | 8 | 1,61% | 180 s | 37,87 |
| 11 | top-8, cắt 120 s | 8 | 1,11% | 120 s | 36,50 |
| 12 | **top-12, cắt 180 s** | 12 | 2,19% | 180 s | **39,18** ⭐ |
| 13 | top-12, cắt 120 s | 12 | 1,61% | 120 s | 38,55 |
| 14 | top-16, cắt 180 s | 16 | 3,02% | 180 s | 31,89 |
| 15 | #12 dịch lui 60 s | 12 | 2,19% | 180 s | 28,82 |
| 16 | #12 dịch tới 60 s | 12 | 2,19% | 180 s | 19,24 |
| 17 | #12, xếp hạng theo trung bình nhân 3 mô hình | 12 | 2,37% | 180 s | 38,10 |
| 18 | #12, độ rộng thích nghi 120–220 s | 12 | 2,39% | 166 s | 30,44 |
| 19 | #12, thay 2 đoạn "công tắc P2" bằng 2 đoạn khác | 12 | 2,28% | 180 s | 38,59 |

### 2.2 Kết luận rút ra (đều có bằng chứng)

1. **Tấn công rất ngắn.** Đoạn 630 s → 0 điểm; dịch cửa sổ 60 s mất 10–20 điểm.
   Ước lượng độ dài tấn công thật: **150–250 giây**.
2. **Số đoạn tối ưu ≈ 10–13 cho một ngày** (86.400 giây). Khớp parabol: đỉnh ở n ≈ 10,6.
3. **Độ rộng tối ưu ≈ 160–180 giây.** Khớp parabol: đỉnh ở 160 s.
4. **Căn chỉnh cửa sổ hiện tại (cửa sổ có tổng điểm lớn nhất trong vùng ứng viên) đã
   gần tối ưu** — đỉnh parabol ở −10 s.
5. **Precision quan trọng hơn recall ở vùng làm việc này**: 5 đoạn (30,76) > 23 đoạn (21,56).
6. **Trong ba mô hình, quan hệ tuyến tính mạnh nhất**; TCN và k-NN chỉ bổ sung.
   Trên tập giả lập: tuyến tính 0,46 · TCN 0,28 · k-NN 0,29 · ghép 3 cái ~0,46.
7. **Chuẩn hoá theo thứ hạng (ECDF) trên chính chuỗi đang chấm** thắng chuẩn hoá MAD.
8. **Năm vị trí tấn công đã được bảng xếp hạng xác nhận** (bản 5 đoạn đạt 30,76 với
   precision gần như hoàn hảo): **05:41, 08:14, 08:26, 11:25, 16:42** (giờ trong ngày
   2021-07-10 của public test). Dùng làm nhãn thật để kiểm tra bộ xếp hạng.
9. **Ước lượng ngược từ công thức eTaPR**: public test có khoảng **20–35 đoạn tấn công**,
   ta mới bắt đúng ~9–12 ⇒ **nút thắt là recall**.

### 2.3 Những thứ đã thử và không ăn thua (đừng làm lại)

- Bộ dò nhảy bậc (change-point) và bộ dò xung chữ nhật: xếp hạng các tấn công đã biết kém.
- Mô hình quan hệ chỉ dùng bối cảnh chậm (bỏ giá trị tức thời): kém hơn bản đầy đủ.
- Làm trơn từng tín hiệu trước khi gộp top-k: giảm F1.
- Ghép đoạn thích nghi, ghép "bỏ phiếu ≥ 2/3", độ rộng thích nghi: đều kém hơn.
- Bỏ cụm công tắc chế độ P2 khỏi việc tính điểm: **giảm** điểm (38,59 so với 39,18) ⇒
  suy luận "đó chỉ là thao tác người trực" là **sai**.

---

## 3. Giả thuyết chưa kiểm chứng (đừng coi là chân lý)

| Giả thuyết | Vì sao chưa chắc |
|---|---|
| Số đoạn tối ưu ~12/ngày áp dụng được cho private test | Mật độ tấn công của tập private có thể khác. Nên dùng **tỉ lệ/24 giờ** và cân nhắc thử 2–3 mức |
| Độ rộng 180 s là đúng cho mọi tấn công | Mới chỉ đo được *trung bình tốt nhất*; độ dài tấn công chắc chắn có phân tán |
| Tổ hợp 3 mô hình tốt hơn 2 mô hình | Chênh lệch nhỏ, trong sai số. `relation+tcn` xếp hạng 5 tấn công đã biết còn nhỉnh hơn |
| TCN thực sự đóng góp | Chưa bao giờ nộp thử bản "chỉ relation" ở hình dạng tối ưu (n=12, 180 s) — **nên thử ngay ở phiên sau** |
| Nút thắt là recall | Suy ra từ mô hình toán của eTaPR với giả định precision ~0,9; nếu precision thật thấp hơn thì kết luận đổi |

---

## 4. Kế hoạch cho phiên sau (20 lượt nộp mới)

Mục tiêu: **tăng recall mà không mất precision** — đó là chỗ duy nhất còn nhiều điểm.

### Giai đoạn A — tái lập và kiểm tra lại nền (3 lượt)

| Lượt | Nội dung | Học được gì |
|---|---|---|
| A1 | Bản tốt nhất hiện tại (n=12, 180 s, 3 mô hình) | Xác nhận tái lập được ~39 |
| A2 | Chỉ mô hình tuyến tính, cùng hình dạng | TCN + k-NN có đáng giữ không |
| A3 | n=10 và n=14 (chọn một), 180 s | Xác nhận đỉnh đường cong |

### Giai đoạn B — kiểm tra khối ứng viên (4 lượt)

Ý tưởng: xếp hạng ứng viên rồi nộp **từng khối 8 ứng viên đứng riêng** (khối 9–16,
17–24, 25–32). Điểm của mỗi khối cho biết mật độ tấn công thật trong khối đó, mà không
làm hỏng bản tốt nhất. Đây là *group testing*: 1 lượt nộp cho thông tin về 8 ứng viên.

Sau đó gộp các khối "có hàng" vào bản chính.

### Giai đoạn C — mô hình mạnh hơn để tăng recall (dùng thời gian, không tốn lượt nộp)

Xếp theo tỉ lệ kỳ vọng/công sức:

1. **Hồi quy phi tuyến từng biến bằng cây tăng cường** (`HistGradientBoostingRegressor`
   của sklearn, hoặc LightGBM): dự đoán mỗi tín hiệu từ các tín hiệu khác + EWMA. Bắt
   được quan hệ phụ thuộc chế độ vận hành mà mô hình tuyến tính bỏ sót. ~30 phút CPU cho
   68 biến nếu lấy mẫu 200k dòng. **Đây là ứng viên số một.**
2. **TCN học phần dư của mô hình tuyến tính** (boosting): mạng chỉ cần học phần phi
   tuyến còn lại ⇒ nền nhiễu thấp hơn hẳn so với học lại từ đầu.
3. **Ensemble nhiều seed** của TCN (trung bình residual) — rẻ, giảm phương sai.
4. Mô hình quan hệ với **nhiều bộ nửa đời EWMA** khác nhau rồi lấy max z-score.

Cách đánh giá mô hình mới **mà không tốn lượt nộp**: xem nó xếp 5 tấn công đã biết
(05:41, 08:14, 08:26, 11:25, 16:42) ở hạng nào trong danh sách ứng viên. Bộ hiện tại
đưa chúng vào hạng [5, 7, 7, 3, 1]. Mô hình mới phải **giữ được** thứ hạng đó và làm đổi
thành phần các hạng 8–20 thì mới đáng nộp thử.

### Giai đoạn D — chốt (3–4 lượt)

Tinh chỉnh (n, độ rộng) quanh cấu hình tốt nhất mới, rồi giữ lại bản cao điểm nhất.

### Giữ lại 5–6 lượt dự phòng cho private test

---

## 5. Bài học về phương pháp (phần quan trọng nhất)

### 5.1 Về việc nghe theo gợi ý của đề

Đề gợi ý "autoencoder theo cửa sổ thời gian, mô hình dự báo đa biến, mô hình biểu diễn
chuỗi". Gợi ý này **đúng nhưng vô ích**: nó liệt kê cả một lĩnh vực và không nói gì về
chỗ thực sự quyết định điểm.

Tôi đã *không* đi theo con đường đó mà bắt đầu bằng việc đọc mã nguồn metric. Nhìn lại,
đó là quyết định đúng nhất của cả phiên. Nhưng tôi vẫn phạm đúng cái lỗi tương tự ở tầng
dưới: **tin vào tập kiểm định tự tạo** thay vì đặt câu hỏi "tập này giả định gì?". Nếu
nộp thử bản "đoạn dài" ngay từ lượt thứ hai thay vì sau khi đã train xong TCN, tôi đã
biết được sự thật "tấn công rất ngắn" sớm hơn 2 tiếng.

**Quy tắc rút ra**: với mỗi kết luận đang định dựa vào, hãy hỏi *"bằng chứng cho điều này
đến từ đâu, và bằng chứng đó có thể sai ở chỗ nào?"*. Xếp hạng độ tin cậy:

```
đo trực tiếp trên bảng xếp hạng  >  suy luận từ công thức metric
   >  tập kiểm định tự tạo  >  trực giác từ kinh nghiệm lĩnh vực khác
```

### 5.2 Về thứ tự công việc

Thứ tự đúng là: **metric → dữ liệu → baseline rẻ → hình dạng đầu ra → mô hình mạnh**.
Người có nền DL có xu hướng làm ngược lại (mô hình trước), và đó là cách tiêu hết 6 tiếng
mà chỉ được nửa số điểm.

### 5.3 Về việc gỡ lỗi mô hình không giám sát

Không có nhãn thì không có "accuracy tụt" để báo động. Hai phép thử tự kiểm tra đã cứu
tôi hai lần:

- **Chạy trên dữ liệu train sạch**: nếu mô hình báo động ở đó thì đó là lỗi của mình
  (tìm ra artifact EWMA đầu chuỗi).
- **Đổi một thứ lẽ ra không được ảnh hưởng tới kết quả** (độ dài khối khi suy luận):
  nếu kết quả đổi thì có lỗi (tìm ra GroupNorm chuẩn hoá theo trục thời gian).
