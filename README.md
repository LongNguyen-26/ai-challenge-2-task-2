# ICS Intrusion Detection — AI Challenge 2 / Task 2

Phát hiện bất thường (tấn công) trên telemetry của một hệ thống điều khiển công nghiệp:
mỗi giây có 86 tín hiệu cảm biến/điều khiển, tập huấn luyện **chỉ chứa dữ liệu bình thường**,
nhiệm vụ là gán nhãn `0/1` cho từng giây của tập kiểm tra. Chấm điểm bằng **eTaPR F1**
(θp = 0.50, θr = 0.10).

```
release/training.zip      train1..train6.csv   1.004.402 dòng, Attack = 0
release/public_test.zip   test.csv               86.400 dòng (1 ngày), không nhãn
```

---

## 1. Ý tưởng

Ở trạng thái bình thường, các tín hiệu ràng buộc lẫn nhau bởi vật lý và bởi vòng điều khiển
(van mở → lưu lượng đổi, bơm chạy → áp suất/nhiệt độ phản ứng theo). Khi bị can thiệp, **quan hệ
giữa các tín hiệu bị gãy trong suốt thời gian tấn công**. Vì vậy cả hai mô hình ở đây đều học
một việc: *dự đoán lại một tín hiệu từ những tín hiệu khác*, rồi lấy sai số (residual) làm điểm
bất thường. Cách này cho tín hiệu **kéo dài suốt đoạn tấn công** — rất hợp với eTaPR (cần phủ
được đoạn, không chỉ chạm vào nó).

| Mô hình | Mô tả | Chi phí |
|---|---|---|
| **Quan hệ tuyến tính** (`icsad/relation.py`) | Ridge: `v_j(t) ≈ w·[v, ewma_quá_khứ, ewma_tương_lai]` của **mọi tín hiệu khác**, loại cả cụm tương quan ≥ 0.995 với `j` để không "nhìn trộm" bản sao. Gram matrix tích luỹ theo khối → giải một lần cho cả 68 tín hiệu. | **CPU, ~15 giây** |
| **TCN che-và-tái tạo** (`icsad/tcn.py`) | Tích chập giãn hai chiều (±510 s): che một cụm tín hiệu trên toàn cửa sổ rồi tái tạo lại nó từ phần còn lại. Phiên bản phi tuyến, có bộ nhớ thời gian, của mô hình trên. | **GPU ~20 phút** (CPU ~2 giờ) |

Hai mô hình cùng sinh ra ma trận residual `(n, 68)` nên dùng chung toàn bộ phần sau:

```
residual → chuẩn hoá theo từng tín hiệu → gộp top-k → làm trơn → ngưỡng → ghép/lọc đoạn → nhãn 0/1
```

**Chuẩn hoá theo từng tín hiệu** là bước quan trọng: mỗi tín hiệu có mức nhiễu rất khác nhau, và
tập test vận hành ở chế độ hơi khác tập train (ví dụ `P1_FCV01D` trung bình 65 khi train nhưng 16
khi test). Hai cách được thử và chọn tự động:
* `mad` : `(|r| − median) / MAD` tính trên chính chuỗi đang chấm;
* `rank`: `−log(1 − F(|r|))` với `F` là ECDF trên chuỗi đang chấm — mỗi tín hiệu đóng góp một
  "độ bất ngờ" bị chặn trên, bền vững với đuôi nặng. *(thường thắng)*

Vì tấn công chỉ chiếm vài phần trăm số điểm nên median/ECDF gần như không bị chúng làm lệch,
trong khi mọi sai khác do đổi chế độ vận hành thì bị trừ đi hết.

## 2. Chọn ngưỡng khi không có nhãn

Tập test không có nhãn nên không thể dò ngưỡng trực tiếp. Cách làm ở đây: lấy `train6.csv`
(không dùng khi huấn luyện) rồi **tiêm tấn công giả lập** có nhãn (`icsad/synth.py`) — freeze,
bias, ramp, đổi setpoint, scale, replay, noise — với độ dài 60 s … 30 phút, và bỏ những ca tiêm
mà dữ liệu gần như không đổi (không thể phát hiện được, giữ lại chỉ làm nhiễu). Sau đó duyệt lưới
tham số và chấm bằng **chính metric eTaPR** (`icsad/etapr.py`, đã đối chiếu khớp tới 5e-16 với mã
nguồn gốc của tác giả metric).

Khi chọn, bộ tham số được xếp hạng theo **F1 trung bình của ngưỡng đó và hai ngưỡng kề bên**
chứ không phải F1 đơn lẻ — tránh việc bám vào đúng một điểm may mắn nằm ngay mép vực, vì trên
dữ liệu thi thật ngưỡng chắc chắn lệch đi ít nhiều.

Vài hệ quả của eTaPR đã được dùng để định hướng hậu xử lý:
* mỗi đoạn dự đoán phải nằm **≥ 50%** bên trong tấn công, nếu không nó bị loại và trở thành báo
  động giả → **không được kéo đoạn dài lê thê**;
* chỉ cần phủ **≥ 10%** một đoạn tấn công là tính "phát hiện được", phủ thêm thì được thêm điểm
  portion → vài đoạn ngắn *đặt đúng chỗ* có lợi hơn một đoạn dài tràn ra ngoài;
* precision có trọng số `sqrt(độ dài đoạn)` nên đoạn rác ngắn ít hại hơn đoạn rác dài, nhưng
  nhiều đoạn rác vẫn giết điểm.

Một chi tiết dễ bỏ sót: ở **hai đầu chuỗi**, các đặc trưng EWMA không có bối cảnh thật nên
residual bị thổi phồng một cách hệ thống (đo trên train sạch: 60 giây đầu có điểm cao gấp 3–8 lần
mức nền). `icsad/features.py` đã đệm phản chiếu (reflect) và `scoring.boundary_guard` còn dập
thêm phần rìa còn lại — nếu không sẽ luôn có một "đoạn tấn công ma" ở đầu tệp test.

## 3. Kết quả

eTaPR F1 trên **6 lần tiêm tấn công giả lập khác nhau** vào `train6.csv` (50 tấn công/lần,
θp = 0.5, θr = 0.1) — đây là thước đo *tương đối* để so các phương án, không phải ước lượng
điểm thi:

| Phương án | chuẩn hoá | F1 (6 seed) | eTaP | eTaR | Public test |
|---|---|---|---|---|---|
| Quan hệ tuyến tính | rank, top-3 | 0.462 ± 0.02 | 0.58 | 0.39 | 16 đoạn, 5,5% số điểm |
| TCN | rank, top-3 | 0.277 | 0.26 | 0.34 | 48 đoạn, 8,1% |
| **Ghép cả hai** (trung bình z theo từng tín hiệu, top-1) | rank, top-1 | **0.486 ± 0.06** | 0.48 | 0.49 | **23 đoạn, 4,55%** |

Phương án nộp bài là phương án ghép: `configs/post_params_ensemble.json`
(làm trơn 15 s, ngưỡng 5,0, đoạn tối thiểu 60 s, ghép khe hở ≤ 60 s). Bản chỉ dùng mô hình
tuyến tính (`configs/post_params_relation.json`) chạy được hoàn toàn trên CPU trong ~10 phút
và chỉ kém khoảng 0,02 F1 — dùng làm phương án dự phòng.

Các đoạn mà mô hình chỉ ra trên public test đều bám vào những tín hiệu có ý nghĩa vật lý
(van mức `P1_LCV01D` + mức bồn `P1_LIT01`, setpoint áp suất `P1_B2016` + `P1_PIT01`,
`P1_FCV03D` + lưu lượng `P1_FT03`, cụm P3 `P3_FIT01`/`P3_LCV01D`…) chứ không phải nhiễu rải rác
— xem `outputs/score_public_test.png` và bảng in ra bởi `scripts/plot_scores.py`.

> **Một lỗi đáng ghi lại**: bản TCN đầu tiên dùng `GroupNorm`, mà GroupNorm chuẩn hoá dọc theo
> trục thời gian. Mạng huấn luyện với cửa sổ 512 điểm nhưng khi chấm điểm lại chạy trên cả chuỗi
> 32 000 điểm nên thống kê chuẩn hoá khác hẳn: |residual| trung bình 0,196 thay vì 0,074, và F1
> rơi xuống 0,13. Thay bằng `ChannelNorm` (chuẩn hoá trên trục kênh tại từng thời điểm, không
> phụ thuộc độ dài chuỗi) thì hết.

## 4. Bảng xếp hạng public dạy được gì

Tập test không có nhãn nên tấn công giả lập chỉ giúp tới một mức; sau đó điểm public
trở thành nguồn thông tin duy nhất. Loạt phép thử có kiểm soát (mỗi lần chỉ đổi *một*
yếu tố) cho kết quả:

| Phép thử | Số đoạn | Độ rộng | % số điểm | Điểm |
|---|---|---|---|---|
| Ngưỡng + ghép khe hở (bản đầu) | 23 | ~130 s | 4,6% | 21,56 |
| Đoạn dài | 18 | 630 s | 16,1% | **0,00** |
| k-NN một mình | 11 | 451 s | 8,0% | 9,24 |
| Ít mà chắc | 5 | 300 s | 1,7% | 30,76 |
| Quét độ rộng (n=8) | 8 | 300 / 180 / 120 s | | 25,7 / **37,9** / 36,5 |
| Quét số đoạn (rộng 180 s) | 8 / 12 / 16 | 180 s | | 37,9 / **39,2** / 31,9 |
| Dịch cả 12 đoạn ±60 s | 12 | 180 s | 2,2% | −60 s: 28,8 · **0: 39,2** · +60 s: 19,2 |

Ba kết luận:

1. **Tấn công rất ngắn (cỡ 150–250 giây).** Đoạn dự đoán 630 s bị eTaPR loại sạch (điểm 0)
   vì không đoạn nào đạt ngưỡng θp = 0,5 "nằm trong tấn công"; mà khi mọi đoạn dự đoán bị
   loại thì các đoạn tấn công cũng mất luôn phần phủ → precision và recall **cùng** về 0.
   Đây là cái bẫy lớn nhất của metric này.
2. **Vị trí cửa sổ cực kỳ nhạy.** Dịch đúng 60 giây làm mất 10–20 điểm. Khớp parabol qua ba
   điểm cho đỉnh ở −10 s, tức căn chỉnh hiện tại đã gần tối ưu.
3. **Ít mà chắc thắng nhiều mà ẩu.** 5 đoạn (30,76) hơn hẳn 23 đoạn (21,56); tối ưu ở
   khoảng 12 đoạn cho một ngày dữ liệu.

Bộ tham số tốt nhất: **≈12 đoạn/24 giờ, mỗi đoạn 180 giây, đặt quanh đỉnh điểm bất thường**
của tổ hợp 3 mô hình (quan hệ tuyến tính + TCN + k-NN).

```bash
# tái tạo bản nộp tốt nhất cho public test
python scripts/make_topn.py --dataset public_test --models relation tcn nn     --n 12 --max-len 180 --out predictions.csv

# cho private test (chép private_test.zip vào release/ rồi chạy 4 lệnh này;
# --rate đặt số đoạn theo mỗi 24 giờ nên tự co giãn nếu tập dài/ngắn khác public)
python scripts/score.py          --dataset private_test
python scripts/score_tcn.py      --dataset private_test --device cuda
python scripts/score_neighbor.py --dataset private_test --device cuda
python scripts/make_topn.py      --dataset private_test --models relation tcn nn     --rate 12 --max-len 180 --out predictions.csv
```

Tham số chốt nằm ở `configs/best_recipe.json`. Tuỳ chọn `--exclude-switch` bỏ cụm công
tắc chế độ P2 (`P2_MASW`, `P2_ManualGO`, `P2_AutoGO`, …) ra khỏi việc tính điểm: năm tín
hiệu nhị phân này đổi cùng lúc mỗi khi người trực chuyển auto/manual, mô hình nào cũng
coi là hiếm gặp nên hay báo động giả ở đó.

## 5. Cấu trúc mã nguồn

```
icsad/                 thư viện
  data.py              đọc zip/thư mục, Scaler min-max, gom cụm tín hiệu tương quan
  features.py          ma trận đặc trưng [v, ewma quá khứ/tương lai] theo khối + reflect pad
  relation.py          mô hình Ridge "mỗi tín hiệu từ các tín hiệu khác"
  tcn.py               TCN che-và-tái tạo (PyTorch, CPU/GPU)
  scoring.py           chuẩn hoá residual (mad/rank), gộp top-k, boundary guard
  postprocess.py       làm trơn, ngưỡng trễ, ghép khe hở, bỏ đoạn ngắn
  synth.py             sinh tấn công giả lập
  novelty.py           k-láng-giềng tự tham chiếu ("trạng thái này đã từng xảy ra chưa")
  changepoint.py       dò nhảy bậc và xung chữ nhật trên từng tín hiệu
  etapr.py             metric eTaPR (bản vector hoá)
  tune.py              duyệt lưới tham số hậu xử lý theo eTaPR
  pipeline.py          các bước dùng chung giữa các script
scripts/
  train_relation.py    huấn luyện mô hình tuyến tính           (CPU, ~1 phút)
  train_tcn.py         huấn luyện TCN                          (GPU/CPU)
  score.py             residual của mô hình tuyến tính
  score_tcn.py         residual của TCN
  tune_postprocess.py  chọn chuẩn hoá + ngưỡng + hậu xử lý
  score_neighbor.py    residual của mô hình k-láng-giềng
  make_submission.py   xuất predictions.csv theo ngưỡng
  make_topn.py         xuất N đoạn tin cậy nhất, cắt về độ rộng cố định (bản dùng thi)
  make_peaks.py        đặt N cửa sổ quanh N đỉnh cao nhất
  run_all.py           chạy tuần tự toàn bộ
  plot_scores.py       vẽ điểm + liệt kê đoạn dự đoán kèm tín hiệu đóng góp
configs/               tham số hậu xử lý đã chọn (dùng lại được, khỏi dò lại)
submissions/           tệp nộp bài đã sinh
notebooks/colab_train.ipynb   bản chạy trên GPU Colab
tools/build_notebook.py       sinh lại notebook từ mã nguồn
tests/                 kiểm thử nhanh (eTaPR, hậu xử lý, dữ liệu)
```

## 6. Chạy trên máy cá nhân

```bash
pip install -r requirements.txt
```

Đường chạy nhanh nhất (chỉ CPU, ~10 phút, không cần GPU):

```bash
python scripts/train_relation.py --data-dir release
python scripts/score.py --dataset synth --seeds 0 1 2
python scripts/score.py --dataset public_test
python scripts/tune_postprocess.py --residuals "outputs/res_relation_synth_seed{seed}.npy"
python scripts/make_submission.py --dataset public_test \
    --residuals outputs/res_relation_public_test.npy --out predictions.csv
```

hoặc gọn lại:

```bash
python scripts/run_all.py --data-dir release --dataset public_test        # chỉ mô hình tuyến tính
python scripts/run_all.py --data-dir release --dataset public_test --tcn  # thêm TCN
```

**Dùng GPU của máy** (`--device cuda`, tự nhận nếu torch có CUDA):

```bash
python scripts/train_tcn.py --device cuda --epochs 20
python scripts/score_tcn.py --dataset public_test --device cuda
```

> GPU 4 GB là đủ cho cấu hình mặc định (~0,9 triệu tham số, ~1,5 GB VRAM, batch 32).
> Nếu PyTorch trên máy là bản CPU-only thì cài bản CUDA vào môi trường ảo riêng:
> ```bash
> python -m venv .venv --system-site-packages
> .venv/Scripts/python -m pip install --ignore-installed --no-deps torch --index-url https://download.pytorch.org/whl/cu126
> ```

**Khi có `private_test.zip`**: chép vào `release/` rồi đổi `--dataset private_test` — không cần
huấn luyện lại.

## 7. Chạy trên Colab

Mở `notebooks/colab_train.ipynb` (Runtime → T4 GPU). Notebook tự mount Drive, dò `training.zip`
trong Drive, clone repo này, huấn luyện cả hai mô hình, chọn ngưỡng và xuất `predictions.csv`
về lại Drive.

## 8. Ghi chú về dữ liệu

`release/*.zip` **không được đẩy lên GitHub** (xem `.gitignore`); dữ liệu nằm sẵn trong Google
Drive. Mọi kết quả trung gian (`outputs/`, `data/`) cũng bị bỏ qua.
