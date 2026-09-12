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

## 3. Cấu trúc mã nguồn

```
icsad/                 thư viện
  data.py              đọc zip/thư mục, Scaler min-max, gom cụm tín hiệu tương quan
  features.py          ma trận đặc trưng [v, ewma quá khứ/tương lai] theo khối + reflect pad
  relation.py          mô hình Ridge "mỗi tín hiệu từ các tín hiệu khác"
  tcn.py               TCN che-và-tái tạo (PyTorch, CPU/GPU)
  scoring.py           chuẩn hoá residual (mad/rank), gộp top-k, boundary guard
  postprocess.py       làm trơn, ngưỡng trễ, ghép khe hở, bỏ đoạn ngắn
  synth.py             sinh tấn công giả lập
  etapr.py             metric eTaPR (bản vector hoá)
  tune.py              duyệt lưới tham số hậu xử lý theo eTaPR
  pipeline.py          các bước dùng chung giữa các script
scripts/
  train_relation.py    huấn luyện mô hình tuyến tính           (CPU, ~1 phút)
  train_tcn.py         huấn luyện TCN                          (GPU/CPU)
  score.py             residual của mô hình tuyến tính
  score_tcn.py         residual của TCN
  tune_postprocess.py  chọn chuẩn hoá + ngưỡng + hậu xử lý
  make_submission.py   xuất predictions.csv
  run_all.py           chạy tuần tự toàn bộ
notebooks/colab_train.ipynb   bản chạy trên GPU Colab
tests/                 kiểm thử nhanh (eTaPR, hậu xử lý, dữ liệu)
```

## 4. Chạy trên máy cá nhân

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

## 5. Chạy trên Colab

Mở `notebooks/colab_train.ipynb` (Runtime → T4 GPU). Notebook tự mount Drive, dò `training.zip`
trong Drive, clone repo này, huấn luyện cả hai mô hình, chọn ngưỡng và xuất `predictions.csv`
về lại Drive.

## 6. Ghi chú về dữ liệu

`release/*.zip` **không được đẩy lên GitHub** (xem `.gitignore`); dữ liệu nằm sẵn trong Google
Drive. Mọi kết quả trung gian (`outputs/`, `data/`) cũng bị bỏ qua.
