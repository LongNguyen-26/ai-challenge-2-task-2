"""Tạo notebooks/colab_train.ipynb từ danh sách cell (dễ bảo trì hơn viết JSON tay)."""
import json
from pathlib import Path

REPO = "https://github.com/LongNguyen-26/ai-challenge-2-task-2.git"
OUT = Path(r"C:\CS_Major\Contest_2026\Olympic_AI\AI_Challenge_2\Task_2\notebooks\colab_train.ipynb")

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").split("\n")}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").split("\n")}

cells = []

cells.append(md(f"""
# ICS Intrusion Detection - Task 2 (train trên GPU Colab)

Quy trình đầy đủ: nạp dữ liệu từ Google Drive → huấn luyện 2 mô hình phát hiện bất thường
(**quan hệ tuyến tính** + **TCN che-và-tái tạo**) → chọn ngưỡng bằng **tấn công giả lập**
→ xuất `predictions.csv` theo đúng định dạng nộp bài.

- Repo: {REPO}
- Runtime cần: **GPU** (Runtime → Change runtime type → T4 GPU). Toàn bộ notebook chạy ~25–40 phút.
- Dữ liệu: đặt `training.zip`, `public_test.zip` (và `private_test.zip` khi có) ở đâu đó trong
  Google Drive của bạn; ô "tìm dữ liệu" bên dưới sẽ tự dò.
"""))

cells.append(md("## 1. Kiểm tra GPU"))
cells.append(code("""
!nvidia-smi
import torch
print('torch', torch.__version__, '| CUDA khả dụng:', torch.cuda.is_available())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f'{p.name}, VRAM {p.total_memory/1024**3:.1f} GB')
"""))

cells.append(md("## 2. Mount Google Drive + tìm dữ liệu"))
cells.append(code("""
from google.colab import drive
drive.mount('/content/drive')
"""))
cells.append(code("""
# Tự dò 3 tệp dữ liệu trong Drive. Nếu biết sẵn thư mục, gán trực tiếp DATA_DIR.
import os, glob, shutil, time

DATA_DIR = None           # ví dụ: '/content/drive/MyDrive/AI_Challenge_2/Task_2'
WANTED = ['training.zip', 'public_test.zip', 'private_test.zip']

if DATA_DIR is None:
    t0 = time.time()
    hits = glob.glob('/content/drive/MyDrive/**/training.zip', recursive=True)
    print(f'tìm {time.time()-t0:.0f}s ->', hits)
    assert hits, 'Không thấy training.zip trong Drive - hãy gán DATA_DIR thủ công.'
    DATA_DIR = os.path.dirname(hits[0])
print('DATA_DIR =', DATA_DIR)

# Copy về đĩa local của Colab: đọc zip từ Drive rất chậm.
LOCAL = '/content/data'
os.makedirs(LOCAL, exist_ok=True)
for name in WANTED:
    src = os.path.join(DATA_DIR, name)
    if os.path.exists(src) and not os.path.exists(os.path.join(LOCAL, name)):
        t0 = time.time()
        shutil.copy(src, LOCAL)
        print(f'copy {name}: {os.path.getsize(src)/1e6:.0f} MB trong {time.time()-t0:.0f}s')
!ls -la /content/data
"""))

cells.append(md("## 3. Lấy mã nguồn"))
cells.append(code(f"""
import os
REPO_URL = '{REPO}'
if not os.path.exists('/content/task2'):
    !git clone -q $REPO_URL /content/task2
else:
    !cd /content/task2 && git pull -q
%cd /content/task2
!pip -q install polars
!ls
"""))

cells.append(md("""
## 4. Huấn luyện mô hình quan hệ tuyến tính (CPU, ~1 phút)

Mỗi tín hiệu được hồi quy từ *các tín hiệu khác* (giá trị hiện tại + EWMA quá khứ/tương lai),
loại trừ các tín hiệu trùng lặp với chính nó. Residual lớn kéo dài = quan hệ vật lý bị gãy.
"""))
cells.append(code("""
!python scripts/train_relation.py --data-dir /content/data --out-dir outputs
"""))

cells.append(md("""
## 5. Huấn luyện TCN che-và-tái tạo (GPU)

Mạng tích chập giãn hai chiều: che một cụm tín hiệu rồi tái tạo lại nó từ các tín hiệu còn lại
và bối cảnh ±510 giây. Đây là phiên bản phi tuyến của mô hình ở bước 4.

Tham số mặc định (160 kênh, cửa sổ 512, 20 epoch × 1500 bước) chiếm ~1.5 GB VRAM.
Muốn mạnh hơn: tăng `--channels 224 --epochs 30`.
"""))
cells.append(code("""
!python scripts/train_tcn.py --data-dir /content/data --out-dir outputs \\
    --device cuda --channels 160 --window 512 --batch-size 32 \\
    --epochs 20 --steps-per-epoch 1500 --name tcn
"""))

cells.append(md("""
## 6. Tính residual cho tập kiểm định giả lập và cho test

`synth` = tệp train chưa dùng khi huấn luyện (train6) + các đoạn tấn công giả lập có nhãn;
dùng để chọn ngưỡng/hậu xử lý một cách có cơ sở thay vì chọn tay.
"""))
cells.append(code("""
DATASET = 'public_test'   # đổi thành 'private_test' khi có dữ liệu private

!python scripts/score.py     --data-dir /content/data --dataset synth --seeds 0 1 2
!python scripts/score.py     --data-dir /content/data --dataset $DATASET
!python scripts/score_tcn.py --data-dir /content/data --dataset synth --seeds 0 1 2 --device cuda
!python scripts/score_tcn.py --data-dir /content/data --dataset $DATASET --device cuda
"""))

cells.append(md("""
## 7. Chọn ngưỡng + hậu xử lý theo eTaPR

Duyệt lưới: cách chuẩn hoá residual (`mad` / `rank`), số tín hiệu gộp (`topk`), cửa sổ làm trơn,
ngưỡng, độ dài đoạn tối thiểu, khoảng cách ghép đoạn — chấm bằng chính metric eTaPR của ban tổ chức
(θp = 0.5, θr = 0.1) trên tập giả lập.
"""))
cells.append(code("""
!python scripts/tune_postprocess.py \\
    --residuals "outputs/res_relation_synth_seed{seed}.npy" "outputs/res_tcn_synth_seed{seed}.npy" \\
    --seeds 0 1 2 --out outputs/post_params.json
"""))

cells.append(md("## 8. Xuất predictions.csv"))
cells.append(code("""
!python scripts/make_submission.py --data-dir /content/data --dataset $DATASET \\
    --residuals outputs/res_relation_$DATASET.npy outputs/res_tcn_$DATASET.npy \\
    --params outputs/post_params.json --out predictions.csv
!head -3 predictions.csv && wc -l predictions.csv
"""))

cells.append(md("""
## 9. Xem lại điểm bất thường theo thời gian

In ra từng đoạn dự đoán kèm những tín hiệu đóng góp nhiều nhất - rất hữu ích để kiểm tra
xem mô hình đang bắt đúng thứ có ý nghĩa vật lý hay chỉ đang bắt nhiễu.
"""))
cells.append(code("""
!python scripts/plot_scores.py --dataset $DATASET \
    --residuals outputs/res_relation_$DATASET.npy outputs/res_tcn_$DATASET.npy \
    --params outputs/post_params.json

from IPython.display import Image, display
display(Image(f'outputs/score_{DATASET}.png'))
"""))

cells.append(md("## 10. Lưu kết quả về Drive"))
cells.append(code("""
import shutil, os
SAVE_DIR = os.path.join(DATA_DIR, 'task2_outputs')
os.makedirs(SAVE_DIR, exist_ok=True)
for f in ['predictions.csv', 'outputs/post_params.json', 'outputs/tcn.pt',
          'outputs/relation.npz', 'outputs/scaler.json', 'outputs/clusters.json']:
    if os.path.exists(f):
        shutil.copy(f, SAVE_DIR)
        print('đã lưu', f)
print('->', SAVE_DIR)
"""))

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
OUT.parent.mkdir(parents=True, exist_ok=True)
# thêm ký tự xuống dòng ở cuối mỗi dòng nguồn (đúng chuẩn nbformat)
for c in nb["cells"]:
    c["source"] = [ln + "\n" for ln in c["source"][:-1]] + [c["source"][-1]]
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
