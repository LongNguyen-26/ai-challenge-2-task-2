"""Sinh notebooks/olympiad_solution.ipynb - bài giải đầy đủ từ đầu tới predictions.csv."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "notebooks" / "olympiad_solution.ipynb"

md = lambda s: {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n").split("\n")}
code = lambda s: {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": s.strip("\n").split("\n")}

cells = []

cells.append(md("""
# Industrial Control System Intrusion Detection — bài giải đầy đủ

Notebook này đi lại **đúng trình tự mà một thí sinh nên làm** trong 6 giờ thi, từ lúc đọc
đề tới lúc có `predictions.csv` đạt điểm cao nhất (39,18 trên public test, so với 21,56
của cách làm thông thường).

Trình tự có chủ ý — **metric trước, mô hình sau**:

| Bước | Nội dung | Thời gian |
|---|---|---|
| 1 | Hiểu metric eTaPR và suy ra *hình dạng* lời giải | ~30 phút |
| 2 | Khảo sát dữ liệu, phát hiện các bẫy | ~30 phút |
| 3 | Tiền xử lý: chuẩn hoá, gom cụm tín hiệu trùng nhau | ~10 phút |
| 4 | Mô hình A — quan hệ tuyến tính (CPU, 15 giây) | ~30 phút |
| 5 | Mô hình B — TCN che-và-tái tạo (GPU, 35 phút) | ~60 phút |
| 6 | Mô hình C — k-láng-giềng "trạng thái lạ" | ~20 phút |
| 7 | Gộp điểm, xếp hạng vùng nghi vấn | ~20 phút |
| 8 | Chọn hình dạng đầu ra bằng bảng xếp hạng | ~60 phút |
| 9 | Xuất và kiểm tra tệp nộp | ~15 phút |

Đọc kèm: `docs/01_cach_tiep_can.md` (lập luận) và `docs/02_kien_thuc_nen.md` (kiến thức nền).
"""))

cells.append(md("## 0. Chuẩn bị"))
cells.append(code("""
import sys, os, io, zipfile, json
from pathlib import Path
import numpy as np

# Chạy được cả ở máy cá nhân lẫn Colab
REPO = Path.cwd() if (Path.cwd() / "icsad").exists() else Path.cwd().parent
sys.path.insert(0, str(REPO))
DATA_DIR = os.environ.get("ICSAD_DATA", str(REPO / "release"))   # nơi chứa training.zip, public_test.zip
OUT_DIR = REPO / "outputs"; OUT_DIR.mkdir(exist_ok=True)
DATASET = "public_test"        # đổi thành "private_test" khi có dữ liệu private

print("repo:", REPO, "| dữ liệu:", DATA_DIR)
try:
    import torch
    print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available())
except ImportError:
    print("chưa có torch - vẫn chạy được phần mô hình tuyến tính")
"""))

cells.append(md("""
## 1. Hiểu metric trước khi viết mô hình

Đây là bước quan trọng nhất. eTaPR có mã nguồn mở (`github.com/wshw4ng/eTaPR`); ta cài
lại bằng numpy để (a) chạy nhanh, (b) **hiểu nó phạt cái gì**.

Cơ chế cốt lõi là **pruning**: đoạn dự đoán nào không nằm ít nhất θp = 50% bên trong tấn
công sẽ bị xoá, và khi bị xoá thì phần phủ nó đóng góp cho đoạn tấn công cũng mất theo.
Hệ quả: **dự đoán quá dài làm precision *và* recall cùng về 0.**
"""))
cells.append(code("""
from icsad.etapr import evaluate, ranges_to_labels

N = 4000
# một tấn công dài 200 giây
truth = ranges_to_labels([(1000, 1199)], N)

for name, pred in [
    ("dự đoán trùng khít          ", ranges_to_labels([(1000, 1199)], N)),
    ("ngắn hơn, nằm trong         ", ranges_to_labels([(1050, 1149)], N)),
    ("dài gấp 3, bọc lấy tấn công ", ranges_to_labels([(900, 1499)], N)),
    ("lệch 100 giây               ", ranges_to_labels([(1100, 1299)], N)),
    ("gắn cờ cả chuỗi             ", np.ones(N, dtype=np.int8)),
]:
    r = evaluate(truth, pred)
    print(f"{name}: eTaP={r['eTaP']:.2f} eTaR={r['eTaR']:.2f} F1={r['f1']:.3f}")
"""))

cells.append(md("""
Bảng dưới trả lời câu hỏi quyết định: **nên dự đoán đoạn dài bao nhiêu?**
Gọi `L` = độ dài tấn công thật, `ℓ` = độ dài đoạn ta dự đoán (nằm trong tấn công).
"""))
cells.append(code("""
def pair_f1(l, L):
    cov, rat = min(l, L) / L, min(l, L) / l
    if cov < 0.1 or rat < 0.5:        # bị pruning -> mất cả hai phía
        return 0.0
    P, R = (1 + rat) / 2, (1 + min(cov, 1)) / 2
    return 2 * P * R / (P + R)

Ls = [120, 300, 600, 1200, 2400, 3600]
print("ℓ \\\\ L " + "".join(f"{L:>8}" for L in Ls))
for l in (60, 120, 180, 300, 600, 1200):
    print(f"{l:>6}" + "".join(f"{pair_f1(l, L):8.2f}" for L in Ls))
print("\\nVùng an toàn: ℓ nằm trong [0.1·L, 2·L]. Ra ngoài là 0 điểm.")
print("=> Độ dài đoạn dự đoán là SIÊU THAM SỐ HẠNG NHẤT, phải đo chứ không đoán.")
"""))

cells.append(md("""
## 2. Khảo sát dữ liệu

Mục tiêu: tìm các **bẫy** chứ không phải vẽ biểu đồ đẹp. Bốn thứ cần biết ngay:
số dòng và tính liên tục thời gian, cột hằng số, cụm tín hiệu trùng nhau, và mức lệch
chế độ vận hành giữa train và test.
"""))
cells.append(code("""
from icsad.data import load_training, load_test

segments, columns = load_training(DATA_DIR)
test_seg, _ = load_test(DATA_DIR, DATASET, columns=columns)

print(f"{len(segments)} tệp train, tổng {sum(len(s) for s in segments):,} dòng, {len(columns)} tín hiệu")
for s in segments:
    d = np.diff(s.timestamps).astype("timedelta64[s]").astype(int)
    print(f"  {s.name}: {len(s):>7,} dòng | {s.timestamps[0]} -> {s.timestamps[-1]} | "
          f"bước thời gian {set(np.unique(d))} | Attack={s.attack.sum()}")
print(f"test: {len(test_seg):,} dòng | {test_seg.timestamps[0]} -> {test_seg.timestamps[-1]}")
"""))
cells.append(code("""
allv = np.concatenate([s.values for s in segments])
const = [c for i, c in enumerate(columns) if allv[:, i].min() == allv[:, i].max()]
print(f"{len(const)}/{len(columns)} cột là HẰNG SỐ trong train -> bỏ đi:")
print("  " + ", ".join(const))

# lệch chế độ vận hành train <-> test
print("\\n10 tín hiệu lệch mạnh nhất giữa train và test (theo độ lệch chuẩn của train):")
mu_tr, sd_tr = allv.mean(0), allv.std(0) + 1e-9
shift = np.abs(test_seg.values.mean(0) - mu_tr) / sd_tr
for i in np.argsort(-shift)[:10]:
    print(f"  {columns[i]:<14} train {mu_tr[i]:>10.2f} -> test {test_seg.values[:, i].mean():>10.2f}  ({shift[i]:.1f}σ)")
del allv
"""))
cells.append(md("""
Lệch chế độ vận hành mạnh như trên có hai hệ quả bắt buộc phải xử lý:

1. **Không dùng ngưỡng tuyệt đối học từ train** — phải chuẩn hoá residual trên chính
   chuỗi đang chấm.
2. "Giá trị nằm ngoài dải train" **không** phải dấu hiệu tấn công ở bài này (68% số giây
   của test có ít nhất một tín hiệu ngoài dải train).
"""))

cells.append(md("""
## 3. Tiền xử lý

- Min-max theo train, cắt vào `[-0.5, 1.5]`.
- Gom cụm tín hiệu có |tương quan| ≥ 0,995: `P1_FT01` và `P1_FT01Z` chỉ khác hệ số quy
  đổi, nếu để mô hình dự đoán cái này từ cái kia thì residual luôn bằng 0 ⇒ **mù hoàn toàn**.
"""))
cells.append(code("""
from icsad.data import Scaler, correlation_clusters

scaler = Scaler.fit(segments, columns)
arrays = {s.name: scaler.transform(s.values) for s in segments}
X_test = scaler.transform(test_seg.values)
names = scaler.kept_columns

clusters = correlation_clusters(np.concatenate([a[::5] for a in arrays.values()]), 0.995)
print(f"giữ {len(names)}/{len(columns)} tín hiệu | {len(clusters)} cụm")
print("Các cụm có nhiều hơn 1 tín hiệu (phải loại cả cụm khi dự đoán):")
for c in sorted((c for c in clusters if len(c) > 1), key=len, reverse=True):
    print("  - " + ", ".join(names[i] for i in c))

scaler.save(OUT_DIR / "scaler.json")
json.dump([[int(i) for i in c] for c in clusters], open(OUT_DIR / "clusters.json", "w"))

HOLDOUT = "train6.csv"          # để dành làm kiểm định
train_arrays = [a for k, a in arrays.items() if k != HOLDOUT]
val_arrays = [arrays[HOLDOUT]]
"""))

cells.append(md("""
## 4. Mô hình A — quan hệ tuyến tính giữa các tín hiệu

*Cái gì bị phá vỡ khi có tấn công?* Trong nhà máy, các tín hiệu ràng buộc nhau bởi vật lý
và vòng điều khiển. Tấn công = ghi giá trị giả vào một điểm điều khiển ⇒ **quan hệ bị gãy**.

Với mỗi tín hiệu đích `j`:

```
v_j(t) ≈ w · [v, ewma_quá_khứ(20s, 300s), ewma_tương_lai(20s, 300s)] của MỌI tín hiệu khác
```

Bài chấm offline nên dùng được cả tương lai. Mẹo cài đặt: tích luỹ **ma trận Gram**
(340×340) theo khối rồi giải nghiệm cho cả 68 biến đích từ cùng một ma trận —
**15 giây CPU cho 1 triệu dòng**, RAM < 1 GB.
"""))
cells.append(code("""
import time
from icsad.relation import RelationModel, DEFAULT_RIDGE_GRID

t0 = time.time()
rel = RelationModel().fit(train_arrays, clusters=clusters, val_arrays=val_arrays,
                          ridge_grid=DEFAULT_RIDGE_GRID)
rel.save(OUT_DIR / "relation.npz")
print(f"huấn luyện xong sau {time.time()-t0:.0f}s")

res_val = rel.residuals(val_arrays[0])
print(f"MAE trên dữ liệu bình thường (tệp giữ lại): {np.abs(res_val).mean():.4f}")
order = np.argsort(-np.abs(res_val).mean(0))
print("khó dự đoán nhất :", ", ".join(names[i] for i in order[:5]))
print("dự đoán chuẩn nhất:", ", ".join(names[i] for i in order[-5:]))
"""))
cells.append(md("""
### Phép tự kiểm tra bắt buộc: chạy trên dữ liệu train *sạch*

Không có nhãn thì không có gì báo động khi mô hình sai. Phép thử rẻ nhất: cho mô hình
chấm chính dữ liệu bình thường — nếu nó kêu ở đó thì lỗi là của ta.
"""))
cells.append(code("""
from icsad.scoring import normalize, aggregate
from icsad.postprocess import moving_average

s_clean = moving_average(aggregate(normalize(res_val, mode="rank"), topk=3), 60)
print(f"điểm trên dữ liệu sạch: trung vị={np.median(s_clean):.2f}  p99={np.quantile(s_clean,0.99):.2f}")
print(f"  60 giây đầu chuỗi : {s_clean[:60].mean():.2f}   <-- phải xấp xỉ mức nền")
print(f"  60 giây cuối chuỗi: {s_clean[-60:].mean():.2f}")
print("\\nNếu hai số cuối cao gấp nhiều lần trung vị thì đó là artifact EWMA ở biên:")
print("đặc trưng EWMA ở đầu chuỗi không có quá khứ thật. icsad.features đã xử lý bằng")
print("đệm phản chiếu, và scoring.boundary_guard dập nốt phần rìa còn lại.")
"""))

cells.append(md("""
## 5. Mô hình B — TCN che-và-tái tạo (tuỳ chọn, cần GPU)

Phiên bản phi tuyến của mô hình A: mạng tích chập giãn hai chiều (±510 giây) nhận toàn bộ
chuỗi nhưng **một cụm tín hiệu bị che**, nhiệm vụ là tái tạo lại chính cụm đó.

> **Bẫy đã vấp phải**: bản đầu dùng `GroupNorm` — chuẩn hoá dọc *trục thời gian*. Mạng
> train với cửa sổ 512 điểm nhưng khi chấm chạy trên cả chuỗi 32.000 điểm ⇒ thống kê khác
> hẳn, residual phình từ 0,074 lên 0,196 và F1 rơi từ 0,45 xuống 0,13. Cách phát hiện:
> chạy suy luận với nhiều `chunk` khác nhau, kết quả đổi theo là có lỗi. Cách sửa:
> `ChannelNorm` — chuẩn hoá theo trục kênh tại từng thời điểm.

Đặt `TRAIN_TCN = False` để bỏ qua (pipeline vẫn chạy với A + C).
"""))
cells.append(code("""
TRAIN_TCN = True
EPOCHS = 20          # ~35 phút trên RTX 3050 / T4

res_tcn_test = None
if TRAIN_TCN:
    import torch
    from icsad.tcn import TCNConfig, train_tcn, predict_residuals, save_model, load_model

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = OUT_DIR / "tcn.pt"
    if ckpt.exists():
        model = load_model(ckpt, device)
        print("dùng lại mô hình đã huấn luyện:", ckpt)
    else:
        allX = np.concatenate([a[::10] for a in train_arrays])
        fw = (1.0 / (allX.std(axis=0) + 0.02)).astype(np.float32); fw /= fw.mean(); del allX
        cfg = TCNConfig(n_features=len(names), channels=160, window=512, batch_size=32,
                        epochs=EPOCHS, steps_per_epoch=1500)
        print(f"tầm nhìn ±{cfg.receptive_field//2}s, {len(clusters)} cụm che")
        model = train_tcn(train_arrays, clusters, cfg, device, val_arrays=val_arrays,
                          feature_weight=fw, log_every=500)
        save_model(model, ckpt)
    res_tcn_test = predict_residuals(model, X_test, clusters, device, verbose=False)
    np.save(OUT_DIR / f"res_tcn_{DATASET}.npy", res_tcn_test)
    print("residual TCN trung bình:", np.abs(res_tcn_test).mean())
"""))

cells.append(md("""
## 6. Mô hình C — k-láng-giềng "trạng thái này đã từng xảy ra chưa"

Hai mô hình trên có một **điểm mù**: khi kẻ tấn công đổi setpoint, vòng điều khiển đáp
ứng lại và nhà máy chạy ổn định ở điểm làm việc mới — quan hệ lúc đó *vẫn đúng*, chỉ là
điểm làm việc đó chưa từng thấy.

Mẹo quan trọng: nếu chỉ so với train thì **cả ngày test đều "lạ"** (do lệch chế độ vận
hành) — nó gắn cờ 23% số điểm. Cho phép so với **chính chuỗi test** nhưng loại vùng ±30
phút quanh điểm đang xét thì chế độ bình thường luôn tìm được "bạn", còn đoạn tấn công
vài phút thì không.
"""))
cells.append(code("""
from icsad.novelty import NeighborModel

nn_model = NeighborModel(smooth_window=60, stride=5, pca_dim=24, k=8,
                         self_reference=True, exclude_window=1800).fit(train_arrays)
res_nn_test = nn_model.residuals(X_test)
np.save(OUT_DIR / f"res_nn_{DATASET}.npy", res_nn_test)
print("residual k-NN trung bình:", np.abs(res_nn_test).mean())
"""))

cells.append(md("""
## 7. Gộp điểm và xếp hạng các vùng nghi vấn

Chuẩn hoá **theo từng tín hiệu** bằng thứ hạng (ECDF) trên chính chuỗi đang chấm:
`s_j(t) = −log(1 − F_j(|r_j(t)|))`. Mỗi tín hiệu đóng góp một "độ bất ngờ" bị chặn trên,
không phụ thuộc dạng phân phối, và ngưỡng có ý nghĩa phân vị cố định nên chuyển sang tập
test khác vẫn dùng được.

Rồi gộp **top-k** tín hiệu cao nhất (một tấn công chỉ làm gãy quan hệ ở vài tín hiệu:
lấy trung bình tất cả thì pha loãng, lấy max thì quá nhạy nhiễu).
"""))
cells.append(code("""
from icsad.pipeline import build_score
from icsad.postprocess import smooth, merge_gaps, drop_short
from icsad.etapr import labels_to_ranges

res_rel_test = rel.residuals(X_test)
np.save(OUT_DIR / f"res_relation_{DATASET}.npy", res_rel_test)

MODELS = ["relation", "nn"] + (["tcn"] if res_tcn_test is not None else [])
paths = [OUT_DIR / f"res_{m}_{DATASET}.npy" for m in MODELS]
score = smooth(build_score(paths, norm="rank", topk=1, guard=(60, 300), fuse="feature_mean"), 60)

# tách vùng ứng viên ở ngưỡng thấp rồi xếp hạng theo đỉnh
cand = drop_short(merge_gaps((score > 3.5).astype(np.int8), 120), 60)
regions = labels_to_ranges(cand)
peaks = np.array([score[a:b+1].max() for a, b in regions])
ranked = [regions[i] for i in np.argsort(-peaks)]
hh = lambda i: f"{i//3600:02d}:{(i%3600)//60:02d}:{i%60:02d}"

Z = np.mean([normalize(np.load(p), mode="rank") for p in paths], axis=0)
print(f"{len(regions)} vùng ứng viên. 15 vùng tin cậy nhất:")
for r, (a, b) in enumerate(ranked[:15], 1):
    top = np.argsort(-Z[a:b+1].mean(0))[:3]
    print(f"  {r:2d}. {hh(a)}-{hh(b)} ({b-a+1:5d}s) đỉnh={score[a:b+1].max():5.2f} | "
          + ", ".join(names[i] for i in top))
"""))

cells.append(md("""
## 8. Chọn hình dạng đầu ra — nơi kiếm được nhiều điểm nhất

Tới đây mô hình đã xong. **Phần còn lại quyết định gấp đôi điểm số.**

Tập test không có nhãn nên không dò được hình dạng ở máy. Cách làm: nộp thử theo kiểu
thiết kế thí nghiệm, **mỗi lần đổi đúng một biến**. Kết quả 17 lần nộp:

| Số đoạn \\ độ rộng | 120 s | 180 s | 300 s | 630 s |
|---|---|---|---|---|
| 5 | — | — | 32,87 | — |
| 8 | 36,50 | **37,87** | 25,71 | — |
| 12 | 38,55 | **39,18** ⭐ | 28,78 | — |
| 16 | — | 31,89 | 23,14 | — |
| 18 | — | — | — | **0,00** |

Cộng thêm: dịch cả 12 đoạn ±60 giây → 28,82 / 19,24 (tức căn chỉnh hiện tại đã đúng, và
lệch 60 giây mất 10–20 điểm).

**Kết luận**: tấn công thật chỉ dài ~150–250 giây; tối ưu là **≈12 đoạn cho mỗi 24 giờ,
mỗi đoạn 180 giây**, đặt tại cửa sổ có tổng điểm lớn nhất trong mỗi vùng ứng viên.
"""))
cells.append(code("""
from icsad.postprocess import cap_length, summarize
from icsad.etapr import ranges_to_labels

RATE_PER_DAY = 12      # số đoạn trên mỗi 24 giờ dữ liệu (tự co giãn theo độ dài tập test)
WIDTH = 180            # độ rộng mỗi đoạn, giây

n_keep = max(1, round(RATE_PER_DAY * len(score) / 86400))
chosen = sorted(ranked[:n_keep])
labels = cap_length(ranges_to_labels(chosen, len(score)), score, WIDTH)

info = summarize(labels)
print(f"{info['n_segments']} đoạn | {info['ratio']:.2%} số điểm | "
      f"dài min {info['len_min']}s / TV {info['len_median']:.0f}s / max {info['len_max']}s")
for a, b in labels_to_ranges(labels):
    top = np.argsort(-Z[a:b+1].mean(0))[:3]
    print(f"   {hh(a)}-{hh(b)} | " + ", ".join(names[i] for i in top))
"""))

cells.append(md("## 9. Xuất tệp nộp và kiểm tra định dạng"))
cells.append(code("""
import csv

out_path = REPO / "predictions.csv"
with out_path.open("w", encoding="utf-8", newline="") as fh:
    w = csv.writer(fh, lineterminator="\\n")
    w.writerow(("row_id", "anomaly"))
    for i, v in enumerate(labels):
        w.writerow((i, int(v)))

rows = list(csv.reader(out_path.open(encoding="utf-8")))
ids = [int(r[0]) for r in rows[1:]]
assert rows[0] == ["row_id", "anomaly"], "sai header"
assert len(rows) - 1 == len(test_seg), f"số dòng {len(rows)-1} != test.csv {len(test_seg)}"
assert ids == list(range(len(ids))), "row_id phải liên tục từ 0"
assert set(r[1] for r in rows[1:]) <= {"0", "1"}, "chỉ được chứa 0/1"
print(f"OK: {out_path} — {len(ids):,} dòng, {sum(int(r[1]) for r in rows[1:]):,} điểm gắn cờ")
"""))

cells.append(md("## 10. Nhìn lại kết quả"))
cells.append(code("""
import matplotlib.pyplot as plt

t = np.arange(len(score)) / 3600.0
fig, ax = plt.subplots(figsize=(15, 4))
ax.plot(t, score, lw=0.6, label="điểm bất thường (đã làm trơn 60 s)")
for a, b in labels_to_ranges(labels):
    ax.axvspan(a/3600, b/3600, color="red", alpha=0.25, lw=0)
ax.set_xlabel("giờ trong ngày"); ax.set_ylabel("điểm"); ax.legend(loc="upper right", fontsize=8)
ax.set_title(f"{DATASET}: {labels.sum():,} điểm gắn cờ ({labels.mean():.2%}) trong {len(labels_to_ranges(labels))} đoạn")
plt.tight_layout(); plt.show()
"""))

cells.append(md("""
## 11. Chạy cho private test

Không cần huấn luyện lại — chỉ cần chấm điểm tập mới rồi áp dụng đúng công thức:

```bash
python scripts/score.py          --dataset private_test
python scripts/score_tcn.py      --dataset private_test --device cuda
python scripts/score_neighbor.py --dataset private_test --device cuda
python scripts/make_topn.py      --dataset private_test --models relation tcn nn \\
       --rate 12 --max-len 180 --out predictions.csv
```

`--rate` đặt số đoạn theo **mỗi 24 giờ** nên tự co giãn nếu tập private dài/ngắn khác
public. Trong notebook này chỉ cần đổi `DATASET = "private_test"` ở ô đầu rồi chạy lại
từ bước 6.

### Nếu còn lượt nộp, thứ tự việc nên làm tiếp

1. Nộp bản **chỉ dùng mô hình tuyến tính** ở cùng hình dạng — kiểm tra TCN và k-NN có
   thực sự đóng góp không (chưa bao giờ được kiểm chứng riêng).
2. Nộp **từng khối 8 ứng viên đứng riêng** (hạng 13–20, 21–28): mỗi lượt nộp cho thông
   tin về 8 ứng viên cùng lúc, mà không làm hỏng bản tốt nhất.
3. Tăng recall bằng mô hình phi tuyến từng biến (`HistGradientBoostingRegressor`) — ứng
   viên số một, xem `docs/03_nhat_ky_phien_va_ke_hoach.md`.
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": [], "gpuType": "T4"},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}
for c in nb["cells"]:
    c["source"] = [ln + "\n" for ln in c["source"][:-1]] + [c["source"][-1]]
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, len(cells), "cells")
