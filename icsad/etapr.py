"""eTaPR - enhanced Time-series aware Precision & Recall (Hwang et al., SAC'22).

Cài đặt lại theo đúng mã nguồn gốc (github.com/wshw4ng/eTaPR, gói `eTaPR_pkg`)
nhưng vector hoá bằng numpy nên nhanh hơn nhiều lần. Judge của cuộc thi dùng
theta_p = 0.50, theta_r = 0.10, delta = 0 (delta = 0 làm vùng "ambiguous" không
đóng góp điểm, xem `_gen_ambiguous`/`_overlap_and_subsequent_score` bản gốc).

Tóm tắt công thức (A = các đoạn tấn công thật, P = các đoạn dự đoán):

    M[a, p] = độ dài giao nhau của đoạn a và đoạn p
    Pruning (lặp tới khi ổn định):
        * hàng a bị xoá nếu 0 < sum_p M[a,p] / |a| < theta_r
        * cột p bị xoá nếu 0 < sum_a M[a,p] / |p| < theta_p
    cov[a]   = sum_p M[a,p] / |a|          (sau pruning)
    rat[p]   = sum_a M[a,p] / |p|          (sau pruning)
    eTaR     = mean_a  ( d_a + d_a * min(cov[a],1) ) / 2 ,  d_a = 1 nếu cov[a] >= theta_r
    eTaP     = sum_p w_p ( d_p + d_p * rat[p] ) / 2 / sum_p w_p ,  w_p = sqrt(|p|),
                                                    d_p = 1 nếu rat[p] >= theta_p
    F1       = 2*eTaP*eTaR / (eTaP + eTaR),   Score = 100 * F1

Hệ quả quan trọng khi tối ưu:
  * mỗi đoạn dự đoán phải nằm >= 50% bên trong tấn công, nếu không nó bị xoá và
    trở thành báo động giả (đoạn dự đoán quá dài rất hại);
  * chỉ cần phủ >= 10% một đoạn tấn công là coi như phát hiện được, nhưng phủ
    càng nhiều thì recall càng cao (điểm portion);
  * precision có trọng số sqrt(độ dài) nên nhiều đoạn ngắn rác vẫn bị phạt,
    song một đoạn đúng và dài sẽ "gánh" được kha khá.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

Range = tuple[int, int]  # (start, end) - bao gồm cả hai đầu


# --------------------------------------------------------------------------- #
# chuyển đổi nhãn <-> đoạn
# --------------------------------------------------------------------------- #
def labels_to_ranges(labels: Iterable[int]) -> list[Range]:
    """Chuỗi nhãn 0/1 -> danh sách đoạn [start, end] (inclusive)."""
    y = np.asarray(list(labels) if not isinstance(labels, np.ndarray) else labels)
    y = (y != 0).astype(np.int8)
    if y.size == 0:
        return []
    pad = np.concatenate(([0], y, [0]))
    diff = np.diff(pad)
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0] - 1
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def ranges_to_labels(ranges: Sequence[Range], length: int) -> np.ndarray:
    """Danh sách đoạn -> chuỗi nhãn 0/1 độ dài `length`."""
    y = np.zeros(length, dtype=np.int8)
    for s, e in ranges:
        y[max(0, s) : min(length, e + 1)] = 1
    return y


# --------------------------------------------------------------------------- #
# metric
# --------------------------------------------------------------------------- #
def _overlap_matrix(anomalies: Sequence[Range], predictions: Sequence[Range]) -> np.ndarray:
    if not anomalies or not predictions:
        return np.zeros((len(anomalies), len(predictions)), dtype=np.float64)
    a = np.asarray(anomalies, dtype=np.int64)
    p = np.asarray(predictions, dtype=np.int64)
    start = np.maximum(a[:, 0][:, None], p[None, :, 0])
    end = np.minimum(a[:, 1][:, None], p[None, :, 1])
    return np.maximum(end - start + 1, 0).astype(np.float64)


def _prune(mat: np.ndarray, len_a: np.ndarray, len_p: np.ndarray, theta_p: float, theta_r: float) -> np.ndarray:
    """Lặp xoá hàng/cột theo đúng thủ tục `_pruning` của bản gốc."""
    m = mat.copy()
    while True:
        cov = m.sum(axis=1) / len_a
        rows = np.where((cov < theta_r) & (cov > 0.0))[0]
        m[rows] = 0.0
        rat = m.sum(axis=0) / len_p
        cols = np.where((rat < theta_p) & (rat > 0.0))[0]
        m[:, cols] = 0.0
        if rows.size == 0 and cols.size == 0:
            return m


def evaluate_ranges(
    anomalies: Sequence[Range],
    predictions: Sequence[Range],
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict:
    """Tính eTaP / eTaR / F1 từ danh sách đoạn."""
    n_a, n_p = len(anomalies), len(predictions)
    out = {
        "eTaP": 0.0, "eTaPd": 0.0, "eTaPp": 0.0,
        "eTaR": 0.0, "eTaRd": 0.0, "eTaRp": 0.0,
        "f1": 0.0, "score": 0.0,
        "n_anomalies": n_a, "n_predictions": n_p,
        "n_detected": 0, "n_correct": 0,
    }
    if n_a == 0 or n_p == 0:
        return out

    len_a = np.array([e - s + 1 for s, e in anomalies], dtype=np.float64)
    len_p = np.array([e - s + 1 for s, e in predictions], dtype=np.float64)
    weights = np.sqrt(len_p)

    mat = _prune(_overlap_matrix(anomalies, predictions), len_a, len_p, theta_p, theta_r)

    cov = mat.sum(axis=1) / len_a                 # tỉ lệ tấn công được phủ
    det = (cov >= theta_r).astype(np.float64)     # đoạn tấn công có bị phát hiện?
    cov_c = np.minimum(cov, 1.0)
    etar = float(((det + det * cov_c) / 2).mean())

    rat = mat.sum(axis=0) / len_p                 # tỉ lệ dự đoán nằm trong tấn công
    cor = (rat >= theta_p).astype(np.float64)     # dự đoán có "đúng" không?
    etap = float((weights * ((cor + cor * rat) / 2)).sum() / weights.sum())

    f1 = 0.0 if etap + etar == 0 else 2 * etap * etar / (etap + etar)
    out.update(
        eTaP=etap,
        eTaPd=float((weights * cor).sum() / weights.sum()),
        eTaPp=float((weights * rat).sum() / weights.sum()),
        eTaR=etar,
        eTaRd=float(det.mean()),
        eTaRp=float(cov_c.mean()),
        f1=f1,
        score=100.0 * f1,
        n_detected=int(det.sum()),
        n_correct=int(cor.sum()),
    )
    return out


def evaluate(
    y_true: Iterable[int],
    y_pred: Iterable[int],
    theta_p: float = 0.5,
    theta_r: float = 0.1,
) -> dict:
    """Tính eTaPR từ hai chuỗi nhãn 0/1 cùng độ dài."""
    return evaluate_ranges(labels_to_ranges(y_true), labels_to_ranges(y_pred), theta_p, theta_r)


def f1_score(y_true: Iterable[int], y_pred: Iterable[int], theta_p: float = 0.5, theta_r: float = 0.1) -> float:
    """Chỉ trả về F1 (tiện cho vòng lặp tinh chỉnh)."""
    return evaluate(y_true, y_pred, theta_p, theta_r)["f1"]
