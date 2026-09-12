"""Hậu xử lý: chuỗi điểm bất thường -> nhãn 0/1 theo từng giây.

Đây là phần ảnh hưởng rất mạnh tới eTaPR:
  * đoạn dự đoán phải nằm >= 50% trong tấn công (theta_p) -> không được kéo dài
    lê thê ra ngoài;
  * chỉ cần phủ >= 10% đoạn tấn công là được tính "phát hiện" (theta_r), phủ
    nhiều hơn thì thêm điểm portion;
  * mỗi đoạn rác đều bị tính là báo động giả (có trọng số sqrt độ dài).

Quy trình: làm trơn -> ngưỡng trễ (hysteresis) -> ghép khe hở nhỏ -> bỏ đoạn
quá ngắn.
"""
from __future__ import annotations

import numpy as np

from .etapr import labels_to_ranges, ranges_to_labels


# --------------------------------------------------------------------------- #
# làm trơn
# --------------------------------------------------------------------------- #
def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Trung bình trượt có tâm, xử lý biên bằng số điểm thực tế."""
    if window <= 1:
        return x.astype(np.float32, copy=True)
    n = len(x)
    half = window // 2
    csum = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (window - half), 0, n)
    return ((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)).astype(np.float32)


def moving_max(x: np.ndarray, window: int) -> np.ndarray:
    """Cực đại trượt có tâm (dùng để "kéo dài" một đỉnh nhọn)."""
    if window <= 1:
        return x.astype(np.float32, copy=True)
    from scipy.ndimage import maximum_filter1d

    return maximum_filter1d(x.astype(np.float32), size=window, mode="nearest")


def moving_median(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x.astype(np.float32, copy=True)
    from scipy.ndimage import median_filter

    return median_filter(x.astype(np.float32), size=window, mode="nearest")


def smooth(x: np.ndarray, window: int, kind: str = "mean") -> np.ndarray:
    if kind == "mean":
        return moving_average(x, window)
    if kind == "median":
        return moving_median(x, window)
    if kind == "max":
        return moving_max(x, window)
    raise ValueError(f"kind không hợp lệ: {kind}")


# --------------------------------------------------------------------------- #
# ngưỡng
# --------------------------------------------------------------------------- #
def hysteresis(score: np.ndarray, th_hi: float, th_lo: float) -> np.ndarray:
    """Bắt đầu đoạn khi score > th_hi, kéo dài chừng nào score > th_lo."""
    if th_lo > th_hi:
        th_lo = th_hi
    above_lo = score > th_lo
    if not above_lo.any():
        return np.zeros(len(score), dtype=np.int8)
    labels = np.zeros(len(score), dtype=np.int8)
    for start, end in labels_to_ranges(above_lo.astype(np.int8)):
        if score[start : end + 1].max() > th_hi:
            labels[start : end + 1] = 1
    return labels


def merge_gaps(labels: np.ndarray, max_gap: int) -> np.ndarray:
    """Nối hai đoạn cách nhau <= max_gap điểm."""
    if max_gap <= 0:
        return labels
    ranges = labels_to_ranges(labels)
    if not ranges:
        return labels
    merged = [list(ranges[0])]
    for s, e in ranges[1:]:
        if s - merged[-1][1] - 1 <= max_gap:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return ranges_to_labels([(s, e) for s, e in merged], len(labels))


def drop_short(labels: np.ndarray, min_len: int) -> np.ndarray:
    """Bỏ các đoạn ngắn hơn min_len (giảm báo động giả rời rạc)."""
    if min_len <= 1:
        return labels
    ranges = [(s, e) for s, e in labels_to_ranges(labels) if e - s + 1 >= min_len]
    return ranges_to_labels(ranges, len(labels))


def cap_length(labels: np.ndarray, score: np.ndarray, max_len: int) -> np.ndarray:
    """Cắt bớt đoạn quá dài, giữ lại cửa sổ max_len có tổng điểm cao nhất."""
    if max_len <= 0:
        return labels
    out = np.zeros_like(labels)
    for s, e in labels_to_ranges(labels):
        if e - s + 1 <= max_len:
            out[s : e + 1] = 1
            continue
        seg = score[s : e + 1].astype(np.float64)
        csum = np.concatenate(([0.0], np.cumsum(seg)))
        win = csum[max_len:] - csum[:-max_len]
        best = int(np.argmax(win))
        out[s + best : s + best + max_len] = 1
    return out


# --------------------------------------------------------------------------- #
# gói gọn
# --------------------------------------------------------------------------- #
def score_to_labels(
    score: np.ndarray,
    th_hi: float,
    th_lo: float | None = None,
    smooth_window: int = 1,
    smooth_kind: str = "mean",
    min_len: int = 1,
    max_gap: int = 0,
    max_len: int = 0,
) -> np.ndarray:
    """Toàn bộ chuỗi hậu xử lý, trả về nhãn int8 (n,)."""
    s = smooth(score, smooth_window, smooth_kind)
    labels = hysteresis(s, th_hi, th_hi if th_lo is None else th_lo)
    labels = merge_gaps(labels, max_gap)
    labels = drop_short(labels, min_len)
    if max_len:
        labels = cap_length(labels, s, max_len)
    return labels


def threshold_from_quantile(score: np.ndarray, q: float) -> float:
    """Ngưỡng tuyệt đối ứng với phân vị q (q tính theo tỉ lệ điểm bị gắn cờ)."""
    return float(np.quantile(score, 1.0 - q))


def summarize(labels: np.ndarray) -> dict:
    ranges = labels_to_ranges(labels)
    lens = np.array([e - s + 1 for s, e in ranges]) if ranges else np.array([0])
    return {
        "n_segments": len(ranges),
        "flagged": int(labels.sum()),
        "ratio": float(labels.mean()),
        "len_min": int(lens.min()),
        "len_median": float(np.median(lens)),
        "len_max": int(lens.max()),
    }
