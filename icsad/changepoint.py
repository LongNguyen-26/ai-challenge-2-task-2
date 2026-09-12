"""Bộ dò điểm-thay-đổi: tín hiệu có nhảy bậc so với chính nó ít phút trước không?

Quan sát từ những đoạn tấn công đã được xác nhận trên bảng xếp hạng public: ở
các thời điểm đó luôn có ít nhất một tín hiệu đổi mức đột ngột so với 15-30 phút
liền trước (ví dụ P2_SCO lệch 3906 lần độ lệch chuẩn nền, P1_B3004 lệch 31 lần).
Đó chính là dấu vết của việc ai đó ghi một giá trị mới vào setpoint/lệnh điều
khiển.

Với mỗi tín hiệu và mỗi thời điểm t:

    score(t) = | trung bình(t .. t+W) - trung bình(t-B .. t) | / (độ lệch nền + sàn)

Khác với mô hình quan hệ (so tín hiệu này với tín hiệu khác), bộ dò này so tín
hiệu với *quá khứ gần của chính nó*, nên bắt được cả những thay đổi mà các tín
hiệu khác "đỡ lời" cho nhau. Đổi lại nó cũng kêu ở các thao tác vận hành bình
thường, nên cần dùng chung với các mô hình kia.
"""
from __future__ import annotations

import numpy as np


def _rolling_mean(a: np.ndarray, window: int, shift: int) -> np.ndarray:
    """Trung bình trượt trên cửa sổ [t+shift, t+shift+window) (đệm bằng biên)."""
    n = len(a)
    csum = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0, dtype=np.float64)])
    lo = np.clip(np.arange(n) + shift, 0, n)
    hi = np.clip(np.arange(n) + shift + window, 0, n)
    return ((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)


def _rolling_std(a: np.ndarray, window: int, shift: int) -> np.ndarray:
    m1 = _rolling_mean(a, window, shift)
    m2 = _rolling_mean(np.square(a, dtype=np.float64).astype(np.float32), window, shift)
    return np.sqrt(np.maximum(m2 - m1 * m1, 0.0))


def change_scores(
    X: np.ndarray,
    window: int = 120,
    baseline: int = 900,
    floor: float = 2e-3,
) -> np.ndarray:
    """(n, F) -> điểm thay đổi (n, F), không âm.

    Args:
        window  : độ dài cửa sổ "sau" (giây).
        baseline: độ dài cửa sổ "trước" dùng làm nền (giây).
        floor   : sàn của độ lệch nền, tránh chia cho 0 ở tín hiệu gần như hằng.
    """
    after = _rolling_mean(X, window, 0)
    before = _rolling_mean(X, baseline, -baseline)
    scale = _rolling_std(X, baseline, -baseline)
    return np.abs(after - before) / np.maximum(scale, floor)


def multi_scale_change_scores(
    X: np.ndarray,
    windows: tuple[int, ...] = (60, 180, 420),
    baseline_ratio: float = 6.0,
    floor: float = 2e-3,
) -> np.ndarray:
    """Lấy cực đại của bộ dò trên nhiều thang thời gian (bắt cả thay đổi nhanh và chậm)."""
    out = None
    for w in windows:
        s = change_scores(X, window=w, baseline=int(w * baseline_ratio), floor=floor)
        out = s if out is None else np.maximum(out, s)
    return out
