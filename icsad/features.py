"""Sinh ma trận đặc trưng cho mô hình quan hệ tuyến tính.

Với mỗi tín hiệu f ta tạo T biến thể:
    0. v(t)                      giá trị hiện tại
    1. ewma quá khứ  (hl ngắn)   trung bình mũ của v(..t-1)
    2. ewma tương lai(hl ngắn)   trung bình mũ của v(t+1..)
    3. ewma quá khứ  (hl dài)
    4. ewma tương lai(hl dài)

Dùng cả hai chiều quá khứ/tương lai vì bài toán được chấm offline: khi dự đoán
ta đã có toàn bộ chuỗi test nên không cần ràng buộc nhân quả.

Cột thứ (f * T + k) ứng với tín hiệu f, biến thể k -> muốn loại một tín hiệu ra
khỏi tập biến giải thích chỉ cần bỏ T cột liên tiếp của nó.

Ma trận đặc trưng được sinh theo *khối* (chunk) kèm vùng đệm (pad) nên RAM chỉ
tốn vài chục MB thay vì vài GB. Ở hai đầu chuỗi, vùng đệm được lấy bằng cách
*phản chiếu* (reflect) dữ liệu: nếu khởi tạo EWMA bằng chính giá trị đầu tiên
thì residual ở đầu chuỗi sẽ lớn giả tạo và tạo ra một "đoạn bất thường" ma.
"""
from __future__ import annotations

from typing import Iterator, Sequence

import numpy as np
from scipy.signal import lfilter

DEFAULT_HALFLIVES = (20.0, 300.0)
N_TRANSFORMS = 1 + 2 * len(DEFAULT_HALFLIVES)


def n_transforms(halflives: Sequence[float] = DEFAULT_HALFLIVES) -> int:
    return 1 + 2 * len(halflives)


def _ewma_past(x: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA của *quá khứ nghiêm ngặt*: out[t] = ewma(x[0..t-1])."""
    alpha = float(np.exp(-np.log(2.0) / halflife))
    # y[t] = alpha*y[t-1] + (1-alpha)*x[t], khởi tạo y[-1] = x[0]
    zi = (alpha * x[0].astype(np.float64))[None, :]
    y, _ = lfilter([1.0 - alpha], [1.0, -alpha], x.astype(np.float64), axis=0, zi=zi)
    out = np.empty_like(y)
    out[0] = x[0]
    out[1:] = y[:-1]
    return out.astype(np.float32, copy=False)


def _ewma_future(x: np.ndarray, halflife: float) -> np.ndarray:
    """EWMA của *tương lai nghiêm ngặt*: out[t] = ewma(x[t+1..])."""
    return _ewma_past(x[::-1], halflife)[::-1]


def transform_block(X: np.ndarray, halflives: Sequence[float] = DEFAULT_HALFLIVES) -> np.ndarray:
    """(n, F) -> (n, F * T) float32, bố cục [feature-major]."""
    n, F = X.shape
    T = n_transforms(halflives)
    out = np.empty((n, F * T), dtype=np.float32)
    out[:, 0::T] = X
    for i, hl in enumerate(halflives):
        out[:, 1 + 2 * i :: T] = _ewma_past(X, hl)
        out[:, 2 + 2 * i :: T] = _ewma_future(X, hl)
    return out


def _pad_reflect(X: np.ndarray, left: int, right: int) -> np.ndarray:
    """Đệm phản chiếu, tự thu hẹp nếu chuỗi quá ngắn."""
    if left == 0 and right == 0:
        return X
    n = len(X)
    mode = "reflect" if n > max(left, right) else "edge"
    return np.pad(X, ((left, right), (0, 0)), mode=mode)


def iter_design(
    arrays: Sequence[np.ndarray],
    chunk: int = 20000,
    pad: int = 4096,
    halflives: Sequence[float] = DEFAULT_HALFLIVES,
) -> Iterator[tuple[int, slice, np.ndarray]]:
    """Duyệt từng khối của từng đoạn dữ liệu.

    Yields:
        (chỉ số đoạn, slice trong đoạn đó, ma trận đặc trưng của khối)
    """
    for si, X in enumerate(arrays):
        n = len(X)
        for start in range(0, n, chunk):
            stop = min(start + chunk, n)
            lo, hi = start - pad, stop + pad
            block = X[max(0, lo) : min(n, hi)]
            block = _pad_reflect(block, max(0, -lo), max(0, hi - n))
            Z = transform_block(block, halflives)
            off = start - lo          # vị trí của `start` bên trong khối đã đệm
            yield si, slice(start, stop), Z[off : off + (stop - start)]


def feature_columns(feature_idx: int, T: int) -> list[int]:
    """Các cột trong ma trận đặc trưng sinh ra bởi một tín hiệu."""
    return list(range(feature_idx * T, feature_idx * T + T))
