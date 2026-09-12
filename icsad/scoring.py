"""Biến residual của mô hình thành điểm bất thường theo từng thời điểm.

Ba bước:
  1. Chuẩn hoá residual theo *từng tín hiệu*. Bước này rất quan trọng: mỗi tín
     hiệu có mức nhiễu khác nhau, và tập test có thể vận hành ở chế độ hơi khác
     tập train nên residual bị lệch một lượng gần như không đổi - chuẩn hoá theo
     thống kê của chính chuỗi đang chấm sẽ trừ đi phần lệch đó.
  2. Gộp các tín hiệu thành một chuỗi điểm duy nhất (trung bình top-k).
  3. Dập điểm ở hai đầu chuỗi, nơi đặc trưng EWMA không có bối cảnh thật.

Tấn công chỉ chiếm vài phần trăm số điểm nên median/MAD/ECDF tính trên toàn
chuỗi test gần như không bị chúng làm lệch.
"""
from __future__ import annotations

import numpy as np


def robust_stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (median, 1.4826 * MAD) theo từng cột."""
    med = np.median(a, axis=0)
    mad = 1.4826 * np.median(np.abs(a - med), axis=0)
    return med.astype(np.float64), mad.astype(np.float64)


def residual_scores(
    residuals: np.ndarray,
    ref_center: np.ndarray | None = None,
    ref_scale: np.ndarray | None = None,
    use_target_stats: bool = True,
    floor: float = 2e-3,
    clip_max: float | None = 40.0,
) -> np.ndarray:
    """Chuẩn hoá kiểu z-score bền vững: residual (n, F) -> điểm không âm (n, F).

    Args:
        residuals       : residual có dấu của mô hình.
        ref_center/scale: thống kê lấy từ dữ liệu bình thường (tuỳ chọn).
        use_target_stats: ước lượng median/MAD ngay trên chuỗi đang chấm.
        floor           : sàn của độ lệch (đơn vị min-max), tránh việc một tín
            hiệu gần như xác định hoàn toàn sinh ra z-score khổng lồ từ nhiễu số.
        clip_max        : cắt trần, tránh một tín hiệu áp đảo tất cả.
    """
    a = np.abs(residuals).astype(np.float32)
    if use_target_stats:
        center, scale = robust_stats(a)
        if ref_scale is not None:
            scale = np.maximum(scale, ref_scale)
    else:
        if ref_center is None or ref_scale is None:
            raise ValueError("Cần ref_center/ref_scale khi use_target_stats=False")
        center, scale = ref_center, ref_scale
    scale = np.maximum(scale, floor)
    z = (a - center.astype(np.float32)) / scale.astype(np.float32)
    np.maximum(z, 0.0, out=z)
    if clip_max is not None:
        np.minimum(z, clip_max, out=z)
    return z


def rank_scores(residuals: np.ndarray, clip_max: float | None = None) -> np.ndarray:
    """Chuẩn hoá theo thứ hạng (ECDF) trên chính chuỗi đang chấm.

    ``s_j(t) = -log(1 - F_j(|r_j(t)|))`` với ``F_j`` là hàm phân phối thực nghiệm
    của ``|residual|``. Mỗi tín hiệu đóng góp một "độ bất ngờ" chặn trên bởi
    log(n), không phụ thuộc dạng phân phối bên dưới, nên bền vững với đuôi nặng
    và với việc đổi chế độ vận hành. Ngưỡng cũng có ý nghĩa cố định: ngưỡng 5.5
    tương ứng phân vị 1 - e^-5.5 ≈ 99,59% của từng tín hiệu, bất kể chuỗi dài bao nhiêu.
    """
    a = np.abs(residuals)
    n = len(a)
    order = np.argsort(a, axis=0, kind="stable")
    ranks = np.empty_like(order, dtype=np.float32)
    idx = np.arange(1, n + 1, dtype=np.float32)[:, None]
    np.put_along_axis(ranks, order, np.broadcast_to(idx, a.shape), axis=0)
    z = -np.log1p(-ranks / (n + 1.0)).astype(np.float32)
    if clip_max is not None:
        np.minimum(z, clip_max, out=z)
    return z


def normalize(residuals: np.ndarray, mode: str = "mad", **kwargs) -> np.ndarray:
    """Bộ điều phối: mode = 'mad' | 'rank'."""
    if mode == "mad":
        return residual_scores(residuals, **kwargs)
    if mode == "rank":
        return rank_scores(residuals, clip_max=kwargs.get("clip_max"))
    raise ValueError(f"mode không hợp lệ: {mode}")


def aggregate(z: np.ndarray, topk: int = 3, how: str = "topk") -> np.ndarray:
    """Gộp điểm của các tín hiệu thành một chuỗi (n,).

    Trung bình top-k là mặc định: một tấn công thường chỉ làm gãy quan hệ ở vài
    tín hiệu, lấy trung bình tất cả sẽ pha loãng còn lấy max thì quá nhạy nhiễu.
    """
    if how == "max":
        return z.max(axis=1)
    if how == "mean":
        return z.mean(axis=1)
    if how == "topk":
        k = min(topk, z.shape[1])
        if k == 1:
            return z.max(axis=1)
        part = np.partition(z, -k, axis=1)[:, -k:]
        return part.mean(axis=1)
    raise ValueError(f"how không hợp lệ: {how}")


def combine_scores(scores: list[np.ndarray], weights: list[float] | None = None,
                   how: str = "mean") -> np.ndarray:
    """Ghép chuỗi điểm của nhiều mô hình (các chuỗi nên cùng thang đo)."""
    arr = np.stack(scores)
    if how == "max":
        return arr.max(axis=0)
    w = np.asarray(weights if weights is not None else [1.0] * len(scores), dtype=np.float64)
    w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0).astype(np.float32)


def boundary_guard(score: np.ndarray, n_zero: int = 60, n_ramp: int = 300) -> np.ndarray:
    """Dập dần điểm ở hai đầu chuỗi.

    Ở đầu và cuối chuỗi, các đặc trưng EWMA chỉ có dữ liệu phản chiếu chứ không
    có bối cảnh thật nên residual bị thổi phồng một cách hệ thống (đo trên dữ
    liệu train sạch: 60 giây đầu có điểm trung bình cao gấp 3-8 lần mức nền).
    Nếu không dập, tệp test nào cũng sẽ có một "đoạn tấn công ma" ngay ở đầu.

    Trọng số: 0 cho tới `n_zero`, rồi tăng tuyến tính lên 1 tại `n_ramp`.
    """
    if n_ramp <= 0:
        return score
    out = score.astype(np.float32, copy=True)
    n = len(out)
    t = np.arange(n, dtype=np.float32)
    span = max(n_ramp - n_zero, 1)
    ramp = np.minimum(np.clip((t - n_zero) / span, 0.0, 1.0),
                      np.clip((t[::-1] - n_zero) / span, 0.0, 1.0))
    return out * ramp
