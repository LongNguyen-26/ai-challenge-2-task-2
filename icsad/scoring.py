"""Biến residual của mô hình thành điểm bất thường theo từng thời điểm.

Hai bước:
  1. Chuẩn hoá residual theo từng tín hiệu (robust z-score). Việc này rất quan
     trọng vì mỗi tín hiệu có mức nhiễu khác nhau, và vì tập test có thể lệch
     chế độ vận hành so với train (bias không đổi sẽ bị trừ đi hết).
  2. Gộp các tín hiệu lại thành một chuỗi điểm duy nhất (trung bình top-k).
"""
from __future__ import annotations

import numpy as np


def robust_stats(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trả về (median, MAD*1.4826) theo từng cột."""
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
    """Residual (n, F) -> z-score không âm (n, F).

    Args:
        residuals      : residual có dấu của mô hình.
        ref_center/scale: thống kê lấy từ dữ liệu bình thường (tuỳ chọn).
        use_target_stats: dùng chính chuỗi đang chấm để ước lượng median/MAD.
            Đây là cách bền vững nhất với dịch chuyển chế độ vận hành: tấn công
            chỉ chiếm vài % số điểm nên median/MAD gần như không bị ảnh hưởng.
        floor          : sàn của độ lệch (đơn vị min-max) để tín hiệu gần như
            xác định hoàn toàn không sinh z-score khổng lồ từ nhiễu số học.
        clip_max       : cắt trần z-score (tránh một tín hiệu áp đảo tất cả).
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


def aggregate(z: np.ndarray, topk: int = 3, how: str = "topk") -> np.ndarray:
    """Gộp z-score của các tín hiệu thành một chuỗi điểm (n,)."""
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


def combine_scores(scores: list[np.ndarray], weights: list[float] | None = None, how: str = "mean") -> np.ndarray:
    """Ghép điểm của nhiều mô hình (mỗi chuỗi nên đã ở thang z tương đương)."""
    arr = np.stack(scores)
    if how == "max":
        return arr.max(axis=0)
    w = np.asarray(weights if weights is not None else [1.0] * len(scores), dtype=np.float64)
    w = w / w.sum()
    return (arr * w[:, None]).sum(axis=0).astype(np.float32)

def rank_scores(residuals: np.ndarray, clip_max: float | None = None) -> np.ndarray:
    """Chuan hoa theo thu hang (ECDF) tren chinh chuoi dang cham.

    s_j(t) = -log(1 - F_j(|r_j(t)|)) voi F_j la ham phan phoi thuc nghiem cua
    |residual| tren toan chuoi. Moi tin hieu dong gop mot "do bat ngo" bi chan
    tren boi log(n), khong phu thuoc dang phan phoi duoi -> ben vung voi duoi
    nang va voi viec doi che do van hanh.
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
    """Dispatcher: mode = 'mad' | 'rank'."""
    if mode == "mad":
        return residual_scores(residuals, **kwargs)
    if mode == "rank":
        return rank_scores(residuals, clip_max=kwargs.get("clip_max"))
    raise ValueError(f"mode khong hop le: {mode}")


def boundary_guard(score: np.ndarray, n_zero: int = 60, n_ramp: int = 300) -> np.ndarray:
    """Dap dan diem o hai dau chuoi.

    O dau va cuoi chuoi, cac dac trung EWMA thieu boi canh that (chi co du lieu
    phan chieu) nen residual bi thoi phong mot cach he thong - do tren du lieu
    train sach, 60 giay dau co diem trung binh cao gap 3-8 lan muc nen. Ham nay
    nhan diem voi trong so 0 -> 1 tang tuyen tinh tu n_zero den n_ramp.
    """
    if n_ramp <= 0:
        return score
    out = score.astype(np.float32, copy=True)
    n = len(out)
    ramp = np.ones(n, dtype=np.float32)
    t = np.arange(n, dtype=np.float32)
    ramp = np.minimum(ramp, np.clip((t - n_zero) / max(n_ramp - n_zero, 1), 0.0, 1.0))
    rev = np.clip((t[::-1] - n_zero) / max(n_ramp - n_zero, 1), 0.0, 1.0)
    ramp = np.minimum(ramp, rev)
    return out * ramp
