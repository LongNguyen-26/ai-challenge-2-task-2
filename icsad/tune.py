"""Tìm siêu tham số hậu xử lý bằng eTaPR trên tập kiểm định có nhãn.

Tập kiểm định ở đây là dữ liệu bình thường đã được tiêm tấn công giả lập
(xem `synth.py`). Có thể truyền nhiều cặp (điểm, nhãn) - ví dụ nhiều seed khác
nhau - để tránh bám vào một lần tiêm may mắn.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Sequence

import numpy as np

from .etapr import evaluate
from .postprocess import score_to_labels, smooth


@dataclass
class PostParams:
    """Bộ tham số biến điểm thành nhãn."""

    smooth_window: int = 60
    smooth_kind: str = "mean"
    th_hi: float = 6.0
    th_lo_ratio: float = 0.6
    min_len: int = 30
    max_gap: int = 60
    max_len: int = 0
    merge_alpha: float = 0.0
    dilate_pad: int = 0

    def apply(self, score: np.ndarray) -> np.ndarray:
        return score_to_labels(
            score,
            th_hi=self.th_hi,
            th_lo=self.th_hi * self.th_lo_ratio,
            smooth_window=self.smooth_window,
            smooth_kind=self.smooth_kind,
            min_len=self.min_len,
            max_gap=self.max_gap,
            max_len=self.max_len,
            merge_alpha=self.merge_alpha,
            dilate_pad=self.dilate_pad,
        )

    def to_dict(self) -> dict:
        return asdict(self)


DEFAULT_GRID = dict(
    smooth_window=(15, 30, 60, 120, 240),
    smooth_kind=("mean",),
    th_hi=(3.0, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0, 22.0),
    th_lo_ratio=(1.0, 0.6),
    min_len=(1, 20, 60, 120),
    max_gap=(0, 60, 180, 420),
    max_len=(0,),
)


def grid_search(
    pairs: Sequence[tuple[np.ndarray, np.ndarray]],
    grid: dict | None = None,
    theta_p: float = 0.5,
    theta_r: float = 0.1,
    top_k: int = 10,
    verbose: bool = True,
    stable_threshold: bool = True,
) -> tuple[PostParams, list[tuple[float, PostParams, dict]]]:
    """Duyệt lưới tham số, trả về (bộ tốt nhất, top_k kết quả).

    Args:
        pairs: danh sách (score, y_true) - mỗi phần tử là một chuỗi kiểm định.
        stable_threshold: xếp hạng theo F1 trung bình của ngưỡng đó và hai ngưỡng
            kề bên thay vì F1 đơn lẻ. Tránh chọn đúng một điểm may mắn nằm ngay
            mép vực - trên tập test thật ngưỡng sẽ lệch đi một chút.
    """
    g = {**DEFAULT_GRID, **(grid or {})}
    results: list[tuple[float, PostParams, dict]] = []

    # làm trơn trước cho từng (window, kind) để không tính lại nhiều lần
    cache: dict[tuple[int, str], list[np.ndarray]] = {}
    for w, kind in product(g["smooth_window"], g["smooth_kind"]):
        cache[(w, kind)] = [smooth(s, w, kind) for s, _ in pairs]

    from .postprocess import drop_short, hysteresis, merge_gaps

    for w, kind in product(g["smooth_window"], g["smooth_kind"]):
        smoothed = cache[(w, kind)]
        for th_hi, ratio in product(g["th_hi"], g["th_lo_ratio"]):
            base = [hysteresis(s, th_hi, th_hi * ratio) for s in smoothed]
            for max_gap in g["max_gap"]:
                merged = [merge_gaps(b, max_gap) for b in base]
                for min_len in g["min_len"]:
                    f1s, details = [], []
                    for lab, (_, y_true) in zip(merged, pairs):
                        y_pred = drop_short(lab, min_len)
                        res = evaluate(y_true, y_pred, theta_p, theta_r)
                        f1s.append(res["f1"])
                        details.append(res)
                    mean_f1 = float(np.mean(f1s))
                    params = PostParams(
                        smooth_window=w, smooth_kind=kind, th_hi=th_hi,
                        th_lo_ratio=ratio, min_len=min_len, max_gap=max_gap,
                    )
                    info = {
                        "f1_each": [round(f, 4) for f in f1s],
                        "flag_ratio": float(np.mean([lab_i.mean() for lab_i in merged])),
                        "eTaP": float(np.mean([d["eTaP"] for d in details])),
                        "eTaR": float(np.mean([d["eTaR"] for d in details])),
                        "n_pred": float(np.mean([d["n_predictions"] for d in details])),
                        "n_det": float(np.mean([d["n_detected"] for d in details])),
                        "n_anom": float(np.mean([d["n_anomalies"] for d in details])),
                    }
                    results.append((mean_f1, params, info))

    results = _prefer_stable_threshold(results, g["th_hi"]) if stable_threshold else results
    results.sort(key=lambda r: -r[0])
    if verbose:
        print(f"  đã thử {len(results)} bộ tham số")
        for f1, p, info in results[:top_k]:
            print(
                f"   F1={f1:.4f} eTaP={info['eTaP']:.3f} eTaR={info['eTaR']:.3f} "
                f"| w={p.smooth_window:<4} th={p.th_hi:<5} lo={p.th_lo_ratio:<4} "
                f"min={p.min_len:<4} gap={p.max_gap:<4} "
                f"| đoạn dự đoán={info['n_pred']:.0f}, phát hiện {info['n_det']:.0f}/{info['n_anom']:.0f}"
            )
    return results[0][1], results[:top_k]


def evaluate_params(params: PostParams, pairs: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict:
    """Chấm một bộ tham số trên các chuỗi kiểm định."""
    out = [evaluate(y, params.apply(s)) for s, y in pairs]
    return {
        "f1": float(np.mean([o["f1"] for o in out])),
        "eTaP": float(np.mean([o["eTaP"] for o in out])),
        "eTaR": float(np.mean([o["eTaR"] for o in out])),
        "detail": out,
    }


def _prefer_stable_threshold(
    results: list[tuple[float, PostParams, dict]],
    th_grid: Sequence[float],
) -> list[tuple[float, PostParams, dict]]:
    """Thay F1 bằng trung bình F1 của ngưỡng đó với hai ngưỡng kề bên.

    Giữ nguyên thông tin gốc trong `info['f1_raw']`.
    """
    th_index = {th: i for i, th in enumerate(th_grid)}
    by_key: dict[tuple, dict[int, float]] = {}
    for f1, p, _ in results:
        key = (p.smooth_window, p.smooth_kind, p.th_lo_ratio, p.min_len, p.max_gap, p.max_len)
        by_key.setdefault(key, {})[th_index[p.th_hi]] = f1

    out = []
    for f1, p, info in results:
        key = (p.smooth_window, p.smooth_kind, p.th_lo_ratio, p.min_len, p.max_gap, p.max_len)
        i = th_index[p.th_hi]
        neigh = [by_key[key].get(j) for j in (i - 1, i, i + 1)]
        vals = [v for v in neigh if v is not None]
        info = {**info, "f1_raw": round(f1, 4), "f1_neighbours": [round(v, 4) for v in vals]}
        out.append((float(np.mean(vals)), p, info))
    return out
