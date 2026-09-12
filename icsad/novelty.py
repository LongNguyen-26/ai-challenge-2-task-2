"""Mô hình k-láng-giềng gần nhất: "trạng thái này đã từng xảy ra chưa?".

Hai mô hình kia (quan hệ tuyến tính và TCN) đều trả lời câu hỏi *quan hệ giữa
các tín hiệu có còn đúng không*. Chúng rất mạnh ở lúc quá độ, nhưng có một điểm
mù quan trọng: khi kẻ tấn công đổi setpoint, vòng điều khiển đáp ứng lại và nhà
máy chạy ổn định ở điểm làm việc mới - lúc đó *quan hệ vẫn đúng*, chỉ có điều
điểm làm việc đó chưa từng xuất hiện trong dữ liệu bình thường.

Mô hình này trả lời câu hỏi còn lại: lấy trạng thái hiện tại, tìm k trạng thái
giống nhất trong toàn bộ dữ liệu train, rồi lấy trung bình của chúng làm "bản
tái tạo". Residual = phần mà không trạng thái bình thường nào giải thích được,
và nó **kéo dài suốt thời gian tấn công** chứ không chỉ ở hai đầu.

Chi tiết cài đặt:
  * trạng thái = giá trị đã làm trơn (mặc định trung bình trượt 60 giây) nên ta
    so khớp *điểm làm việc* chứ không phải nhiễu tức thời;
  * khoảng cách tính trong không gian PCA (mặc định 24 chiều) cho nhanh;
  * tìm láng giềng bằng torch (chạy GPU nếu có), chia khối để vừa bộ nhớ.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


def _smooth_columns(a: np.ndarray, window: int) -> np.ndarray:
    """Trung bình trượt có tâm theo từng cột."""
    if window <= 1:
        return a.astype(np.float32, copy=False)
    n = len(a)
    half = window // 2
    csum = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0, dtype=np.float64)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (window - half), 0, n)
    return ((csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None]).astype(np.float32)


@dataclass
class NeighborModel:
    """Tái tạo trạng thái bằng k láng giềng gần nhất trong dữ liệu bình thường."""

    smooth_window: int = 60
    stride: int = 5             # lấy mẫu tập tham chiếu (5 giây/điểm)
    pca_dim: int = 24
    k: int = 8
    chunk: int = 512
    self_reference: bool = True     # dùng thêm chính chuỗi đang chấm làm tham chiếu
    exclude_window: int = 1800      # loại tham chiếu cách điểm đang xét < ngần này (giây)

    # tham số học được
    mean: np.ndarray | None = None          # (F,)
    scale: np.ndarray | None = None         # (F,) độ lệch chuẩn, để cân bằng khoảng cách
    components: np.ndarray | None = None    # (F, d)
    ref_proj: np.ndarray | None = None      # (N, d) tập tham chiếu trong không gian PCA
    ref_full: np.ndarray | None = None      # (N, F) giá trị gốc tương ứng

    # ------------------------------------------------------------------ #
    def fit(self, arrays: Sequence[np.ndarray], verbose: bool = True) -> "NeighborModel":
        ref = np.concatenate([_smooth_columns(X, self.smooth_window)[:: self.stride] for X in arrays])
        self.mean = ref.mean(axis=0)
        self.scale = ref.std(axis=0) + 1e-3
        Z = (ref - self.mean) / self.scale
        # PCA qua ma trận hiệp phương sai (F nhỏ nên rẻ)
        cov = (Z.T @ Z) / len(Z)
        vals, vecs = np.linalg.eigh(cov.astype(np.float64))
        order = np.argsort(-vals)
        d = min(self.pca_dim, len(vals))
        self.components = vecs[:, order[:d]].astype(np.float32)
        self.ref_proj = (Z @ self.components).astype(np.float32)
        self.ref_full = ref.astype(np.float32)
        if verbose:
            ev = vals[order[:d]].sum() / vals.sum()
            print(f"  tập tham chiếu {len(ref):,} trạng thái, PCA {d} chiều giữ {ev:.1%} phương sai")
        return self

    # ------------------------------------------------------------------ #
    def residuals(self, X: np.ndarray, device=None, verbose: bool = False) -> np.ndarray:
        """(n, F) -> residual (n, F) so với trung bình k láng giềng gần nhất.

        Nếu `self_reference` bật, *chính chuỗi đang chấm* cũng được đưa vào tập
        tham chiếu nhưng loại đi những điểm cách điểm đang xét dưới
        `exclude_window` giây. Lý do: tập test có thể vận hành ở chế độ hơi khác
        tập train (ví dụ van FCV01 gần như đóng cả ngày trong khi train thì mở),
        nếu chỉ so với train thì cả ngày đều "lạ". Cho phép so với các thời điểm
        khác trong cùng ngày thì chế độ vận hành bình thường luôn tìm được bạn,
        còn một đoạn tấn công 10 phút thì không (vùng quanh nó đã bị loại).
        """
        assert self.ref_proj is not None, "Mô hình chưa được huấn luyện"
        import torch

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Xs = _smooth_columns(X, self.smooth_window)
        Q = ((Xs - self.mean) / self.scale) @ self.components

        ref_p = torch.from_numpy(self.ref_proj).to(device)
        ref_f = torch.from_numpy(self.ref_full).to(device)
        if self.self_reference:
            self_idx = np.arange(0, len(Q), self.stride)
            ref_p = torch.cat([ref_p, torch.from_numpy(Q[self_idx]).to(device)])
            ref_f = torch.cat([ref_f, torch.from_numpy(Xs[self_idx]).to(device)])
            self_time = torch.from_numpy(self_idx.astype(np.int64)).to(device)
            n_train = len(self.ref_proj)

        out = np.empty_like(X)
        with torch.no_grad():
            for start in range(0, len(Q), self.chunk):
                stop = min(start + self.chunk, len(Q))
                q = torch.from_numpy(Q[start:stop]).to(device)
                dist = torch.cdist(q, ref_p)                     # (b, N)
                if self.self_reference:
                    t = torch.arange(start, stop, device=device).unsqueeze(1)
                    near = (self_time.unsqueeze(0) - t).abs() < self.exclude_window
                    dist[:, n_train:] = dist[:, n_train:].masked_fill(near, float("inf"))
                idx = dist.topk(self.k, dim=1, largest=False).indices
                pred = ref_f[idx].mean(dim=1)                    # (b, F)
                out[start:stop] = (torch.from_numpy(Xs[start:stop]).to(device) - pred).cpu().numpy()
                if verbose and (start // self.chunk) % 10 == 0:
                    print(f"    {stop:,}/{len(Q):,}", flush=True)
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path, mean=self.mean, scale=self.scale, components=self.components,
            ref_proj=self.ref_proj, ref_full=self.ref_full,
            params=np.array([self.smooth_window, self.stride, self.pca_dim, self.k,
                             int(self.self_reference), self.exclude_window], dtype=np.int64),
        )

    @classmethod
    def load(cls, path: str | Path) -> "NeighborModel":
        z = np.load(path)
        vals = [int(v) for v in z["params"]]
        sw, st, pd, k = vals[:4]
        sr, ew = (bool(vals[4]), vals[5]) if len(vals) > 5 else (True, 1800)
        m = cls(smooth_window=sw, stride=st, pca_dim=pd, k=k, self_reference=sr, exclude_window=ew)
        m.mean, m.scale, m.components = z["mean"], z["scale"], z["components"]
        m.ref_proj, m.ref_full = z["ref_proj"], z["ref_full"]
        return m
