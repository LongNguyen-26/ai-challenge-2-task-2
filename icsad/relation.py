"""Mô hình quan hệ tuyến tính giữa các tín hiệu (Ridge, chạy trên CPU).

Ý tưởng: ở trạng thái bình thường, mỗi tín hiệu là một hàm (gần như tuyến tính)
của các tín hiệu còn lại và của bối cảnh thời gian xung quanh. Với mỗi tín hiệu
đích j ta học

    v_j(t)  ~=  w_j . Z(t)          , Z = [v, ewma_past, ewma_future] của MỌI
                                      tín hiệu KHÁC (loại cả cụm tương quan
                                      >= 0.995 với j để tránh "nhìn trộm" bản sao)

Khi bị tấn công, quan hệ này gãy trong suốt thời gian tấn công nên residual
|v_j - v̂_j| lớn liên tục -> rất hợp với eTaPR (cần phủ đoạn, không chỉ chạm).

Cài đặt: tích luỹ ma trận Gram (D x D, D = F * 5 ~ 350) theo khối nên toàn bộ
1 triệu dòng train chỉ tốn vài chục MB RAM và ~30 giây CPU. Sau khi có Gram,
nghiệm cho tất cả tín hiệu đích được lấy ra từ cùng một ma trận -> rất nhanh,
và có thể thử nhiều hệ số ridge gần như miễn phí.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from .features import DEFAULT_HALFLIVES, iter_design, n_transforms

DEFAULT_RIDGE_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


@dataclass
class RelationModel:
    """Bộ hồi quy tuyến tính "mỗi tín hiệu từ các tín hiệu khác"."""

    halflives: tuple[float, ...] = tuple(DEFAULT_HALFLIVES)
    ridge: float = 1e-3
    chunk: int = 20000
    pad: int = 4096

    # tham số học được
    W: np.ndarray | None = None            # (F, D) trọng số
    b: np.ndarray | None = None            # (F,)  hệ số tự do
    clusters: list[list[int]] = field(default_factory=list)
    ridge_per_target: np.ndarray | None = None
    train_residual_scale: np.ndarray | None = None   # (F,) MAD residual trên train

    # ------------------------------------------------------------------ #
    @property
    def n_transforms(self) -> int:
        return n_transforms(self.halflives)

    def _excluded_columns(self, target: int, F: int) -> np.ndarray:
        """Các cột đặc trưng bị cấm dùng khi dự đoán tín hiệu `target`."""
        T = self.n_transforms
        same_cluster = next((c for c in self.clusters if target in c), [target])
        cols = [f * T + k for f in same_cluster for k in range(T)]
        return np.asarray(sorted(set(cols)), dtype=np.int64)

    # ------------------------------------------------------------------ #
    def _accumulate(self, arrays: Sequence[np.ndarray], D: int):
        """Tích luỹ n, tổng và Gram của ma trận đặc trưng."""
        total = 0
        s1 = np.zeros(D, dtype=np.float64)
        gram = np.zeros((D, D), dtype=np.float64)
        for _, _, Z in iter_design(arrays, self.chunk, self.pad, self.halflives):
            Z64 = Z.astype(np.float64)
            total += len(Z64)
            s1 += Z64.sum(axis=0)
            gram += Z64.T @ Z64
        return total, s1, gram

    def fit(
        self,
        arrays: Sequence[np.ndarray],
        clusters: Sequence[Sequence[int]] | None = None,
        val_arrays: Sequence[np.ndarray] | None = None,
        ridge_grid: Sequence[float] | None = None,
        verbose: bool = True,
    ) -> "RelationModel":
        """Huấn luyện trên dữ liệu bình thường đã chuẩn hoá.

        Args:
            arrays     : danh sách mảng (n, F) - mỗi phần tử là một đoạn liên tục.
            clusters   : nhóm tín hiệu gần trùng (xem data.correlation_clusters).
            val_arrays : nếu có, dùng để chọn hệ số ridge riêng cho từng tín hiệu.
            ridge_grid : lưới hệ số ridge cần thử.
        """
        F = arrays[0].shape[1]
        T = self.n_transforms
        D = F * T
        self.clusters = [list(map(int, c)) for c in (clusters or [[i] for i in range(F)])]

        n, s1, gram = self._accumulate(arrays, D)
        mean = s1 / n
        cov = gram / n - np.outer(mean, mean)
        std = np.sqrt(np.maximum(np.diag(cov), 1e-24))
        corr = cov / np.outer(std, std)

        grid = list(ridge_grid) if ridge_grid is not None else [self.ridge]
        W_grid = np.zeros((len(grid), F, D), dtype=np.float32)
        b_grid = np.zeros((len(grid), F), dtype=np.float32)

        for j in range(F):
            y_col = j * T                       # cột "giá trị hiện tại" của tín hiệu j
            excl = self._excluded_columns(j, F)
            keep = np.setdiff1d(np.arange(D), excl, assume_unique=False)
            A = corr[np.ix_(keep, keep)]
            rhs = corr[keep, y_col]
            sy = std[y_col]
            for gi, lam in enumerate(grid):
                beta = np.linalg.solve(A + lam * np.eye(len(keep)), rhs)
                w = beta * sy / std[keep]
                W_grid[gi, j, keep] = w.astype(np.float32)
                b_grid[gi, j] = float(mean[y_col] - w @ mean[keep])

        if len(grid) == 1 or val_arrays is None:
            best = np.zeros(F, dtype=np.int64)
        else:
            mse = np.zeros((len(grid), F), dtype=np.float64)
            count = 0
            for _, sl, Z in iter_design(val_arrays, self.chunk, self.pad, self.halflives):
                y = Z[:, 0::T]
                count += len(Z)
                for gi in range(len(grid)):
                    err = y - (Z @ W_grid[gi].T + b_grid[gi])
                    mse[gi] += np.square(err, dtype=np.float64).sum(axis=0)
            mse /= max(count, 1)
            best = mse.argmin(axis=0)
            if verbose:
                chosen = np.array(grid)[best]
                print(f"  ridge đã chọn: " + ", ".join(f"{v:.0e}x{int((chosen == v).sum())}" for v in sorted(set(chosen))))

        self.W = np.stack([W_grid[best[j], j] for j in range(F)]).astype(np.float32)
        self.b = np.array([b_grid[best[j], j] for j in range(F)], dtype=np.float32)
        self.ridge_per_target = np.array(grid, dtype=np.float64)[best]

        ref = val_arrays if val_arrays is not None else arrays
        step = max(1, sum(len(a) for a in ref) // 300_000)
        res = np.concatenate([self.residuals(X)[::step] for X in ref])
        self.train_residual_scale = _mad(np.abs(res))
        if verbose:
            print(f"  MAE residual trung bình trên dữ liệu bình thường: {np.abs(res).mean():.5f}")
        return self

    # ------------------------------------------------------------------ #
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Dự đoán lại toàn bộ tín hiệu: (n, F) -> (n, F)."""
        assert self.W is not None and self.b is not None, "Mô hình chưa được huấn luyện"
        out = np.empty_like(X)
        for _, sl, Z in iter_design([X], self.chunk, self.pad, self.halflives):
            out[sl] = Z @ self.W.T + self.b
        return out

    def residuals(self, X: np.ndarray) -> np.ndarray:
        """Residual có dấu: (n, F) -> (n, F)."""
        assert self.W is not None and self.b is not None, "Mô hình chưa được huấn luyện"
        T = self.n_transforms
        out = np.empty_like(X)
        for _, sl, Z in iter_design([X], self.chunk, self.pad, self.halflives):
            out[sl] = Z[:, 0::T] - (Z @ self.W.T + self.b)
        return out

    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            W=self.W,
            b=self.b,
            halflives=np.asarray(self.halflives, dtype=np.float64),
            ridge_per_target=self.ridge_per_target,
            train_residual_scale=self.train_residual_scale,
            clusters=np.array([",".join(map(str, c)) for c in self.clusters], dtype=object),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RelationModel":
        z = np.load(path, allow_pickle=True)
        model = cls(halflives=tuple(float(x) for x in z["halflives"]))
        model.W = z["W"]
        model.b = z["b"]
        model.ridge_per_target = z["ridge_per_target"]
        model.train_residual_scale = z["train_residual_scale"]
        model.clusters = [[int(v) for v in s.split(",")] for s in z["clusters"].tolist()]
        return model


def _mad(a: np.ndarray, axis: int = 0) -> np.ndarray:
    """Median absolute deviation (đã nhân 1.4826 để xấp xỉ độ lệch chuẩn)."""
    med = np.median(a, axis=axis)
    return 1.4826 * np.median(np.abs(a - med), axis=axis) + 1e-12
