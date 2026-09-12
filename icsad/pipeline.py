"""Các bước dùng chung giữa những script chấm điểm.

Mục tiêu: mọi mô hình (tuyến tính hay TCN) đều sinh ra cùng một thứ - ma trận
residual (n, F) lưu ở outputs/res_<model>_<tag>.npy - nên phần chuẩn hoá, chọn
ngưỡng và tạo submission dùng chung được cho tất cả.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .data import Scaler, load_test, load_training
from .scoring import aggregate, boundary_guard, normalize
from .synth import SynthConfig, describe, inject
from .utils import load_json, save_json

FUSE_MODES = ("feature_mean", "feature_max", "score_mean", "score_max", "score_vote")


@dataclass
class ScoringSet:
    """Một chuỗi cần chấm điểm."""

    tag: str
    X: np.ndarray                    # (n, F) đã chuẩn hoá
    labels: np.ndarray | None = None  # nhãn thật (chỉ có với tấn công giả lập)
    meta: dict | None = None


def load_artifacts(out_dir: str | Path) -> tuple[Scaler, list[list[int]]]:
    out_dir = Path(out_dir)
    scaler = Scaler.load(out_dir / "scaler.json")
    clusters = [[int(i) for i in c] for c in load_json(out_dir / "clusters.json")]
    return scaler, clusters


def prepare_sets(
    dataset: str,
    data_dir: str | Path,
    scaler: Scaler,
    clusters: Sequence[Sequence[int]],
    seeds: Sequence[int] = (0, 1, 2),
    synth_file: str = "train6.csv",
    rate_per_hour: float = 0.7,
    out_dir: str | Path | None = None,
    verbose: bool = True,
) -> list[ScoringSet]:
    """Chuẩn bị dữ liệu cần chấm.

    dataset:
        public_test / private_test : tập kiểm tra thật
        synth                      : dữ liệu bình thường + tấn công giả lập
        train                      : toàn bộ tệp train (để xem phân bố residual)
    """
    out = []
    if dataset in ("public_test", "private_test"):
        seg, _ = load_test(data_dir, dataset, columns=scaler.columns)
        meta = {"n_rows": int(len(seg)), "t_start": str(seg.timestamps[0]),
                "t_end": str(seg.timestamps[-1])}
        if verbose:
            print(f"  {dataset}: {len(seg):,} dòng, {seg.timestamps[0]} -> {seg.timestamps[-1]}")
        out.append(ScoringSet(dataset, scaler.transform(seg.values), None, meta))

    elif dataset == "synth":
        segments, _ = load_training(data_dir, names=[synth_file])
        X0 = scaler.transform(segments[0].values)
        for seed in seeds:
            cfg = SynthConfig(seed=seed, rate_per_hour=rate_per_hour)
            Xa, labels, attacks = inject(X0, clusters, cfg)
            if verbose:
                print(f"  [seed {seed}] {describe(attacks)}")
            meta = {"attacks": [{"start": a.start, "end": a.end, "kind": a.kind,
                                 "features": [scaler.kept_columns[f] for f in a.features],
                                 "magnitude": round(a.magnitude, 3),
                                 "transient": a.transient} for a in attacks]}
            if out_dir is not None:
                np.save(Path(out_dir) / f"labels_synth_seed{seed}.npy", labels)
                save_json(meta["attacks"], Path(out_dir) / f"attacks_synth_seed{seed}.json")
            out.append(ScoringSet(f"synth_seed{seed}", Xa, labels, meta))

    elif dataset == "train":
        segments, _ = load_training(data_dir)
        for seg in segments:
            out.append(ScoringSet(seg.name.replace(".csv", ""), scaler.transform(seg.values)))

    else:
        raise ValueError(f"dataset không hợp lệ: {dataset}")
    return out


def residual_path(out_dir: str | Path, model: str, tag: str) -> Path:
    return Path(out_dir) / f"res_{model}_{tag}.npy"


def build_score(
    res_paths: Sequence[str | Path],
    norm: str = "rank",
    topk: int = 3,
    guard: tuple[int, int] = (60, 300),
    fuse: str = "feature_mean",
    feature_smooth: int = 0,
) -> np.ndarray:
    """Residual của một hay nhiều mô hình -> một chuỗi điểm bất thường.

    Args:
        feature_smooth: làm trơn |residual| của *từng tín hiệu* trước khi chuẩn
            hoá. Đây là điểm khác biệt quan trọng so với làm trơn chuỗi điểm
            cuối: tấn công làm gãy quan hệ **liên tục** ở một vài tín hiệu, còn
            nhiễu vận hành chỉ là những nhát nhọn ở các tín hiệu khác nhau. Làm
            trơn trước khi gộp top-k sẽ chỉ giữ lại loại thứ nhất.
        fuse: cách ghép nhiều mô hình -
            feature_mean / feature_max : ghép ở mức *từng tín hiệu* rồi gộp top-k
            score_mean   / score_max   : gộp top-k riêng từng mô hình rồi ghép
    """
    zs = []
    for path in res_paths:
        R = np.abs(np.load(Path(path)))
        if feature_smooth > 1:
            R = _smooth_columns(R, feature_smooth)
        zs.append(normalize(R, mode=norm))
    if len(zs) == 1:
        score = aggregate(zs[0], topk=topk)
    elif fuse == "feature_mean":
        score = aggregate(np.mean(zs, axis=0), topk=topk)
    elif fuse == "feature_max":
        score = aggregate(np.maximum.reduce(zs), topk=topk)
    elif fuse == "score_mean":
        score = np.mean([aggregate(z, topk=topk) for z in zs], axis=0)
    elif fuse == "score_max":
        score = np.maximum.reduce([aggregate(z, topk=topk) for z in zs])
    elif fuse == "score_vote":
        # điểm thứ nhì trong các mô hình: chỉ cao khi >= 2 mô hình cùng báo động,
        # nên một mô hình nhiễu riêng lẻ không kéo được điểm lên
        parts = np.sort(np.stack([aggregate(z, topk=topk) for z in zs]), axis=0)
        score = parts[-2] if len(zs) >= 2 else parts[-1]
    else:
        raise ValueError(f"fuse không hợp lệ: {fuse}")
    return boundary_guard(score.astype(np.float32), *guard)


def _smooth_columns(a: np.ndarray, window: int) -> np.ndarray:
    """Trung bình trượt có tâm theo từng cột (nhanh, dùng cumsum)."""
    n = len(a)
    half = window // 2
    csum = np.concatenate([np.zeros((1, a.shape[1])), np.cumsum(a, axis=0, dtype=np.float64)])
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + (window - half), 0, n)
    out = (csum[hi] - csum[lo]) / np.maximum(hi - lo, 1)[:, None]
    return out.astype(np.float32)
