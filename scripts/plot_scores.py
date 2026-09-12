"""Vẽ điểm bất thường theo thời gian + bảng các đoạn dự đoán (để kiểm tra bằng mắt).

    python scripts/plot_scores.py --dataset public_test \
        --residuals outputs/res_relation_public_test.npy outputs/res_tcn_public_test.npy

Ảnh lưu ở outputs/score_<dataset>.png
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import Scaler
from icsad.etapr import labels_to_ranges
from icsad.pipeline import build_score
from icsad.postprocess import smooth
from icsad.scoring import normalize
from icsad.tune import PostParams
from icsad.utils import load_json, stdout_utf8


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="public_test")
    ap.add_argument("--residuals", nargs="+", required=True)
    ap.add_argument("--params", default="outputs/post_params.json")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--labels", default=None, help="tệp nhãn thật (nếu có, để đối chiếu)")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    cfg = load_json(args.params)
    params = PostParams(**cfg["post"])
    guard = (cfg.get("guard_zero", 60), cfg.get("guard_ramp", 300))
    score = build_score([Path(p) for p in args.residuals], cfg["norm"], cfg["topk"], guard,
                        cfg.get("fuse", "feature_mean"))
    labels = params.apply(score)
    ranges = labels_to_ranges(labels)

    names = Scaler.load(out_dir / "scaler.json").kept_columns
    Z = np.mean([normalize(np.load(p), mode=cfg["norm"]) for p in args.residuals], axis=0)

    hhmmss = lambda i: f"{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
    print(f"{len(ranges)} đoạn dự đoán ({labels.mean():.2%} số điểm):")
    for a, b in ranges:
        top = np.argsort(-Z[a : b + 1].mean(0))[:4]
        print(f"  {hhmmss(a)}-{hhmmss(b)} ({b - a + 1:5d}s) đỉnh={score[a:b+1].max():5.2f} | "
              + ", ".join(f"{names[i]}={Z[a:b+1, i].mean():.1f}" for i in top))

    t = np.arange(len(score)) / 3600.0
    smoothed = smooth(score, params.smooth_window, params.smooth_kind)
    fig, ax = plt.subplots(figsize=(16, 4.5))
    ax.plot(t, score, lw=0.4, color="0.75", label="điểm thô")
    ax.plot(t, smoothed, lw=0.7, color="C0",
            label=f"đã làm trơn {params.smooth_window}s (dùng để cắt ngưỡng)")
    ax.axhline(params.th_hi, color="r", ls="--", lw=0.8, label=f"ngưỡng {params.th_hi:g}")
    for a, b in ranges:
        ax.axvspan(a / 3600, b / 3600, color="red", alpha=0.15, lw=0)
    if args.labels:
        y = np.load(args.labels)
        for a, b in labels_to_ranges(y):
            ax.axvspan(a / 3600, b / 3600, color="green", alpha=0.12, lw=0)
    ax.set_xlabel("giờ"); ax.set_ylabel("điểm")
    ax.set_title(f"{args.dataset}: {len(ranges)} đoạn, {labels.sum():,} điểm bị gắn cờ "
                 f"({labels.mean():.2%})")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path = out_dir / f"score_{args.dataset}.png"
    fig.savefig(path, dpi=110)
    print(f"Đã lưu {path}")


if __name__ == "__main__":
    main()
