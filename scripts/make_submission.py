"""Sinh tệp nộp bài predictions.csv từ residual của (các) mô hình.

    python scripts/make_submission.py --dataset public_test \
        --residuals outputs/res_relation_public_test.npy \
        --params outputs/post_params.json --out predictions.csv

Tuỳ chọn hữu ích:
    --th               ghi đè ngưỡng th_hi
    --target-ratio R   tự dò ngưỡng sao cho tỉ lệ điểm bị gắn cờ ~ R (vd 0.05)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import load_test
from icsad.etapr import labels_to_ranges
from icsad.pipeline import build_score
from icsad.postprocess import summarize
from icsad.tune import PostParams
from icsad.utils import load_json, stdout_utf8


def solve_threshold(score: np.ndarray, params: PostParams, target_ratio: float) -> float:
    """Dò nhị phân ngưỡng để tỉ lệ điểm bị gắn cờ xấp xỉ target_ratio."""
    lo, hi = float(np.quantile(score, 0.5)), float(score.max())
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        p = PostParams(**{**params.to_dict(), "th_hi": mid})
        ratio = float(p.apply(score).mean())
        if ratio > target_ratio:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="public_test")
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs", help="thư mục kết quả (chỉ dùng để tương thích)")
    ap.add_argument("--residuals", nargs="+", required=True)
    ap.add_argument("--params", default="outputs/post_params.json")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--th", type=float, default=None, help="ghi đè ngưỡng th_hi")
    ap.add_argument("--target-ratio", type=float, default=None,
                    help="tự dò ngưỡng để đạt tỉ lệ gắn cờ này (0-1)")
    ap.add_argument("--n-rows", type=int, default=None,
                    help="số dòng của test.csv (mặc định: đọc từ dữ liệu gốc)")
    args = ap.parse_args()

    cfg = load_json(args.params)
    params = PostParams(**cfg["post"])
    guard = (cfg.get("guard_zero", 60), cfg.get("guard_ramp", 300))

    score = build_score([Path(p) for p in args.residuals], cfg["norm"], cfg["topk"], guard,
                        cfg.get("fuse", "feature_mean"))
    print(f"chuẩn hoá={cfg['norm']} topk={cfg['topk']} fuse={cfg.get('fuse', '-')} | "
          f"điểm: trung vị={np.median(score):.2f} "
          f"p99={np.quantile(score, 0.99):.2f} max={score.max():.2f}")

    if args.target_ratio is not None:
        params.th_hi = solve_threshold(score, params, args.target_ratio)
        print(f"ngưỡng tự dò cho tỉ lệ {args.target_ratio:.1%}: th_hi={params.th_hi:.3f}")
    elif args.th is not None:
        params.th_hi = args.th
    print(f"tham số hậu xử lý: {params}")

    labels = params.apply(score)
    info = summarize(labels)
    print(f"kết quả: {info['n_segments']} đoạn, {info['flagged']:,} điểm bị gắn cờ "
          f"({info['ratio']:.2%}), độ dài đoạn: min={info['len_min']}s "
          f"trung vị={info['len_median']:.0f}s max={info['len_max']}s")

    n_rows = args.n_rows
    if n_rows is None:
        try:
            seg, _ = load_test(args.data_dir, args.dataset)
            n_rows = len(seg)
        except Exception as exc:  # pragma: no cover - chỉ để cảnh báo
            print(f"(!) không đọc được dữ liệu gốc để kiểm tra số dòng: {exc}")
            n_rows = len(labels)
    if n_rows != len(labels):
        raise SystemExit(f"Số dòng không khớp: điểm có {len(labels)}, test.csv có {n_rows}")

    out = Path(args.out)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("row_id", "anomaly"))
        for i, v in enumerate(labels):
            w.writerow((i, int(v)))
    print(f"Đã ghi {out} ({len(labels):,} dòng)")

    hhmmss = lambda i: f"{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
    print("Các đoạn dự đoán:")
    for s, e in labels_to_ranges(labels):
        print(f"   row {s:6d}-{e:6d}  ({hhmmss(s)}-{hhmmss(e)}, {e - s + 1}s)  đỉnh={score[s:e+1].max():.2f}")


if __name__ == "__main__":
    main()
