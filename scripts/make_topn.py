"""Sinh submission từ N đoạn nghi vấn *chắc chắn nhất*, mỗi đoạn bị giới hạn độ dài.

Phản hồi từ bảng xếp hạng public cho thấy hai điều:
  * tấn công thật NGẮN - đoạn dự đoán dài ~10 phút bị eTaPR loại sạch (0 điểm),
  * precision quyết định - 5 đoạn đặt đúng hơn hẳn 23 đoạn.

Nên cách sinh submission hợp lý là: xếp hạng các đoạn nghi vấn theo độ tin cậy
(đỉnh điểm bất thường), lấy N đoạn đầu, và cắt mỗi đoạn về tối đa `max-len` giây
quanh chỗ điểm cao nhất.

    python scripts/make_topn.py --n 8 --max-len 300 --out submissions/S_n8.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import Scaler
from icsad.etapr import labels_to_ranges, ranges_to_labels
from icsad.pipeline import build_score
from icsad.postprocess import cap_length, drop_short, merge_gaps, smooth, summarize
from icsad.scoring import normalize
from icsad.utils import stdout_utf8


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["relation", "tcn", "nn"])
    ap.add_argument("--dataset", default="public_test")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--norm", default="rank")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--fuse", default="feature_mean")
    ap.add_argument("--smooth", type=int, default=60)
    ap.add_argument("--base-th", type=float, default=3.5,
                    help="ngưỡng thấp để tách đoạn ứng viên (xếp hạng sau)")
    ap.add_argument("--n", type=int, default=8, help="số đoạn giữ lại")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=300, help="độ dài tối đa mỗi đoạn")
    ap.add_argument("--max-gap", type=int, default=120)
    ap.add_argument("--out", default="submissions/topn.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    paths = [out_dir / f"res_{m}_{args.dataset}.npy" for m in args.models]
    score = build_score(paths, args.norm, args.topk, (60, 300), args.fuse)
    s = smooth(score, args.smooth)

    # tách đoạn ứng viên ở ngưỡng thấp rồi xếp hạng theo đỉnh
    cand = (s > args.base_th).astype(np.int8)
    cand = merge_gaps(cand, args.max_gap)
    cand = drop_short(cand, args.min_len)
    ranges = labels_to_ranges(cand)
    peaks = [float(s[a : b + 1].max()) for a, b in ranges]
    order = np.argsort(-np.asarray(peaks))[: args.n]
    chosen = sorted(ranges[i] for i in order)

    labels = cap_length(ranges_to_labels(chosen, len(s)), s, args.max_len)
    info = summarize(labels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("row_id", "anomaly"))
        for i, v in enumerate(labels):
            w.writerow((i, int(v)))

    print(f"{out}: {info['n_segments']} đoạn, {info['ratio']:.2%} số điểm, "
          f"dài min {info['len_min']}s / TV {info['len_median']:.0f}s / max {info['len_max']}s")
    if not args.quiet:
        names = Scaler.load(out_dir / "scaler.json").kept_columns
        Z = np.mean([normalize(np.load(p), mode=args.norm) for p in paths], axis=0)
        hh = lambda i: f"{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
        for a, b in labels_to_ranges(labels):
            top = np.argsort(-Z[a : b + 1].mean(0))[:3]
            print(f"   {hh(a)}-{hh(b)} ({b - a + 1:4d}s) đỉnh={s[a:b+1].max():5.2f} | "
                  + ", ".join(names[i] for i in top))


if __name__ == "__main__":
    main()
