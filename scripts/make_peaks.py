"""Sinh submission bằng cách đặt N cửa sổ cố định quanh N đỉnh cao nhất.

Đây là cách tham số hoá gọn nhất cho việc dò bằng bảng xếp hạng, vì nó tách bạch
đúng hai thứ mà eTaPR quan tâm:

    --n      : số đoạn dự đoán  (đánh đổi precision <-> recall)
    --width  : độ rộng mỗi đoạn (phải cùng cỡ với độ dài tấn công thật)

Thuật toán: lấy điểm cao nhất, đặt một cửa sổ rộng `width` quanh nó, xoá vùng
`suppress` quanh đỉnh đó rồi lặp lại. Nhờ vậy các đoạn không dính nhau và trải
đều theo độ tin cậy giảm dần.

    python scripts/make_peaks.py --n 12 --width 240 --out submissions/K_n12_w240.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import Scaler
from icsad.etapr import labels_to_ranges
from icsad.pipeline import build_score
from icsad.postprocess import smooth, summarize
from icsad.scoring import normalize
from icsad.utils import stdout_utf8


def pick_peaks(s: np.ndarray, n: int, width: int, suppress: int) -> list[tuple[int, int]]:
    work = s.copy()
    half = width // 2
    out: list[tuple[int, int]] = []
    for _ in range(n):
        p = int(np.argmax(work))
        if not np.isfinite(work[p]) or work[p] <= -np.inf:
            break
        a, b = max(0, p - half), min(len(s) - 1, p - half + width - 1)
        out.append((a, b))
        work[max(0, p - suppress) : min(len(s), p + suppress + 1)] = -np.inf
    # gộp các cửa sổ chạm nhau để không lãng phí "suất" cho cùng một sự kiện
    out.sort()
    merged: list[list[int]] = []
    for a, b in out:
        if merged and a - merged[-1][1] <= 5:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models", nargs="+", default=["relation", "tcn"])
    ap.add_argument("--dataset", default="public_test")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--norm", default="rank")
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--fuse", default="feature_mean")
    ap.add_argument("--smooth", type=int, default=60)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--width", type=int, default=240)
    ap.add_argument("--suppress", type=int, default=None,
                    help="bán kính triệt tiêu quanh mỗi đỉnh (mặc định = width // 2)")
    ap.add_argument("--out", default="submissions/peaks.csv")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    paths = [out_dir / f"res_{m}_{args.dataset}.npy" for m in args.models]
    s = smooth(build_score(paths, args.norm, args.topk, (60, 300), args.fuse), args.smooth)

    ranges = pick_peaks(s, args.n, args.width, args.suppress or max(args.width // 2, 30))
    labels = np.zeros(len(s), dtype=np.int8)
    for a, b in ranges:
        labels[a : b + 1] = 1
    info = summarize(labels)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(("row_id", "anomaly"))
        for i, v in enumerate(labels):
            w.writerow((i, int(v)))

    print(f"{out}: {info['n_segments']} đoạn rộng {args.width}s, {info['ratio']:.2%} số điểm")
    if not args.quiet:
        names = Scaler.load(out_dir / "scaler.json").kept_columns
        Z = np.mean([normalize(np.load(p), mode=args.norm) for p in paths], axis=0)
        hh = lambda i: f"{i // 3600:02d}:{(i % 3600) // 60:02d}"
        for a, b in labels_to_ranges(labels):
            top = np.argsort(-Z[a : b + 1].mean(0))[:3]
            print(f"   {hh(a)}-{hh(b)} đỉnh={s[a:b+1].max():5.2f} | " + ", ".join(names[i] for i in top))


if __name__ == "__main__":
    main()
