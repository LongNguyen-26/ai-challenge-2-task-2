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

# Cụm công tắc chế độ của tổ P2: 5 tín hiệu nhị phân luôn đổi cùng lúc khi người
# trực chuyển auto/manual. Mô hình nào cũng coi đó là "hiếm gặp" nên hay báo động,
# nhưng đó là thao tác vận hành bình thường.
SWITCH_FEATURES = ("P2_MASW", "P2_MASW_Lamp", "P2_ManualGO", "P2_AutoGO", "P2_ATSW_Lamp")


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
    ap.add_argument("--n", type=int, default=0, help="số đoạn giữ lại (0 = tính theo --rate)")
    ap.add_argument("--rate", type=float, default=12.0,
                    help="số đoạn trên mỗi 24 giờ dữ liệu - dùng khi không đặt --n. "
                         "Tập private có thể dài ngắn khác public nên đặt theo tỉ lệ sẽ an toàn hơn.")
    ap.add_argument("--shift", type=int, default=0, help="dịch toàn bộ các đoạn (giây)")
    ap.add_argument("--var-width", action="store_true",
                    help="độ rộng mỗi đoạn tỉ lệ với bề rộng vùng ứng viên (vẫn bị chặn bởi --max-len)")
    ap.add_argument("--min-len", type=int, default=60)
    ap.add_argument("--max-len", type=int, default=300, help="độ dài tối đa mỗi đoạn")
    ap.add_argument("--max-gap", type=int, default=120)
    ap.add_argument("--out", default="submissions/topn.csv")
    ap.add_argument("--exclude-switch", action="store_true",
                    help="bỏ cụm công tắc chế độ P2 khỏi việc tính điểm")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    paths = [out_dir / f"res_{m}_{args.dataset}.npy" for m in args.models]
    kept = Scaler.load(out_dir / "scaler.json").kept_columns
    exclude = [kept.index(c) for c in SWITCH_FEATURES if c in kept] if args.exclude_switch else None
    score = build_score(paths, args.norm, args.topk, (60, 300), args.fuse, exclude=exclude)
    s = smooth(score, args.smooth)

    # tách đoạn ứng viên ở ngưỡng thấp rồi xếp hạng theo đỉnh
    cand = (s > args.base_th).astype(np.int8)
    cand = merge_gaps(cand, args.max_gap)
    cand = drop_short(cand, args.min_len)
    ranges = labels_to_ranges(cand)
    peaks = [float(s[a : b + 1].max()) for a, b in ranges]
    n_keep = args.n if args.n > 0 else max(1, int(round(args.rate * len(s) / 86400.0)))
    order = np.argsort(-np.asarray(peaks))[:n_keep]
    chosen = sorted(ranges[i] for i in order)

    if args.var_width:
        # mỗi đoạn giữ một nửa bề rộng vùng ứng viên của nó, trong khoảng [min_len, max_len]
        trimmed = []
        for a, b in chosen:
            width = int(np.clip((b - a + 1) * 0.5, args.min_len, args.max_len))
            seg = s[a : b + 1].astype(np.float64)
            csum = np.concatenate(([0.0], np.cumsum(seg)))
            if len(seg) > width:
                best = int(np.argmax(csum[width:] - csum[:-width]))
                trimmed.append((a + best, a + best + width - 1))
            else:
                trimmed.append((a, b))
        labels = ranges_to_labels(trimmed, len(s))
    else:
        labels = cap_length(ranges_to_labels(chosen, len(s)), s, args.max_len)
    if args.shift:
        n_all = len(labels)
        labels = ranges_to_labels(
            [(max(0, a + args.shift), min(n_all - 1, b + args.shift)) for a, b in labels_to_ranges(labels)],
            n_all,
        )
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
        names = kept
        Z = np.mean([normalize(np.load(p), mode=args.norm) for p in paths], axis=0)
        hh = lambda i: f"{i // 3600:02d}:{(i % 3600) // 60:02d}:{i % 60:02d}"
        for a, b in labels_to_ranges(labels):
            top = np.argsort(-Z[a : b + 1].mean(0))[:3]
            print(f"   {hh(a)}-{hh(b)} ({b - a + 1:4d}s) đỉnh={s[a:b+1].max():5.2f} | "
                  + ", ".join(names[i] for i in top))


if __name__ == "__main__":
    main()
