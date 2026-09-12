"""Huấn luyện mô hình quan hệ tuyến tính (Ridge) trên dữ liệu vận hành bình thường.

Chạy hoàn toàn bằng CPU, khoảng 1-3 phút và < 1 GB RAM cho toàn bộ 1.0 triệu dòng.

    python scripts/train_relation.py --data-dir release --out-dir outputs

Kết quả lưu trong out-dir:
    scaler.json        min-max scaler + danh sách cột giữ lại
    clusters.json      các cụm tín hiệu gần trùng nhau
    relation.npz       trọng số mô hình
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import Scaler, correlation_clusters, load_training
from icsad.relation import DEFAULT_RIDGE_GRID, RelationModel
from icsad.utils import save_json, stdout_utf8, timed


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="release", help="thư mục chứa training.zip")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--holdout", default="train6.csv",
                    help="tệp để dành làm kiểm định (không dùng khi khớp hệ số), 'none' để dùng hết")
    ap.add_argument("--cluster-threshold", type=float, default=0.995)
    ap.add_argument("--halflives", type=float, nargs="+", default=[20.0, 300.0])
    ap.add_argument("--chunk", type=int, default=20000)
    ap.add_argument("--exclude-current", action="store_true",
                    help="chỉ dùng bối cảnh EWMA, bỏ giá trị tức thời của mọi tín hiệu")
    ap.add_argument("--name", default="relation", help="tên tệp mô hình (outputs/<name>.npz)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with timed("nạp dữ liệu train"):
        segments, columns = load_training(args.data_dir)
        print(f"  {len(segments)} tệp, tổng {sum(len(s) for s in segments):,} dòng, {len(columns)} cột tín hiệu")

    with timed("chuẩn hoá"):
        scaler = Scaler.fit(segments, columns)
        print(f"  giữ {len(scaler.keep)}/{len(columns)} cột "
              f"(bỏ {len(columns) - len(scaler.keep)} cột hằng số: "
              f"{', '.join(c for i, c in enumerate(columns) if i not in set(scaler.keep))})")
        arrays = {s.name: scaler.transform(s.values) for s in segments}
        scaler.save(out_dir / "scaler.json")

    with timed("gom cụm tín hiệu tương quan cao"):
        sample = np.concatenate([a[::5] for a in arrays.values()])
        clusters = correlation_clusters(sample, args.cluster_threshold)
        del sample
        big = sorted((c for c in clusters if len(c) > 1), key=len, reverse=True)
        print(f"  {len(clusters)} cụm; {len(big)} cụm có >1 phần tử")
        names = scaler.kept_columns
        for c in big[:12]:
            print("   - " + ", ".join(names[i] for i in c))
        save_json([[int(i) for i in c] for c in clusters], out_dir / "clusters.json")

    holdout = None if args.holdout.lower() == "none" else args.holdout
    train_arrays = [a for name, a in arrays.items() if name != holdout]
    val_arrays = [arrays[holdout]] if holdout in arrays else None
    print(f"train: {[n for n in arrays if n != holdout]} | kiểm định: {holdout}")

    with timed("huấn luyện mô hình quan hệ"):
        model = RelationModel(halflives=tuple(args.halflives), chunk=args.chunk,
                              exclude_current=args.exclude_current)
        model.fit(train_arrays, clusters=clusters, val_arrays=val_arrays,
                  ridge_grid=DEFAULT_RIDGE_GRID)
        model.save(out_dir / f"{args.name}.npz")

    # residual trên tập kiểm định: tín hiệu nào khó dự đoán nhất?
    if val_arrays:
        res = np.abs(model.residuals(val_arrays[0]))
        order = np.argsort(-res.mean(axis=0))
        names = scaler.kept_columns
        print("Các tín hiệu khó dự đoán nhất (MAE trên tập kiểm định, đơn vị min-max):")
        for i in order[:10]:
            print(f"   {names[i]:<14} {res[:, i].mean():.5f}")
        print("Các tín hiệu dự đoán chính xác nhất:")
        for i in order[-10:]:
            print(f"   {names[i]:<14} {res[:, i].mean():.6f}")

    print(f"\nĐã lưu mô hình vào {out_dir}")


if __name__ == "__main__":
    main()
