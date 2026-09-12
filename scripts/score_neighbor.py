"""Huấn luyện + chấm điểm mô hình k-láng-giềng (trạng thái chưa từng thấy).

    python scripts/score_neighbor.py --dataset public_test --device cuda
    python scripts/score_neighbor.py --dataset synth --seeds 0 1 2 3 4 5

Mô hình chỉ gồm tập tham chiếu nên "huấn luyện" là tức thì; lần chạy đầu sẽ tạo
outputs/neighbor.npz rồi những lần sau dùng lại.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import load_training
from icsad.novelty import NeighborModel
from icsad.pipeline import load_artifacts, prepare_sets, residual_path
from icsad.utils import save_json, stdout_utf8, timed, torch_device


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="public_test",
                    choices=["public_test", "private_test", "synth", "train"])
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--tag", default="nn")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--holdout", default="train6.csv")
    ap.add_argument("--smooth-window", type=int, default=60)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--pca-dim", type=int, default=24)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--synth-file", default="train6.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5])
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    device = torch_device(args.device)
    scaler, clusters = load_artifacts(out_dir)

    model_path = out_dir / "neighbor.npz"
    if model_path.exists() and not args.refit:
        model = NeighborModel.load(model_path)
        print(f"dùng lại {model_path}: {len(model.ref_proj):,} trạng thái tham chiếu")
    else:
        with timed("dựng tập tham chiếu"):
            segments, _ = load_training(args.data_dir)
            holdout = None if args.holdout.lower() == "none" else args.holdout
            arrays = [scaler.transform(s.values) for s in segments if s.name != holdout]
            model = NeighborModel(smooth_window=args.smooth_window, stride=args.stride,
                                  pca_dim=args.pca_dim, k=args.k).fit(arrays)
            model.save(model_path)

    with timed(f"chuẩn bị dữ liệu ({args.dataset})"):
        sets = prepare_sets(args.dataset, args.data_dir, scaler, clusters,
                            seeds=args.seeds, synth_file=args.synth_file, out_dir=out_dir)

    for s in sets:
        with timed(f"residual {s.tag} ({len(s.X):,} dòng)"):
            res = model.residuals(s.X, device=device)
        np.save(residual_path(out_dir, args.tag, s.tag), res)
        print(f"  |residual| trung bình = {np.abs(res).mean():.5f}")
        if s.meta and "n_rows" in s.meta:
            save_json(s.meta, out_dir / f"meta_{s.tag}.json")


if __name__ == "__main__":
    main()
