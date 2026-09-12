"""Tính residual của mô hình quan hệ tuyến tính cho một tập dữ liệu.

    # tập test công khai
    python scripts/score.py --dataset public_test
    # tập kiểm định = train6 + tấn công giả lập (3 lần tiêm khác nhau)
    python scripts/score.py --dataset synth --seeds 0 1 2

Kết quả:
    outputs/res_relation_<tag>.npy       residual có dấu (n, F)
    outputs/labels_synth_seedK.npy       nhãn thật của tấn công giả lập
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.pipeline import load_artifacts, prepare_sets, residual_path
from icsad.relation import RelationModel
from icsad.utils import save_json, stdout_utf8, timed


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="public_test",
                    choices=["public_test", "private_test", "synth", "train"])
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="outputs/relation.npz")
    ap.add_argument("--tag", default="relation", help="tiền tố tên tệp residual")
    ap.add_argument("--synth-file", default="train6.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rate-per-hour", type=float, default=0.7)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    scaler, clusters = load_artifacts(out_dir)
    model = RelationModel.load(args.model)

    with timed(f"chuẩn bị dữ liệu ({args.dataset})"):
        sets = prepare_sets(args.dataset, args.data_dir, scaler, clusters,
                            seeds=args.seeds, synth_file=args.synth_file,
                            rate_per_hour=args.rate_per_hour, out_dir=out_dir)

    for s in sets:
        with timed(f"residual {s.tag}"):
            res = model.residuals(s.X)
        np.save(residual_path(out_dir, args.tag, s.tag), res)
        print(f"  |residual| trung bình = {np.abs(res).mean():.5f}")
        if s.meta and "n_rows" in s.meta:
            save_json(s.meta, out_dir / f"meta_{s.tag}.json")


if __name__ == "__main__":
    main()
