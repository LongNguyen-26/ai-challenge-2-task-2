"""Tính residual của mô hình TCN (che-và-tái tạo) cho một tập dữ liệu.

    python scripts/score_tcn.py --dataset public_test --model outputs/tcn.pt
    python scripts/score_tcn.py --dataset synth --seeds 0 1 2 --model outputs/tcn.pt

Mỗi cụm tín hiệu được che một lượt rồi tái tạo, nên chi phí tỉ lệ với số cụm
(~55). Trên GPU mất vài chục giây cho một ngày dữ liệu; trên CPU vài phút - có
thể tăng --batch-clusters để chạy nhanh hơn (đổi lại độ nhạy giảm nhẹ vì mỗi
lượt che nhiều tín hiệu hơn).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.pipeline import load_artifacts, prepare_sets, residual_path
from icsad.tcn import load_model, predict_residuals
from icsad.utils import save_json, stdout_utf8, timed, torch_device


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="public_test",
                    choices=["public_test", "private_test", "synth", "train"])
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--model", default="outputs/tcn.pt")
    ap.add_argument("--tag", default="tcn", help="tiền tố tên tệp residual")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=30000)
    ap.add_argument("--batch-clusters", type=int, default=1)
    ap.add_argument("--synth-file", default="train6.csv")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--rate-per-hour", type=float, default=0.7)
    args = ap.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    out_dir = Path(args.out_dir)
    device = torch_device(args.device)
    scaler, clusters = load_artifacts(out_dir)
    model = load_model(args.model, device)
    print(f"mô hình: {args.model} | {sum(p.numel() for p in model.parameters()):,} tham số "
          f"| tầm nhìn +-{model.cfg.receptive_field // 2}s | {len(clusters)} cụm che")

    with timed(f"chuẩn bị dữ liệu ({args.dataset})"):
        sets = prepare_sets(args.dataset, args.data_dir, scaler, clusters,
                            seeds=args.seeds, synth_file=args.synth_file,
                            rate_per_hour=args.rate_per_hour, out_dir=out_dir)

    for s in sets:
        with timed(f"residual {s.tag} ({len(s.X):,} dòng)"):
            res = predict_residuals(model, s.X, clusters, device, chunk=args.chunk,
                                    batch_clusters=args.batch_clusters)
        np.save(residual_path(out_dir, args.tag, s.tag), res)
        print(f"  |residual| trung bình = {np.abs(res).mean():.5f}")
        if s.meta and "n_rows" in s.meta:
            save_json(s.meta, out_dir / f"meta_{s.tag}.json")


if __name__ == "__main__":
    main()
