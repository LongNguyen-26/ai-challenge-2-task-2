"""Chọn cách chuẩn hoá điểm + tham số hậu xử lý bằng eTaPR trên tấn công giả lập.

    python scripts/tune_postprocess.py \
        --residuals "outputs/res_relation_synth_seed{seed}.npy" \
        --seeds 0 1 2 --out outputs/post_params.json

Nhiều mô hình thì truyền nhiều mẫu --residuals; khi đó cách ghép chúng lại
(`fuse`) cũng được duyệt luôn.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.pipeline import FUSE_MODES, build_score
from icsad.tune import DEFAULT_GRID, grid_search
from icsad.utils import save_json, stdout_utf8, timed

TH_GRID = {
    "mad": (4.0, 5.5, 7.0, 9.0, 12.0, 15.0, 19.0, 24.0, 30.0, 36.0),
    "rank": (1.5, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 7.0, 8.5, 10.0),
}


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--residuals", nargs="+", required=True,
                    help="mẫu đường dẫn residual, dùng {seed} làm chỗ thay thế")
    ap.add_argument("--labels", default="outputs/labels_synth_seed{seed}.npy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--norms", nargs="+", default=["mad", "rank"])
    ap.add_argument("--topk", type=int, nargs="+", default=[1, 3, 5, 10])
    ap.add_argument("--fuse", nargs="+", default=list(FUSE_MODES),
                    help="cách ghép nhiều mô hình (bỏ qua nếu chỉ có 1 residual)")
    ap.add_argument("--guard-zero", type=int, default=60)
    ap.add_argument("--guard-ramp", type=int, default=300)
    ap.add_argument("--out", default="outputs/post_params.json")
    args = ap.parse_args()

    labels = {s: np.load(args.labels.format(seed=s)) for s in args.seeds}
    guard = (args.guard_zero, args.guard_ramp)
    fuses = list(args.fuse) if len(args.residuals) > 1 else ["feature_mean"]

    best = None
    for norm in args.norms:
        for topk in args.topk:
            for fuse in fuses:
                pairs = []
                for seed in args.seeds:
                    files = [Path(p.format(seed=seed)) for p in args.residuals]
                    pairs.append((build_score(files, norm, topk, guard, fuse), labels[seed]))
                grid = dict(DEFAULT_GRID)
                grid["th_hi"] = TH_GRID[norm]
                with timed(f"tìm tham số | norm={norm} topk={topk} fuse={fuse}"):
                    params, top = grid_search(pairs, grid, top_k=3)
                f1 = top[0][0]
                if best is None or f1 > best[0]:
                    best = (f1, norm, topk, fuse, params, top[0][2])

    f1, norm, topk, fuse, params, info = best  # type: ignore[misc]
    print()
    print(f">>> Tốt nhất: F1={f1:.4f} | norm={norm} topk={topk} fuse={fuse}")
    print(f"    {params}")
    save_json(
        {
            "norm": norm,
            "topk": topk,
            "fuse": fuse,
            "guard_zero": args.guard_zero,
            "guard_ramp": args.guard_ramp,
            "post": params.to_dict(),
            "val_f1": f1,
            "val_info": info,
            "residual_patterns": args.residuals,
            "seeds": args.seeds,
        },
        args.out,
    )
    print(f"Đã lưu {args.out}")


if __name__ == "__main__":
    main()
