"""Chạy tuần tự toàn bộ quy trình: huấn luyện → chấm điểm → chọn ngưỡng → nộp bài.

    # chỉ mô hình tuyến tính (CPU, ~10 phút)
    python scripts/run_all.py --data-dir release --dataset public_test

    # thêm TCN (cần GPU cho nhanh)
    python scripts/run_all.py --data-dir release --dataset public_test --tcn --device cuda

Bỏ qua bước đã có kết quả bằng --skip-train / --skip-score.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from icsad.utils import stdout_utf8


def run(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    t0 = time.perf_counter()
    res = subprocess.run([sys.executable, *cmd], cwd=ROOT)
    if res.returncode != 0:
        raise SystemExit(f"Lệnh thất bại (mã {res.returncode})")
    print(f"  ({time.perf_counter() - t0:.0f}s)", flush=True)


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--dataset", default="public_test", choices=["public_test", "private_test"])
    ap.add_argument("--tcn", action="store_true", help="dùng thêm mô hình TCN")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--channels", type=int, default=160)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    seeds = [str(s) for s in args.seeds]
    base = ["--data-dir", args.data_dir, "--out-dir", args.out_dir]

    if not args.skip_train:
        run(["scripts/train_relation.py", *base])
        if args.tcn:
            run(["scripts/train_tcn.py", *base, "--device", args.device,
                 "--epochs", str(args.epochs), "--steps-per-epoch", str(args.steps_per_epoch),
                 "--channels", str(args.channels)])

    if not args.skip_score:
        run(["scripts/score.py", *base, "--dataset", "synth", "--seeds", *seeds])
        run(["scripts/score.py", *base, "--dataset", args.dataset])
        if args.tcn:
            run(["scripts/score_tcn.py", *base, "--dataset", "synth", "--seeds", *seeds,
                 "--device", args.device])
            run(["scripts/score_tcn.py", *base, "--dataset", args.dataset, "--device", args.device])

    models = ["relation"] + (["tcn"] if args.tcn else [])
    synth_res = [f"{args.out_dir}/res_{m}_synth_seed{{seed}}.npy" for m in models]
    test_res = [f"{args.out_dir}/res_{m}_{args.dataset}.npy" for m in models]

    run(["scripts/tune_postprocess.py", "--residuals", *synth_res, "--seeds", *seeds,
         "--labels", f"{args.out_dir}/labels_synth_seed{{seed}}.npy",
         "--out", f"{args.out_dir}/post_params.json"])
    run(["scripts/make_submission.py", *base, "--dataset", args.dataset,
         "--residuals", *test_res, "--params", f"{args.out_dir}/post_params.json",
         "--out", args.out])

    print(f"\nHoàn tất. Tệp nộp bài: {args.out}")


if __name__ == "__main__":
    main()
