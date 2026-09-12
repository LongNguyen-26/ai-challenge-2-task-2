"""Huấn luyện TCN che-và-tái tạo trên dữ liệu vận hành bình thường.

    # tự chọn GPU nếu có, không thì CPU
    python scripts/train_tcn.py --data-dir release --out-dir outputs --epochs 20
    # ép dùng CPU / GPU
    python scripts/train_tcn.py --device cpu
    python scripts/train_tcn.py --device cuda --batch-size 32

Cấu hình mặc định (160 kênh, 8 tầng giãn, cửa sổ 512) chỉ chiếm ~1.5 GB VRAM với
batch 32, chạy tốt trên GPU 4 GB. Trên CPU thì nên giảm --steps-per-epoch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.data import Scaler, correlation_clusters, load_training
from icsad.tcn import TCNConfig, save_model, train_tcn
from icsad.utils import load_json, save_json, stdout_utf8, timed, torch_device


def main() -> None:
    stdout_utf8()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="release")
    ap.add_argument("--out-dir", default="outputs")
    ap.add_argument("--holdout", default="train6.csv")
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--channels", type=int, default=160)
    ap.add_argument("--dilations", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--steps-per-epoch", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-amp", action="store_true")
    ap.add_argument("--threads", type=int, default=8, help="số luồng CPU cho torch")
    ap.add_argument("--name", default="tcn", help="tên tệp mô hình (outputs/<name>.pt)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    import torch

    torch.set_num_threads(args.threads)
    device = torch_device(args.device)

    with timed("nạp dữ liệu"):
        segments, columns = load_training(args.data_dir)
        scaler_path = out_dir / "scaler.json"
        if scaler_path.exists():
            scaler = Scaler.load(scaler_path)
        else:
            scaler = Scaler.fit(segments, columns)
            scaler.save(scaler_path)
        arrays = {s.name: scaler.transform(s.values) for s in segments}
        cl_path = out_dir / "clusters.json"
        if cl_path.exists():
            clusters = [list(map(int, c)) for c in load_json(cl_path)]
        else:
            clusters = correlation_clusters(np.concatenate([a[::5] for a in arrays.values()]), 0.995)
            save_json(clusters, cl_path)
        print(f"  {len(arrays)} tệp, {len(scaler.keep)} tín hiệu, {len(clusters)} cụm che")

    holdout = None if args.holdout.lower() == "none" else args.holdout
    train_arrays = [a for name, a in arrays.items() if name != holdout]
    val_arrays = [arrays[holdout]] if holdout in arrays else None

    # trọng số loss: cân bằng đóng góp giữa các tín hiệu (tín hiệu ít biến thiên
    # thì sai số tuyệt đối nhỏ, nếu không cân bằng sẽ bị bỏ qua)
    allX = np.concatenate([a[::10] for a in train_arrays])
    fw = (1.0 / (allX.std(axis=0) + 0.02)).astype(np.float32)
    fw = fw / fw.mean()
    del allX

    cfg = TCNConfig(
        n_features=len(scaler.keep),
        channels=args.channels,
        dilations=tuple(args.dilations),
        window=args.window,
        batch_size=args.batch_size,
        steps_per_epoch=args.steps_per_epoch,
        epochs=args.epochs,
        lr=args.lr,
        dropout=args.dropout,
        amp=not args.no_amp,
        seed=args.seed,
    )
    print(f"cấu hình: {cfg}")
    print(f"tầm nhìn (receptive field) = +-{cfg.receptive_field // 2} giây")

    t0 = time.perf_counter()
    model = train_tcn(train_arrays, clusters, cfg, device, val_arrays=val_arrays, feature_weight=fw)
    dt = time.perf_counter() - t0
    n_par = sum(p.numel() for p in model.parameters())
    print(f"huấn luyện xong sau {dt / 60:.1f} phút, {n_par:,} tham số")

    path = out_dir / f"{args.name}.pt"
    save_model(model, path, extra={"feature_weight": fw.tolist(), "holdout": holdout,
                                   "train_minutes": dt / 60})
    print(f"Đã lưu {path}")


if __name__ == "__main__":
    main()
