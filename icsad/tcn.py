"""TCN che-và-tái tạo (masked reconstruction) cho telemetry ICS.

Ý tưởng giống mô hình quan hệ tuyến tính nhưng phi tuyến và có bộ nhớ thời gian:
mạng nhận toàn bộ chuỗi nhưng một *cụm tín hiệu bị che* (giá trị bị xoá, kèm cờ
báo "đang bị che"), nhiệm vụ là tái tạo lại chính những tín hiệu đó từ các tín
hiệu còn lại và từ bối cảnh thời gian hai phía.

    đầu vào : [x * (1 - m) ; m]    -> 2F kênh
    đầu ra  : x̂                    -> F kênh, chỉ tính loss ở phần bị che

Khi chấm điểm, ta chạy mạng nhiều lượt, mỗi lượt che một cụm, rồi ghép residual
của các tín hiệu trong cụm đó lại thành ma trận residual (n, F) - cùng định dạng
với mô hình quan hệ nên dùng chung toàn bộ phần chuẩn hoá/hậu xử lý.

Mạng là tích chập giãn (dilated conv) *không nhân quả*: mỗi điểm nhìn được
+-255 giây xung quanh với cấu hình mặc định.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # torch là phụ thuộc tuỳ chọn (phần mô hình tuyến tính không cần)
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    nn = object  # type: ignore


@dataclass
class TCNConfig:
    n_features: int = 68
    channels: int = 160
    kernel: int = 3
    dilations: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128)
    dropout: float = 0.0
    window: int = 512            # độ dài cửa sổ khi huấn luyện
    batch_size: int = 32
    steps_per_epoch: int = 1500
    epochs: int = 20
    lr: float = 2e-3
    weight_decay: float = 1e-5
    mask_probs: tuple[float, ...] = (0.7, 0.2, 0.1)   # che 1 / 2 / 3 cụm
    edge_margin: int = 64        # bỏ qua rìa cửa sổ khi tính loss
    amp: bool = True
    seed: int = 0

    @property
    def receptive_field(self) -> int:
        return 1 + 2 * (self.kernel - 1) * sum(self.dilations)


class ChannelNorm(nn.Module):
    """LayerNorm trên trục kênh tại *từng thời điểm*.

    Bắt buộc phải như vậy: GroupNorm/BatchNorm chuẩn hoá dọc theo trục thời gian
    nên thống kê phụ thuộc độ dài chuỗi. Mạng huấn luyện với cửa sổ 512 nhưng khi
    chấm điểm lại chạy trên cả chuỗi vài chục nghìn điểm -> thống kê khác hẳn,
    residual phình to giả tạo (đo được: |residual| 0.074 khi chạy đúng độ dài
    huấn luyện so với 0.196 khi chạy cả chuỗi). Chuẩn hoá theo kênh không dính
    vấn đề này.
    """

    def __init__(self, ch: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, ch, 1))
        self.bias = nn.Parameter(torch.zeros(1, ch, 1))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(dim=1, keepdim=True)
        var = x.var(dim=1, keepdim=True, unbiased=False)
        return (x - mu) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class ResBlock(nn.Module):
    def __init__(self, ch: int, kernel: int, dilation: int, dropout: float = 0.0):
        super().__init__()
        pad = dilation * (kernel - 1) // 2      # 'same' padding, không nhân quả
        self.conv1 = nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation)
        self.norm1 = ChannelNorm(ch)
        self.conv2 = nn.Conv1d(ch, ch, 1)
        self.norm2 = ChannelNorm(ch)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        h = self.act(self.norm1(self.conv1(x)))
        h = self.drop(h)
        h = self.norm2(self.conv2(h))
        return self.act(x + h)


class MaskedTCN(nn.Module):
    """Tái tạo các tín hiệu bị che từ phần còn lại."""

    def __init__(self, cfg: TCNConfig):
        super().__init__()
        self.cfg = cfg
        F, C = cfg.n_features, cfg.channels
        self.inp = nn.Conv1d(2 * F, C, 1)
        self.blocks = nn.ModuleList(
            [ResBlock(C, cfg.kernel, d, cfg.dropout) for d in cfg.dilations]
        )
        self.head = nn.Conv1d(C, F, 1)

    def forward(self, x: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
        """x, mask: (B, F, L) -> (B, F, L). mask = 1 tại chỗ bị che."""
        h = torch.cat([x * (1.0 - mask), mask], dim=1)
        h = self.inp(h)
        for blk in self.blocks:
            h = blk(h)
        return self.head(h)


# --------------------------------------------------------------------------- #
# huấn luyện
# --------------------------------------------------------------------------- #
def _sample_batch(
    arrays: Sequence[np.ndarray],
    probs: np.ndarray,
    clusters: Sequence[Sequence[int]],
    cfg: TCNConfig,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Bốc ngẫu nhiên một batch cửa sổ + mặt nạ che."""
    L, B, F = cfg.window, cfg.batch_size, cfg.n_features
    xb = np.empty((B, L, F), dtype=np.float32)
    mb = np.zeros((B, L, F), dtype=np.float32)
    n_masks = np.arange(1, len(cfg.mask_probs) + 1)
    for i in range(B):
        ai = int(rng.choice(len(arrays), p=probs))
        arr = arrays[ai]
        start = int(rng.integers(0, len(arr) - L))
        xb[i] = arr[start : start + L]
        k = int(rng.choice(n_masks, p=np.asarray(cfg.mask_probs)))
        chosen = rng.choice(len(clusters), size=min(k, len(clusters)), replace=False)
        for ci in chosen:
            mb[i][:, list(clusters[ci])] = 1.0
    return xb, mb


def train_tcn(
    arrays: Sequence[np.ndarray],
    clusters: Sequence[Sequence[int]],
    cfg: TCNConfig,
    device,
    val_arrays: Sequence[np.ndarray] | None = None,
    feature_weight: np.ndarray | None = None,
    log_every: int = 250,
) -> MaskedTCN:
    """Huấn luyện mô hình; trả về mạng đã huấn luyện (ở chế độ eval)."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = MaskedTCN(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = max(cfg.epochs * cfg.steps_per_epoch, 1)
    if total_steps >= 20:
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=cfg.lr, total_steps=total_steps, pct_start=0.1
        )
    else:  # chạy thử vài bước: giữ lr cố định
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    use_amp = bool(cfg.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    lens = np.array([len(a) for a in arrays], dtype=np.float64)
    probs = lens / lens.sum()
    if feature_weight is None:
        feature_weight = np.ones(cfg.n_features, dtype=np.float32)
    fw = torch.tensor(feature_weight, dtype=torch.float32, device=device).view(1, -1, 1)
    m0, m1 = cfg.edge_margin, cfg.window - cfg.edge_margin

    step = 0
    for epoch in range(cfg.epochs):
        model.train()
        running = 0.0
        for _ in range(cfg.steps_per_epoch):
            xb, mb = _sample_batch(arrays, probs, clusters, cfg, rng)
            x = torch.from_numpy(xb).to(device, non_blocking=True).transpose(1, 2)
            m = torch.from_numpy(mb).to(device, non_blocking=True).transpose(1, 2)
            with torch.autocast("cuda", enabled=use_amp):
                pred = model(x, m)
                err = (pred - x)[:, :, m0:m1] * m[:, :, m0:m1] * fw
                denom = (m[:, :, m0:m1] * fw).sum().clamp_min(1.0)
                loss = torch.nn.functional.huber_loss(
                    err, torch.zeros_like(err), delta=0.05, reduction="sum"
                ) / denom
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            running += float(loss.detach())
            step += 1
            if log_every and step % log_every == 0:
                print(f"  epoch {epoch + 1}/{cfg.epochs} step {step}/{total_steps} "
                      f"loss={running / log_every:.5f} lr={sched.get_last_lr()[0]:.2e}", flush=True)
                running = 0.0
        if val_arrays is not None:
            vl = validate(model, val_arrays, clusters, cfg, device, feature_weight, rng_seed=1234)
            print(f"  [epoch {epoch + 1}] loss kiểm định = {vl:.5f}", flush=True)
    model.eval()
    return model


def validate(model, val_arrays, clusters, cfg, device, feature_weight, rng_seed=0, n_batches=20) -> float:
    """MAE trên phần bị che của tập kiểm định."""
    model.eval()
    rng = np.random.default_rng(rng_seed)
    lens = np.array([len(a) for a in val_arrays], dtype=np.float64)
    probs = lens / lens.sum()
    fw = torch.tensor(feature_weight, dtype=torch.float32, device=device).view(1, -1, 1)
    m0, m1 = cfg.edge_margin, cfg.window - cfg.edge_margin
    tot = 0.0
    with torch.no_grad():
        for _ in range(n_batches):
            xb, mb = _sample_batch(val_arrays, probs, clusters, cfg, rng)
            x = torch.from_numpy(xb).to(device).transpose(1, 2)
            m = torch.from_numpy(mb).to(device).transpose(1, 2)
            pred = model(x, m)
            err = (pred - x) * m * fw
            tot += float(err.abs()[:, :, m0:m1].sum() / (m[:, :, m0:m1] * fw).sum().clamp_min(1.0))
    return tot / n_batches


# --------------------------------------------------------------------------- #
# suy luận
# --------------------------------------------------------------------------- #
def predict_residuals(
    model: MaskedTCN,
    X: np.ndarray,
    clusters: Sequence[Sequence[int]],
    device,
    chunk: int = 30000,
    overlap: int | None = None,
    batch_clusters: int = 1,
    verbose: bool = True,
) -> np.ndarray:
    """Residual (n, F): với mỗi tín hiệu, dùng lượt chạy mà chính nó bị che."""
    model.eval()
    cfg = model.cfg
    n, F = X.shape
    if overlap is None:
        overlap = cfg.receptive_field
    res = np.zeros((n, F), dtype=np.float32)

    groups: list[list[int]] = []
    for i in range(0, len(clusters), batch_clusters):
        merged: list[int] = []
        for c in clusters[i : i + batch_clusters]:
            merged.extend(c)
        groups.append(sorted(merged))

    with torch.no_grad():
        for gi, feats in enumerate(groups):
            for start in range(0, n, chunk):
                stop = min(start + chunk, n)
                lo, hi = max(0, start - overlap), min(n, stop + overlap)
                xb = torch.from_numpy(X[lo:hi].T[None]).to(device)     # (1, F, L)
                mb = torch.zeros_like(xb)
                mb[:, feats, :] = 1.0
                with torch.autocast("cuda", enabled=(device.type == "cuda" and cfg.amp)):
                    pred = model(xb, mb)
                out = (xb - pred.float())[0].T.cpu().numpy()           # (L, F)
                res[start:stop, feats] = out[start - lo : stop - lo][:, feats]
            if verbose and (gi + 1) % 10 == 0:
                print(f"  đã chạy {gi + 1}/{len(groups)} nhóm che", flush=True)
    return res


# --------------------------------------------------------------------------- #
# lưu / nạp
# --------------------------------------------------------------------------- #
def save_model(model: MaskedTCN, path: str | Path, extra: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": asdict(model.cfg), "state": model.state_dict(), "extra": extra or {}}, path)


def load_model(path: str | Path, device) -> MaskedTCN:
    blob = torch.load(path, map_location=device, weights_only=False)
    cfg_dict = dict(blob["cfg"])
    cfg_dict["dilations"] = tuple(cfg_dict["dilations"])
    cfg_dict["mask_probs"] = tuple(cfg_dict["mask_probs"])
    model = MaskedTCN(TCNConfig(**cfg_dict)).to(device)
    model.load_state_dict(blob["state"])
    model.eval()
    return model
