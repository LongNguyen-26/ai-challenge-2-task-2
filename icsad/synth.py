"""Sinh tấn công giả lập trên dữ liệu bình thường.

Tập test không có nhãn, nên muốn chọn ngưỡng/hậu xử lý một cách có cơ sở ta cần
một tập kiểm định *có nhãn*. Cách làm: lấy một tệp train (dữ liệu bình thường)
chưa dùng để huấn luyện, tiêm vào đó các đoạn bất thường mô phỏng đúng những
kiểu can thiệp hay gặp trong ICS:

    freeze   : giữ nguyên giá trị (cảm biến bị "đóng băng"/replay)
    bias     : cộng thêm một lượng không đổi
    ramp     : trôi tuyến tính
    setpoint : ép về một hằng số khác (đổi setpoint)
    scale    : nhân biên độ quanh giá trị trung bình
    replay   : phát lại một đoạn khác của chính tín hiệu đó
    noise    : thêm nhiễu

Một nửa số tấn công được sinh ở dạng **quá độ hai đầu** (`transient_prob`): nhãn
phủ trọn [s, e] nhưng dữ liệu chỉ bị sửa ở một quãng ngắn đầu và một quãng ngắn
cuối. Đây là mô phỏng cho kiểu tấn công phổ biến nhất trong ICS: kẻ tấn công đổi
setpoint/lệnh điều khiển, vòng điều khiển *đáp ứng lại*, và sau giai đoạn quá độ
nhà máy chạy ổn định ở điểm làm việc mới - quan hệ giữa các tín hiệu lúc đó trông
lại bình thường. Nếu chỉ sinh loại tấn công "lệch liên tục", việc dò tham số sẽ
kết luận sai rằng không cần ghép khe hở và đoạn dự đoán nên ngắn.

Lưu ý: tấn công giả lập KHÔNG lan truyền qua động học của nhà máy như tấn công
thật, nên con số eTaPR ở đây chỉ dùng để *so sánh tương đối* các bộ tham số hậu
xử lý, không phải ước lượng điểm thi.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

KINDS = ("freeze", "bias", "ramp", "setpoint", "scale", "replay", "noise")


@dataclass
class SynthConfig:
    rate_per_hour: float = 0.7          # số tấn công trên mỗi giờ dữ liệu
    min_len: int = 120                  # giây
    max_len: int = 3600                 # giây
    min_gap: int = 300                  # khoảng cách tối thiểu giữa hai tấn công
    magnitude: tuple[float, float] = (0.3, 3.0)     # theo đơn vị "độ lệch chuẩn" của tín hiệu
    cluster_prob: float = 0.5           # xác suất tấn công cả cụm tín hiệu gần trùng
    min_effect: float = 0.01            # biên độ thay đổi tối thiểu (đơn vị min-max)
    transient_prob: float = 0.5         # tỉ lệ tấn công kiểu "chỉ gãy ở hai đầu"
    transient_len: tuple[int, int] = (30, 180)   # độ dài mỗi quá độ (giây)
    kinds: tuple[str, ...] = KINDS
    weights: tuple[float, ...] = (1.5, 1.5, 1.0, 1.5, 0.8, 1.0, 0.7)
    seed: int = 0


@dataclass
class SynthAttack:
    start: int
    end: int                 # inclusive
    features: list[int]
    kind: str
    magnitude: float
    transient: bool = False  # chỉ sửa dữ liệu ở hai đầu đoạn


def _apply(
    X: np.ndarray,
    sl: slice,
    feats: Sequence[int],
    kind: str,
    mag: float,
    std: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Sửa X tại chỗ cho một tấn công."""
    n = sl.stop - sl.start
    for f in feats:
        col = X[sl, f]
        s = float(std[f])
        sign = 1.0 if rng.random() < 0.5 else -1.0
        if kind == "freeze":
            X[sl, f] = X[max(sl.start - 1, 0), f]
        elif kind == "bias":
            X[sl, f] = col + sign * mag * s
        elif kind == "ramp":
            X[sl, f] = col + sign * mag * s * np.linspace(0.0, 1.0, n, dtype=np.float32)
        elif kind == "setpoint":
            target = float(np.quantile(X[:, f], rng.uniform(0.02, 0.98)))
            X[sl, f] = target
        elif kind == "scale":
            m = float(col.mean())
            factor = 1.0 + sign * min(mag, 2.0) * 0.5
            X[sl, f] = m + (col - m) * factor
        elif kind == "replay":
            src = int(rng.integers(0, max(len(X) - n, 1)))
            X[sl, f] = X[src : src + n, f]
        elif kind == "noise":
            X[sl, f] = col + rng.normal(0.0, mag * s, size=n).astype(np.float32)
        else:
            raise ValueError(f"kind không hợp lệ: {kind}")


def inject(
    X: np.ndarray,
    clusters: Sequence[Sequence[int]] | None = None,
    config: SynthConfig | None = None,
) -> tuple[np.ndarray, np.ndarray, list[SynthAttack]]:
    """Tiêm tấn công giả lập vào một đoạn dữ liệu bình thường đã chuẩn hoá.

    Returns:
        (X_bị_tấn_công, nhãn 0/1, danh sách mô tả tấn công)
    """
    cfg = config or SynthConfig()
    rng = np.random.default_rng(cfg.seed)
    Xa = X.copy()
    n, F = X.shape
    labels = np.zeros(n, dtype=np.int8)

    std = X.std(axis=0)
    std[std < 1e-4] = 1e-4          # tín hiệu gần như hằng: dùng mức tối thiểu
    cand = [f for f in range(F) if X[:, f].std() > 0]
    cluster_of: dict[int, list[int]] = {}
    for c in clusters or []:
        for m in c:
            cluster_of[m] = list(c)

    n_attacks = max(1, int(round(cfg.rate_per_hour * n / 3600.0)))
    weights = np.asarray(cfg.weights, dtype=np.float64)
    weights = weights / weights.sum()

    attacks: list[SynthAttack] = []
    occupied = np.zeros(n, dtype=bool)
    tries = 0
    while len(attacks) < n_attacks and tries < n_attacks * 50:
        tries += 1
        length = int(np.exp(rng.uniform(np.log(cfg.min_len), np.log(cfg.max_len))))
        start = int(rng.integers(cfg.min_gap, max(n - length - cfg.min_gap, cfg.min_gap + 1)))
        end = start + length - 1
        lo = max(0, start - cfg.min_gap)
        hi = min(n, end + cfg.min_gap)
        if occupied[lo:hi].any():
            continue
        occupied[start : end + 1] = True

        f0 = int(rng.choice(cand))
        feats = cluster_of.get(f0, [f0]) if rng.random() < cfg.cluster_prob else [f0]
        kind = str(rng.choice(cfg.kinds, p=weights))
        mag = float(rng.uniform(*cfg.magnitude))

        transient = bool(rng.random() < cfg.transient_prob) and length > 400
        before = Xa[start : end + 1, feats].copy()
        if transient:
            # chỉ phá quan hệ ở quãng đầu và quãng cuối
            for lo_, hi_ in ((start, start + int(rng.integers(*cfg.transient_len))),
                             (end - int(rng.integers(*cfg.transient_len)), end + 1)):
                _apply(Xa, slice(max(lo_, 0), min(hi_, len(Xa))), feats, kind, mag, std, rng)
        else:
            _apply(Xa, slice(start, end + 1), feats, kind, mag, std, rng)
        # Nếu phép tiêm gần như không làm dữ liệu thay đổi (ví dụ "đóng băng" một
        # tín hiệu vốn đã hằng số) thì đó là tấn công KHÔNG THỂ phát hiện; giữ lại
        # chỉ làm nhiễu việc chọn ngưỡng -> hoàn tác và bốc lại.
        if float(np.abs(Xa[start : end + 1, feats] - before).max()) < cfg.min_effect:
            Xa[start : end + 1, feats] = before
            occupied[start : end + 1] = False
            continue

        labels[start : end + 1] = 1
        attacks.append(SynthAttack(start=start, end=end, features=list(feats), kind=kind,
                                   magnitude=mag, transient=transient))

    order = np.argsort([a.start for a in attacks])
    attacks = [attacks[i] for i in order]
    return Xa, labels, attacks


def describe(attacks: Sequence[SynthAttack]) -> str:
    lens = np.array([a.end - a.start + 1 for a in attacks])
    kinds: dict[str, int] = {}
    for a in attacks:
        kinds[a.kind] = kinds.get(a.kind, 0) + 1
    n_tr = sum(1 for a in attacks if a.transient)
    return (
        f"{len(attacks)} tấn công giả lập ({n_tr} kiểu quá độ hai đầu) | độ dài: min "
        f"{lens.min()}s, trung vị {int(np.median(lens))}s, max {lens.max()}s | kiểu: "
        + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
    )
