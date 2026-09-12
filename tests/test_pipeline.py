"""Kiểm thử nhanh, không cần dữ liệu thật.

    python -m pytest tests -q          (hoặc)     python tests/test_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from icsad.etapr import evaluate, labels_to_ranges, ranges_to_labels
from icsad.features import iter_design, transform_block
from icsad.postprocess import drop_short, hysteresis, merge_gaps, moving_average, score_to_labels
from icsad.relation import RelationModel
from icsad.scoring import aggregate, boundary_guard, normalize
from icsad.synth import SynthConfig, inject


# --------------------------------------------------------------------------- #
def test_labels_ranges_roundtrip():
    y = np.array([0, 1, 1, 0, 0, 1, 0, 1, 1, 1], dtype=np.int8)
    ranges = labels_to_ranges(y)
    assert ranges == [(1, 2), (5, 5), (7, 9)]
    assert (ranges_to_labels(ranges, len(y)) == y).all()
    assert labels_to_ranges(np.ones(3, dtype=np.int8)) == [(0, 2)]
    assert labels_to_ranges(np.zeros(3, dtype=np.int8)) == []


def test_etapr_known_values():
    """Các giá trị dưới đây đã được đối chiếu với gói eTaPR gốc (theta_p=.5, theta_r=.1)."""
    # dự đoán trùng khít tấn công -> điểm tuyệt đối
    y = ranges_to_labels([(10, 109)], 200)
    assert evaluate(y, y)["f1"] == 1.0

    # dự đoán phủ 20% đoạn tấn công, nằm trọn bên trong -> phát hiện được
    p = ranges_to_labels([(10, 29)], 200)
    r = evaluate(y, p)
    assert r["eTaR"] == 0.6 and r["eTaP"] == 1.0          # (1 + 0.2)/2 ; (1 + 1)/2

    # dự đoán chỉ phủ 5% (< theta_r) -> bị prune, không phát hiện, precision = 0
    p = ranges_to_labels([(10, 14)], 200)
    r = evaluate(y, p)
    assert r["eTaR"] == 0.0 and r["eTaP"] == 0.0

    # đoạn dự đoán dài gấp 3 tấn công (chỉ 1/3 nằm trong) -> < theta_p -> bị loại
    p = ranges_to_labels([(10, 309)], 400)
    r = evaluate(ranges_to_labels([(10, 109)], 400), p)
    assert r["eTaP"] == 0.0 and r["eTaR"] == 0.0

    # một đoạn đúng + một đoạn rác -> precision có trọng số sqrt(độ dài)
    y2 = ranges_to_labels([(0, 99)], 400)
    p2 = ranges_to_labels([(0, 99), (200, 203)], 400)
    r = evaluate(y2, p2)
    w1, w2 = 10.0, 2.0                                    # sqrt(100), sqrt(4)
    assert abs(r["eTaP"] - (w1 * 1.0) / (w1 + w2)) < 1e-12
    assert r["eTaR"] == 1.0


def test_postprocess_ops():
    score = np.array([0, 0, 5, 5, 0, 0, 6, 0, 0, 0], dtype=np.float32)
    lab = hysteresis(score, 4.0, 2.0)
    assert lab.tolist() == [0, 0, 1, 1, 0, 0, 1, 0, 0, 0]
    assert merge_gaps(lab, 2).tolist() == [0, 0, 1, 1, 1, 1, 1, 0, 0, 0]
    assert drop_short(lab, 2).tolist() == [0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    assert np.allclose(moving_average(np.ones(5, dtype=np.float32), 3), 1.0)
    out = score_to_labels(score, 4.0, 2.0, smooth_window=1, min_len=2, max_gap=2)
    assert out.sum() == 5


def test_boundary_guard():
    g = boundary_guard(np.ones(1000, dtype=np.float32), 10, 100)
    assert g[0] == 0 and g[-1] == 0 and g[500] == 1.0
    assert 0 < g[50] < 1


def test_features_and_relation():
    rng = np.random.default_rng(0)
    n = 20000
    t = np.arange(n)
    base = np.sin(t / 50.0).astype(np.float32)
    X = np.stack([base, base * 0.5 + 0.2, rng.normal(0, 0.1, n).astype(np.float32)], axis=1)
    X = ((X - X.min(0)) / (X.max(0) - X.min(0))).astype(np.float32)

    Z = transform_block(X)
    assert Z.shape == (n, X.shape[1] * 5)
    assert np.isfinite(Z).all()

    # chia khối không được làm đổi kết quả: ở giữa chuỗi (nơi vùng đệm đã đủ dài)
    # bản tính theo khối phải trùng với bản tính một lần trên cả chuỗi
    stacked = np.concatenate([z for _, _, z in iter_design([X], chunk=1000, pad=2048)])
    assert stacked.shape == Z.shape
    assert np.abs(stacked[5000:15000] - Z[5000:15000]).max() < 1e-2
    # đệm càng dài thì càng hội tụ về bản tính một lần trên cả chuỗi
    wide = np.concatenate([z for _, _, z in iter_design([X], chunk=1000, pad=6000)])
    assert np.abs(wide[8000:12000] - Z[8000:12000]).max() < 1e-4

    model = RelationModel().fit([X], clusters=[[0], [1], [2]], verbose=False)
    res = model.residuals(X)
    assert res.shape == X.shape
    # cột 1 là hàm tuyến tính của cột 0 -> phải dự đoán gần như hoàn hảo
    assert np.abs(res[:, 1]).mean() < 0.01


def test_synth_injection_changes_data():
    rng = np.random.default_rng(1)
    X = rng.random((20000, 6)).astype(np.float32)
    Xa, labels, attacks = inject(X, [[0], [1], [2], [3], [4], [5]],
                                 SynthConfig(seed=3, rate_per_hour=2.0))
    assert len(attacks) > 0
    assert labels.sum() == sum(a.end - a.start + 1 for a in attacks)
    for a in attacks:                       # mỗi tấn công phải thực sự đổi dữ liệu
        d = np.abs(Xa[a.start : a.end + 1, a.features] - X[a.start : a.end + 1, a.features]).max()
        assert d >= 0.01
    assert np.array_equal(Xa[labels == 0], X[labels == 0])


def test_scoring_shapes():
    rng = np.random.default_rng(0)
    R = rng.normal(0, 0.01, (5000, 7)).astype(np.float32)
    R[1000:1100, 3] += 0.5                  # một đoạn bất thường
    for mode in ("mad", "rank"):
        z = normalize(R, mode=mode)
        assert z.shape == R.shape and (z >= 0).all()
        # sau khi làm trơn, đoạn bất thường phải nổi hẳn lên: nhiễu đơn lẻ bị
        # san phẳng còn lệch kéo dài thì không
        s = moving_average(aggregate(z, topk=1), 31)
        assert 1000 <= int(np.argmax(s)) < 1100
        outside = np.concatenate([s[:950], s[1150:]])
        assert s[1000:1100].mean() > outside.max()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} kiểm thử đều đạt")
