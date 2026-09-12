"""Nạp dữ liệu telemetry ICS (định dạng HAI) và chuẩn hoá.

Dữ liệu gốc nằm trong `release/`:
    training.zip     -> train1.csv .. train6.csv  (timestamp, 86 tín hiệu, Attack=0)
    public_test.zip  -> test.csv                  (row_id, timestamp, 86 tín hiệu)

Mọi thứ đọc thẳng từ .zip (các zip này không nén nên đọc rất nhanh, ~6s cho
toàn bộ train) nên không cần giải nén ra đĩa.
"""
from __future__ import annotations

import io
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import polars as pl

META_COLS = ("timestamp", "row_id", "Attack")
TRAIN_NAMES = tuple(f"train{i}.csv" for i in range(1, 7))


# --------------------------------------------------------------------------- #
# đọc csv
# --------------------------------------------------------------------------- #
def _read_csv_bytes(raw: bytes) -> pl.DataFrame:
    """Đọc CSV từ bytes; ép mọi cột tín hiệu về Float64 (tránh polars suy luận i64)."""
    header = raw[: raw.index(b"\n")].decode("utf-8-sig").strip().split(",")
    overrides = {c: pl.Float64 for c in header if c not in META_COLS}
    return pl.read_csv(io.BytesIO(raw), try_parse_dates=True, schema_overrides=overrides)


@dataclass
class Segment:
    """Một đoạn telemetry liên tục (tương ứng 1 tệp csv)."""

    name: str
    values: np.ndarray              # (n, n_cols) float32 theo thứ tự `columns`
    timestamps: np.ndarray          # (n,) datetime64[s]
    row_id: np.ndarray | None = None
    attack: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.values)


def _signal_columns(df: pl.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in META_COLS]


def _to_segment(df: pl.DataFrame, name: str, columns: Sequence[str]) -> Segment:
    ts = df["timestamp"].to_numpy().astype("datetime64[s]")
    vals = df.select(list(columns)).to_numpy().astype(np.float32)
    rid = df["row_id"].to_numpy() if "row_id" in df.columns else None
    atk = df["Attack"].to_numpy().astype(np.int8) if "Attack" in df.columns else None
    return Segment(name=name, values=vals, timestamps=ts, row_id=rid, attack=atk)


def load_csv_set(
    source: str | Path,
    names: Iterable[str] | None = None,
    columns: Sequence[str] | None = None,
) -> tuple[list[Segment], list[str]]:
    """Nạp một tập csv từ file .zip hoặc từ thư mục.

    Args:
        source : đường dẫn .zip hoặc thư mục chứa csv.
        names  : tên các tệp cần đọc (None = tất cả csv trong nguồn).
        columns: thứ tự cột tín hiệu (None = lấy theo tệp đầu tiên).

    Returns:
        (danh sách Segment, danh sách tên cột tín hiệu)
    """
    source = Path(source)
    segments: list[Segment] = []
    cols = list(columns) if columns is not None else None

    if source.is_dir():
        files = sorted(p.name for p in source.glob("*.csv"))
        if names is not None:
            wanted = set(names)
            files = [n for n in files if n in wanted]
        for fname in files:
            df = _read_csv_bytes((source / fname).read_bytes())
            cols = cols or _signal_columns(df)
            segments.append(_to_segment(df, fname, cols))
    else:
        with zipfile.ZipFile(source) as zf:
            files = sorted(n for n in zf.namelist() if n.lower().endswith(".csv"))
            if names is not None:
                wanted = set(names)
                files = [n for n in files if Path(n).name in wanted]
            for fname in files:
                df = _read_csv_bytes(zf.read(fname))
                cols = cols or _signal_columns(df)
                segments.append(_to_segment(df, Path(fname).name, cols))

    if not segments:
        raise FileNotFoundError(f"Không tìm thấy csv nào trong {source}")
    return segments, cols  # type: ignore[return-value]


def find_source(data_dir: str | Path, kind: str) -> Path:
    """Tìm nguồn dữ liệu `kind` ('training' | 'public_test' | 'private_test').

    Ưu tiên thư mục đã giải nén, sau đó tới file zip (kể cả nằm trong thư mục con).
    """
    data_dir = Path(data_dir)
    cand_dir = data_dir / kind
    if cand_dir.is_dir() and any(cand_dir.glob("*.csv")):
        return cand_dir
    cand_zip = data_dir / f"{kind}.zip"
    if cand_zip.is_file():
        return cand_zip
    zips = sorted(data_dir.rglob(f"{kind}.zip"))
    if zips:
        return zips[0]
    dirs = [p for p in sorted(data_dir.rglob(kind)) if p.is_dir() and any(p.glob("*.csv"))]
    if dirs:
        return dirs[0]
    raise FileNotFoundError(f"Không tìm thấy {kind}.zip hoặc thư mục {kind}/ trong {data_dir}")


def load_training(data_dir: str | Path, names: Iterable[str] | None = None):
    """Nạp tập huấn luyện (toàn bộ hoặc một số tệp)."""
    return load_csv_set(find_source(data_dir, "training"), names=names)


def load_test(data_dir: str | Path, kind: str = "public_test", columns=None) -> tuple[Segment, list[str]]:
    """Nạp một tập kiểm tra; trả về (Segment, columns).

    Đề bài nói mỗi gói kiểm tra chỉ có test.csv. Nếu vì lý do nào đó gói chứa
    nhiều tệp, chúng được nối lại theo thứ tự tên tệp (và cảnh báo ra màn hình)
    để pipeline vẫn chạy được.
    """
    segs, cols = load_csv_set(find_source(data_dir, kind), columns=columns)
    if len(segs) == 1:
        return segs[0], cols

    print(f"(!) {kind} có {len(segs)} tệp csv ({[s.name for s in segs]}), đang nối lại theo thứ tự tên")
    merged = Segment(
        name="+".join(s.name for s in segs),
        values=np.concatenate([s.values for s in segs]),
        timestamps=np.concatenate([s.timestamps for s in segs]),
        row_id=(np.concatenate([s.row_id for s in segs]) if all(s.row_id is not None for s in segs) else None),
        attack=(np.concatenate([s.attack for s in segs]) if all(s.attack is not None for s in segs) else None),
    )
    return merged, cols


# --------------------------------------------------------------------------- #
# chuẩn hoá
# --------------------------------------------------------------------------- #
@dataclass
class Scaler:
    """Min-max scaler + loại bỏ cột hằng số (thống kê lấy từ tập train).

    Giá trị được đưa về [0, 1] theo min/max của train rồi cắt vào
    [clip_lo, clip_hi]: điểm nằm ngoài dải train vẫn "vượt ngưỡng" nhưng không
    tạo ra giá trị cực lớn phá huỷ mô hình.
    """

    columns: list[str]      # toàn bộ cột tín hiệu gốc
    keep: list[int]         # chỉ số các cột không phải hằng số
    vmin: list[float]
    vmax: list[float]
    clip_lo: float = -0.5
    clip_hi: float = 1.5

    @property
    def kept_columns(self) -> list[str]:
        return [self.columns[i] for i in self.keep]

    @classmethod
    def fit(
        cls,
        segments: Sequence[Segment],
        columns: Sequence[str],
        eps: float = 1e-8,
        clip_lo: float = -0.5,
        clip_hi: float = 1.5,
    ) -> "Scaler":
        mins = np.full(len(columns), np.inf, dtype=np.float64)
        maxs = np.full(len(columns), -np.inf, dtype=np.float64)
        for seg in segments:
            mins = np.minimum(mins, seg.values.min(axis=0))
            maxs = np.maximum(maxs, seg.values.max(axis=0))
        keep = np.where(maxs - mins > eps)[0]
        return cls(
            columns=list(columns),
            keep=[int(i) for i in keep],
            vmin=[float(x) for x in mins[keep]],
            vmax=[float(x) for x in maxs[keep]],
            clip_lo=clip_lo,
            clip_hi=clip_hi,
        )

    def transform(self, values: np.ndarray) -> np.ndarray:
        """(n, n_cols_gốc) -> (n, n_kept) float32 đã chuẩn hoá."""
        vmin = np.asarray(self.vmin, dtype=np.float32)
        rng = np.asarray(self.vmax, dtype=np.float32) - vmin
        out = (values[:, self.keep] - vmin) / rng
        n_bad = int((~np.isfinite(out)).sum())
        if n_bad:
            print(f"(!) {n_bad} giá trị thiếu/không hữu hạn -> thay bằng giữa dải train")
            out = np.nan_to_num(out, nan=0.5, posinf=self.clip_hi, neginf=self.clip_lo)
        np.clip(out, self.clip_lo, self.clip_hi, out=out)
        return out.astype(np.float32, copy=False)

    def transform_segments(self, segments: Sequence[Segment]) -> list[np.ndarray]:
        return [self.transform(s.values) for s in segments]

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Scaler":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


# --------------------------------------------------------------------------- #
# gom nhóm tín hiệu gần trùng nhau
# --------------------------------------------------------------------------- #
def correlation_clusters(X: np.ndarray, threshold: float = 0.995) -> list[list[int]]:
    """Gom các cột có |tương quan| >= threshold vào cùng cụm (union-find).

    Dùng cho hai việc:
      1. mô hình quan hệ không được phép nhìn "bản sao" của biến đích;
      2. khi che (mask) đầu vào cho TCN thì che nguyên cụm.
    """
    n_cols = X.shape[1]
    Xc = X.astype(np.float32) - X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = Xc.std(axis=0)
    std[std == 0] = 1.0
    Xc = Xc / std
    corr = (Xc.T @ Xc) / len(Xc)

    parent = list(range(n_cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    idx_i, idx_j = np.where(np.abs(corr) >= threshold)
    for i, j in zip(idx_i, idx_j):
        if i < j:
            ri, rj = find(int(i)), find(int(j))
            if ri != rj:
                parent[rj] = ri

    groups: dict[int, list[int]] = {}
    for i in range(n_cols):
        groups.setdefault(find(i), []).append(i)
    return [sorted(v) for v in groups.values()]


def cluster_of(clusters: Sequence[Sequence[int]], n_cols: int) -> np.ndarray:
    """Mảng (n_cols,) -> chỉ số cụm của từng cột."""
    out = np.zeros(n_cols, dtype=np.int32)
    for ci, members in enumerate(clusters):
        for m in members:
            out[m] = ci
    return out
