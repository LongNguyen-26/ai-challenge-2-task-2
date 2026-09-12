"""Tiện ích dùng chung cho các script."""
from __future__ import annotations

import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path


def stdout_utf8() -> None:
    """Bật UTF-8 cho stdout/stderr (console Windows mặc định không phải UTF-8)."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
        except Exception:
            pass


@contextmanager
def timed(label: str):
    t0 = time.perf_counter()
    print(f"[{label}] ...", flush=True)
    yield
    print(f"[{label}] xong sau {time.perf_counter() - t0:.1f}s", flush=True)


def save_json(obj, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def load_json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def torch_device(prefer: str = "auto"):
    """Chọn thiết bị cho PyTorch và in thông tin (VRAM) để biết máy có đủ không."""
    import torch

    if prefer not in ("auto", "cuda", "cpu"):
        raise ValueError(prefer)
    if prefer == "cpu" or (prefer == "auto" and not torch.cuda.is_available()):
        if prefer == "auto" and not torch.cuda.is_available():
            print("GPU: không khả dụng (torch không thấy CUDA) -> dùng CPU")
        return torch.device("cpu")
    if not torch.cuda.is_available():
        raise RuntimeError("Yêu cầu CUDA nhưng torch không thấy GPU nào")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name}, VRAM {props.total_memory / 1024**3:.1f} GB, CUDA {torch.version.cuda}")
    return torch.device("cuda")
