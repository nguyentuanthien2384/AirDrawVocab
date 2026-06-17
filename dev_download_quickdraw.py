"""
Tải dữ liệu QuickDraw numpy_bitmap cho 19 lớp vào data/npy_28/.
Tải từng phần (HTTP Range) để nhanh, chỉ lấy số mẫu cần.

    .\.venv311\Scripts\python.exe dev_download_quickdraw.py --per-class 8000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CATEGORIES, DATA_DIR

BASE = "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{}.npy"


def download_partial(label: str, n: int) -> np.ndarray:
    url = BASE.format(label)
    nbytes = 256 + n * 784
    r = requests.get(url, headers={"Range": f"bytes=0-{nbytes}"}, timeout=180)
    r.raise_for_status()
    raw = r.content
    assert raw[:6] == b"\x93NUMPY", f"{label}: not npy"
    major = raw[6]
    if major == 1:
        hlen = int.from_bytes(raw[8:10], "little")
        data_off = 10 + hlen
    else:
        hlen = int.from_bytes(raw[8:12], "little")
        data_off = 12 + hlen
    avail = (len(raw) - data_off) // 784
    arr = np.frombuffer(raw[data_off:data_off + avail * 784], dtype=np.uint8).reshape(avail, 784)
    return arr


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=8000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for i, label in enumerate(CATEGORIES, 1):
        out = DATA_DIR / f"{label}.npy"
        if out.exists() and not args.force:
            existing = np.load(out)
            if len(existing) >= args.per_class:
                print(f"[{i:2d}/19] {label:10s} da co {len(existing)} mau, bo qua")
                continue
        arr = download_partial(label, args.per_class)
        np.save(out, arr)
        print(f"[{i:2d}/19] {label:10s} -> {len(arr)} mau, {out.stat().st_size//1024} KB")
    print("Xong tai du lieu vao", DATA_DIR)


if __name__ == "__main__":
    main()
