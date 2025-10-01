#!/usr/bin/env python3
import sys
from pathlib import Path
import numpy as np

folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".").expanduser().resolve()
for p in sorted(folder.glob("*.npy")):  # use rglob("*.npy") for subfolders
    try:
        arr = np.load(p, mmap_mode="r", allow_pickle=False)
        if not np.issubdtype(arr.dtype, np.number):
            print(f"{p.name}: non-numeric dtype {arr.dtype}")
            continue
        mn, mx = float(arr.min()), float(arr.max())
        print(f"{p.name}: {mn:.6g}, {mx:.6g}")
    except Exception as e:
        print(f"{p.name}: ERROR {e}")
