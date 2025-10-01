#!/usr/bin/env python3
from pathlib import Path
import numpy as np

FOLDER = r"./data_patches_H200_W100_strpp2_11days_Aligned"   # <-- put your folder path here

for p in sorted(Path(FOLDER).expanduser().glob("*.npy")):   # use rglob("*.npy") for subfolders
    a = np.load(p, mmap_mode="r", allow_pickle=False)

    print(f"{p.name}: {a.min():.6g}, {a.max():.6g}")