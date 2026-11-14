#!/usr/bin/env python3
"""把 DTC 官方 dtc_datasets 的 origin npz 转成 $DATA/<name>/<name>.mtx (pattern)."""
import os
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite

SRC = "/dev/shm/dtc/origin"
OUT = os.environ.get("DR_REAL_DIR", "/dev/shm/real_matrices")
NAMES = ["ddi", "DD", "web-BerkStan", "Yeast", "OVCAR-8H", "YeastH", "protein", "reddit"]
for name in NAMES:
    d = np.load(os.path.join(SRC, name + ".npz"))
    n = int(d["num_nodes"])
    src = d["src_li"]
    dst = d["dst_li"]
    A = sp.coo_matrix((np.ones(len(src), np.int32), (src, dst)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    A.sort_indices()
    p = os.path.join(OUT, name, name + ".mtx")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    mmwrite(p, sp.coo_matrix(A), field="integer")
    print(f"{name}: shape={A.shape} nnz={A.nnz} -> {p} ({os.path.getsize(p)/1e6:.0f} MB)",
          flush=True)
