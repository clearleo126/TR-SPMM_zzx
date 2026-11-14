#!/usr/bin/env python3
"""IGB-small -> $DATA/igb_small/igb_small.mtx (integer 字段, 值全 1).

IGB 官方下载地址 (igb/download.py 中的 homogeneous/small):
  https://igb-public-awsopen.s3.amazonaws.com/igb-homogeneous/igb_homogeneous_small.tar.gz
只需 small/processed/paper__cites__paper/edge_index.npy, 无需 1M x 1024 特征.

用法:
  curl -L -o /dev/shm/igb/igb_small.tar.gz <url>
  tar xzf /dev/shm/igb/igb_small.tar.gz -C /dev/shm/igb \
      --wildcards 'small/processed/paper__cites__paper/edge_index.npy'
  python prepare_igb_small.py
"""
import os
import numpy as np
import scipy.sparse as sp
from scipy.io import mmwrite

EDGE = "/dev/shm/igb/small/processed/paper__cites__paper/edge_index.npy"
OUT = os.environ.get("DR_REAL_DIR", "/dev/shm/real_matrices")
N = 1000000          # igb-small 节点数

e = np.load(EDGE)                      # (12070502, 2), 含重复
A = sp.coo_matrix((np.ones(len(e), np.int32), (e[:, 0], e[:, 1])),
                  shape=(N, N)).tocsr()
A.sum_duplicates()
A.sort_indices()
A.data = np.ones(A.nnz, np.int32)      # 去重后强制值全 1 (期望 nnz = 12068130)
p = os.path.join(OUT, "igb_small", "igb_small.mtx")
os.makedirs(os.path.dirname(p), exist_ok=True)
mmwrite(p, sp.coo_matrix(A), field="integer")
print(f"igb_small shape={A.shape} nnz={A.nnz} -> {p}")
