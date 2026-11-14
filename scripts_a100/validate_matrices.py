#!/usr/bin/env python3
"""严格校验 11 个矩阵的 shape/nnz/值全 1, 并打印结果."""
import os
import sys
from scipy.io import mmread

EXPECT = {
    "ddi": (4267, 4267, 2140089),
    "DD": (334925, 334925, 1686092),
    "ogbn_arxiv": (169343, 169343, 1166243),
    "web-BerkStan": (685230, 685230, 7600595),
    "Yeast": (1710902, 1710902, 3636546),
    "OVCAR-8H": (1889542, 1889542, 3946402),
    "igb_small": (1000000, 1000000, 12068130),
    "YeastH": (3138114, 3138114, 6487230),
    "protein": (132534, 132534, 79255038),
    "ogbn_products": (2449029, 2449029, 123718152),
    "reddit": (232965, 232965, 114848857),
}
DATA = os.environ.get("DR_REAL_DIR", "/dev/shm/real_matrices")
ok = True
for name, exp in EXPECT.items():
    p = os.path.join(DATA, name, name + ".mtx")
    if not os.path.exists(p):
        print(f"MISSING {name}: {p}")
        ok = False
        continue
    A = mmread(p).tocsr()
    A.sum_duplicates()
    got = (A.shape[0], A.shape[1], A.nnz)
    ones = abs(A.data - 1).max() if A.nnz else 0
    good = (got == exp) and ones == 0
    print(f"{'OK  ' if good else 'FAIL'} {name:14s} got={got} exp={exp} max|v-1|={ones}")
    ok = ok and good
print("ALL_OK" if ok else "VALIDATION_FAILED")
sys.exit(0 if ok else 1)
