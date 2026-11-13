#!/usr/bin/env python3
"""官方 Libra (ParCIS/Libra) SpMM 正确性 + 计时对拍.

关键: 官方 SpMM 的规定配置是 window=8, wide=8 (见
eva100/kernel/spmm/spmm_h100_fp16_test_args.py), partsize_t=32,
partsize_c=32, shortsize=3, density=8. 用 window=16 会得到错误结果,
因为 TC 路径的 8 行窗口编码与 16 行输入不匹配.

用法: CUDA_VISIBLE_DEVICES=2 python test_libra_official.py ddi [N] [epoches]
"""
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch

ROOT = Path("/home/zhangzhixuan/DR_SPMM_ABL")
LIBRA = ROOT / "baseline" / "libra" / "Libra-source"
sys.path.insert(0, str(LIBRA))
sys.path.insert(0, str(LIBRA / "SpMM"))

import Libra5Block  # noqa: E402
import Libra6SpMM  # noqa: E402

DATA = Path("/data1/zhangzhixuan_mp/tr_datasets/dgl_dataset/sp_matrix")

WINDOW, WIDE = 8, 8        # 官方 SpMM 规定值
PART_T, PART_C, SHORT = 32, 32, 3
DENSITY = 8


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "ddi"
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    epoches = int(sys.argv[3]) if len(sys.argv) > 3 else 50

    raw = np.load(DATA / f"{name}.npz")
    rows, cols = int(raw["num_nodes_src"]), int(raw["num_nodes_dst"])
    M = rows + (WINDOW - rows % WINDOW) % WINDOW
    A = sp.coo_matrix((np.ones(len(raw["src_li"])),
                       (raw["src_li"], raw["dst_li"])), shape=(M, cols)).tocsr()
    A.sort_indices()

    rp = torch.IntTensor(A.indptr)
    ci = torch.IntTensor(A.indices)
    deg = torch.ones(A.nnz, dtype=torch.float32)

    t0 = time.time()
    out = Libra5Block.block_tf32_tcu_cuda_binary(
        rp, ci, deg, PART_T, PART_C, SHORT, DENSITY, WINDOW, WIDE)
    prep_ms = (time.time() - t0) * 1000.0
    (t_ro, t_blk, t_col, t_val, t_wrow, t_atomic, t_binary,
     c_ro, c_row, c_atomic, c_col, c_val,
     c_ro_s, c_row_s, c_atomic_s, c_col_s, c_val_s, _dur) = out

    dev = "cuda"
    x = torch.randn(cols, N, dtype=torch.float16, device=dev)
    args = [t_ro.to(dev), t_blk.to(dev), t_col.to(dev), t_val.half().to(dev),
            t_wrow.to(dev), t_atomic.to(dev), t_binary.to(dev),
            c_ro.to(dev), c_row.to(dev), c_atomic.to(dev), c_col.to(dev),
            c_val.half().to(dev),
            c_ro_s.to(dev), c_row_s.to(dev), c_atomic_s.to(dev), c_col_s.to(dev),
            c_val_s.half().to(dev)]
    Y, ms = Libra6SpMM.forward_fp16_tcu_cuda(
        *args, x, t_atomic.shape[0], c_atomic.shape[0], PART_C,
        c_atomic_s.shape[0], N, M, cols, epoches)
    torch.cuda.synchronize()

    o = Y[:rows].float().cpu()
    ref = torch.tensor(A[:rows] @ x.float().cpu().numpy())
    e = (o - ref).abs()
    rel = float(e.max()) / (float(ref.abs().max()) + 1e-8)
    print(f"{name} N={N}: prep={prep_ms:.1f}ms kernel={float(ms):.4f}ms "
          f"max_abs={float(e.max()):.3e} rel={rel:.3e}")
    print(f"  路由 nnz: TC={t_val.numel()} long={c_val.numel()} "
          f"short={c_val_s.numel()} sum={t_val.numel() + c_val.numel() + c_val_s.numel()}"
          f" / {A.nnz}")


if __name__ == "__main__":
    main()
