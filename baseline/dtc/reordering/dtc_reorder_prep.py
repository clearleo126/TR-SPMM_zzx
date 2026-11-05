#!/usr/bin/env python3
"""
dtc_reorder_prep.py — 为 DTC-SpMM 官方 GPU TCA reorder 准备输入

DTC 官方 reordering/TCA_reorder_local.py 读取 <name>.npz 的 src_li/dst_li/num_nodes,
但 TR 的 sp_matrix/*.npz 是 num_nodes_src/num_nodes_dst/num_edges 格式。
本脚本把 TR npz 转换/软链成官方脚本可读的 g_dataset/<name>.npz (补 num_nodes 字段),
然后调用官方 GPU TCA reorder 脚本生成 asplos_dtc_dataset/<name>.reorder.npz.

用法: python dtc_reorder_prep.py <name> [gpu=3]
产出: baseline/dtc/reordering/asplos_dtc_dataset/<name>.reorder.npz
      (keys: src_li, dst_li, num_nodes — 已按重排映射的行列索引)
"""
import os
import subprocess
import sys
import time
from pathlib import Path

REORDER_DIR = Path("/home/zhangzhixuan/TR-SPMM_zzx/baseline/dtc/reordering")
G_DATASET = REORDER_DIR / "g_dataset"
ASPLOS = REORDER_DIR / "asplos_dtc_dataset"
SRC_NPZ = Path("/data1/zhangzhixuan_mp/tr_datasets/dgl_dataset/sp_matrix")


def prepare(name):
    """把 TR npz 转成官方脚本可读格式 (补 num_nodes). 返回耗时. """
    import numpy as np
    t0 = time.perf_counter()
    raw = np.load(SRC_NPZ / f"{name}.npz")
    src = raw["src_li"]
    dst = raw["dst_li"]
    num_nodes = int(raw["num_nodes_src"])
    out = G_DATASET / f"{name}.npz"
    # 与官方数据集的 'num_nodes' 字段一致 (官方用 num_nodes 作方形矩阵行数)
    np.savez(out, src_li=src, dst_li=dst, num_nodes=num_nodes)
    return time.perf_counter() - t0


def reorder(name, gpu):
    """调官方 GPU TCA reorder (local 路径版). 返回 (耗时, 是否成功). """
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["LD_LIBRARY_PATH"] = (
        "/home/zhangzhixuan/miniconda3/envs/libra/lib:"
        "/home/zhangzhixuan/miniconda3/envs/libra/lib/python3.9/site-packages/torch/lib:"
        + env.get("LD_LIBRARY_PATH", ""))
    env["LD_PRELOAD"] = "/usr/lib/x86_64-linux-gnu/libgomp.so.1"
    py = "/home/zhangzhixuan/miniconda3/envs/libra/bin/python"
    t0 = time.perf_counter()
    r = subprocess.run(
        [py, str(REORDER_DIR / "TCA_reorder_local.py"), "--dataset", name],
        cwd=str(REORDER_DIR), env=env, capture_output=True, text=True,
        timeout=3600)
    ok = (ASPLOS / f"{name}.reorder.npz").exists()
    if not ok:
        print(r.stdout[-2000:], file=sys.stderr)
        print(r.stderr[-2000:], file=sys.stderr)
    return time.perf_counter() - t0, ok


if __name__ == "__main__":
    name = sys.argv[1]
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    if not (ASPLOS / f"{name}.reorder.npz").exists():
        prepare_s = prepare(name)
        reorder_s, ok = reorder(name, gpu)
        print(f"[dtc_reorder] {name}: prep={prepare_s:.2f}s reorder={reorder_s:.2f}s ok={ok}")
    else:
        print(f"[dtc_reorder] {name}: reorder 已存在, 跳过")
