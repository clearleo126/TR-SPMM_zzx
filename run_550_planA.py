#!/usr/bin/env python3
"""在指定 GPU 上批量跑 TR-source 主代码的 550 矩阵验证 (方案A 全量回归).

与旧版 run_tr_3090_spmm.py 同口径: npz 边列表输入, warmup=50,
epoches 默认 50, 输出 dataset,rows,cols,nnz,tr_kernel_ms,tr_gflops.

用法:
    CUDA_VISIBLE_DEVICES=2 python run_550_planA.py --dimN 128 --epoches 50
"""
import argparse
import csv
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "TR-source"  # 主代码; 消融实验可用 --src TR-abl-source 覆盖
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SRC / "SpMM"))

DATA_DIR = Path("/data1/zhangzhixuan_mp/tr_datasets/dgl_dataset/sp_matrix")
OUT_DEFAULT = SRC / "result" / "tr_planA_n128.csv"


def load_edge_list_npz(filepath):
    import numpy as np
    import scipy.sparse as sp
    raw = np.load(filepath)
    src = raw['src_li'].astype(np.int64)
    dst = raw['dst_li'].astype(np.int64)
    ns = int(raw['num_nodes_src'])
    nd = int(raw['num_nodes_dst'])
    mat = sp.coo_matrix((np.ones(len(src), dtype=np.float32), (src, dst)), shape=(ns, nd))
    mat = mat.tocsr()
    mat.sort_indices()
    return mat


DT = 16   # TC 路由阈值默认 16 (16x16 稠密块, 与 SPTC 块形状统一)
TM = 0    # 生成 TC 块所需的最小残余列数 (0 = 取 DT, 即原行为)


def run_one(name, dimN, epoches, src=None):
    import torch  # noqa: F401  先加载 libc10.so
    from spmm import matrix_to_csr_arrays, run_gpu_spmm
    import matching_utils

    filepath = DATA_DIR / f"{name}.npz"
    mat = load_edge_list_npz(str(filepath))
    row_ptr, col_ind, values, rows, cols = matrix_to_csr_arrays(mat)
    nnz = mat.nnz
    rd = matching_utils.match_2to4(
        row_ptr, col_ind, values, window_size=16, dense_threshold=DT, t_max=8,
        tc_min_cols=TM)
    _, elapsed_ms = run_gpu_spmm(rd, rows, cols, dimN, epoches=epoches,
                                 mode=0, ablation=0)
    kernel_ms = float(elapsed_ms)
    gflops = 2 * nnz * dimN / (kernel_ms * 1e6) if kernel_ms > 0 else 0.0
    return kernel_ms, rows, cols, nnz, gflops


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dimN", type=int, default=128)
    ap.add_argument("--epoches", type=int, default=50)
    ap.add_argument("--out", type=str, default=str(OUT_DEFAULT))
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--matrix_file", type=str, default=None)
    ap.add_argument("--one", type=str, default=None)
    ap.add_argument("--dt", type=int, default=16,
                    help="TC 路由阈值 dense_threshold, 默认 16")
    ap.add_argument("--tcmincol", type=int, default=0,
                    help="生成 TC 块所需最小残余列数, 0=取 dt (原行为)")
    ap.add_argument("--src", type=str, default=None,
                    help="源码目录名, 如 TR-source(默认主代码) 或 TR-abl-source(消融)")
    args = ap.parse_args()
    global DT, TM
    DT = args.dt
    TM = args.tcmincol

    if args.src:
        global SRC
        SRC = ROOT / args.src
        sys.path.insert(0, str(SRC))
        sys.path.insert(0, str(SRC / "SpMM"))
        import importlib
        for m in ("spmm", "matching_utils"):
            sys.modules.pop(m, None)

    if args.one:
        ms, rows, cols, nnz, gf = run_one(args.one, args.dimN, args.epoches)
        print(f"RESULT,{args.one},{rows},{cols},{nnz},{ms:.5f},{gf:.4f}")
        return

    if not DATA_DIR.is_dir():
        raise SystemExit(f"缺少数据目录 {DATA_DIR}")

    if args.matrix_file:
        names = [l.strip() for l in open(args.matrix_file)
                 if l.strip() and not l.startswith('#')]
    else:
        names = sorted(p.stem for p in DATA_DIR.glob("*.npz"))

    out = Path(args.out)
    done = set()
    if out.exists():
        with open(out) as r:
            for row in csv.reader(r):
                if row and row[0] and row[0] != "dataset":
                    done.add(row[0])
    f = open(out, "a", newline="")
    w = csv.writer(f)
    if not done and out.stat().st_size == 0:
        w.writerow(["dataset", "rows", "cols", "nnz", "tr_kernel_ms", "tr_gflops"])
        f.flush()

    todo = [n for n in names if n not in done]
    print(f"共 {len(names)} 个矩阵, 已完成 {len(done)}, 待跑 {len(todo)}", file=sys.stderr)
    t0 = time.time()
    for i, name in enumerate(todo, 1):
        st = time.time()
        cmd = [sys.executable, os.path.abspath(__file__), "--one", name,
               "--dimN", str(args.dimN), "--epoches", str(args.epoches)]
        cmd += ["--dt", str(args.dt), "--tcmincol", str(args.tcmincol)]
        if args.src:
            cmd += ["--src", args.src]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=(args.timeout or None))
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT,")), None)
            if line is None:
                tail = "\n".join((p.stderr or p.stdout).splitlines()[-5:])
                raise RuntimeError(f"无 RESULT 行 (exit={p.returncode})\n{tail}")
            _, _, rows, cols, nnz, ms, gf = line.split(",")
            mode = next((l for l in p.stdout.splitlines() if "路由模式" in l), "")
            tag = "串行" if "串行" in mode else ("并发" if mode else "?")
            print(f"[{i}/{len(todo)}] {name} {float(ms):.4f}ms [{tag}] "
                  f"({time.time()-st:.1f}s)", file=sys.stderr)
            w.writerow([name, rows, cols, nnz, ms, gf])
        except subprocess.TimeoutExpired:
            print(f"[{i}/{len(todo)}] {name} 超时 >{args.timeout}s", file=sys.stderr)
            w.writerow([name, -1, -1, -1, "NA", "NA"])
        except Exception as e:
            print(f"[{i}/{len(todo)}] {name} 失败: {e}", file=sys.stderr)
            w.writerow([name, -1, -1, -1, "NA", "NA"])
        f.flush()
    f.close()
    print(f"完成, 输出 {out}, 总耗时 {time.time()-t0:.1f}s", file=sys.stderr)


if __name__ == "__main__":
    main()
