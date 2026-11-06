#!/usr/bin/env python3
"""End-to-end GCN forward pass (2-layer, fwd only).

    e2e_s = prep_s + wall(cold call) + wall(hot call)      # 不含文件 I/O

矩阵文件读取 (load_s) 对四家 baseline 完全相同, 且是纯 I/O 噪声源
(实测 protein 两次 5.98s / 4.09s, 46% 波动), 因此不计入 e2e_s;
`e2e_incl_load_s` 列保留含 I/O 的值以便对照.

The two SpMM calls are timed separately and both enter e2e_s: layer 1 pays any
one-off host work (metadata compression, cudaMalloc, first H2D), layer 2 runs
from the cached prepared state.  Hiding the cold call outside the timer would
credit each method differently, so it is counted.

Columns per (matrix, method):
    cold_ms         layer 1 wall clock (includes one-off host work)
    hot_ms          layer 2 wall clock (steady state)
    chain_2call_ms  cold_ms + hot_ms, the quantity added to load+prep
    kernel_ms       kernel time reported by the extension itself, per call

Usage:
    CUDA_VISIBLE_DEVICES=2 python gnn_e2e.py --matrices ddi Yeast --dimN 128
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch

from gnn_train import (BACKENDS, DIM_N, HEADER as _TRAIN_HEADER, REAL_DIR,
                       RESULTS_DIR, discover, load_mtx)


def _reset_tr_cache():
    """Drop TR's device-side prep cache between matrices (avoid OOM)."""
    try:
        import TR_SPMM as _T
        if hasattr(_T, "clear_prep_cache"):
            _T.clear_prep_cache()
    except Exception:
        pass


def _reset_mp_cache():
    try:
        import MP_SPMM_WRAP
        if hasattr(MP_SPMM_WRAP, "clear_prep"):
            MP_SPMM_WRAP.clear_prep()
    except Exception:
        pass


def _reset_libra_cache():
    try:
        from tr_csr_fp16 import test_tr_csr_online  # noqa: F401  (确保 .so 已加载)
        from tr_csr_fp16 import __path__ as _p  # noqa: F401
        import Libra6SpMMOnline
        if hasattr(Libra6SpMMOnline, "clear_prep_cache"):
            Libra6SpMMOnline.clear_prep_cache()
    except Exception:
        pass


def _kernel_time(method, backend):
    """Kernel time reported by the extension after the last call (ms/call)."""
    if method in ("tr", "libra", "dtc"):
        return getattr(backend, "_last_kernel_ms", float("nan"))
    return float("nan")


def run_one(name, A, load_s, method, dimN, device):
    backend = BACKENDS[method]()
    if hasattr(backend, "set_dataset"):
        backend.set_dataset(name)
    # forward only: A^T is never used, so preprocess A alone (legacy prep_s)
    backend.preprocess(A, device, assume_undirected=True)

    n, k = A.shape
    X = torch.randn(k, dimN, device=device, dtype=torch.float32)

    # 真实两层 GCN 前向: 第 1 层付一次性 host 开销 (元数据压缩 / H2D /
    # cudaMalloc 等), 第 2 层命中缓存只付稳态成本. 两者都必须计入 e2e,
    # 否则等于把各家的冷启动成本藏到计时窗口外.
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    Y = backend.spmm_A(X)
    torch.cuda.synchronize()
    first_ms = (time.perf_counter() - t0) * 1e3

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    backend.spmm_A(Y)
    torch.cuda.synchronize()
    second_ms = (time.perf_counter() - t0) * 1e3

    chain_ms = first_ms + second_ms
    e2e_s = backend.prep_A_s + chain_ms / 1e3          # 不含 load
    e2e_load_s = load_s + e2e_s                        # 含 load, 仅作对照

    return {
        "dataset": name, "rows": n, "cols": k, "nnz": int(A.nnz),
        "method": method, "dimN": dimN, "load_s": f"{load_s:.4f}",
        "prep_s": f"{backend.prep_A_s:.4f}",
        "cold_ms": f"{first_ms:.4f}",
        "hot_ms": f"{second_ms:.4f}",
        "chain_2call_ms": f"{chain_ms:.4f}",
        "kernel_ms": f"{_kernel_time(method, backend):.4f}",
        "e2e_s": f"{e2e_s:.4f}",
        "e2e_incl_load_s": f"{e2e_load_s:.4f}",
    }


HEADER = ["dataset", "rows", "cols", "nnz", "method", "dimN", "load_s",
          "prep_s", "cold_ms", "hot_ms", "chain_2call_ms", "kernel_ms",
          "e2e_s", "e2e_incl_load_s"]


def _median_rec(recs):
    """按 e2e_s 取中位数那一轮 (reps>1 时抗噪, 不用 min)."""
    ordered = sorted(recs, key=lambda r: float(r["e2e_s"]))
    return ordered[len(ordered) // 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=None)
    ap.add_argument("--methods", nargs="+", default=["tr", "mp", "libra"],
                    choices=["tr", "mp", "libra", "dtc"])
    ap.add_argument("--dimN", nargs="+", type=int, default=[DIM_N])
    ap.add_argument("--tag", type=str, default="128")
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--reps", type=int, default=1,
                    help="每 (矩阵, 方法, N) 重复次数, 取 e2e 中位数那一轮")
    ap.add_argument("--append", action="store_true",
                    help="追加模式: 逐行即时落盘 (分矩阵调用/长跑防中断丢数据)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    device = torch.device("cuda")

    out_path = args.out or str(RESULTS_DIR / f"gnn_e2e_new_{args.tag}.csv")
    targets = discover(args.matrices)
    print(f"{len(targets)} matrices, methods={args.methods}, "
          f"dimN={args.dimN}, reps={args.reps}", file=sys.stderr)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_p = Path(out_path)
    fresh = not (out_p.exists() and out_p.stat().st_size > 0) or not args.append
    f = open(out_path, "a" if args.append and not fresh else "w", newline="")
    w = csv.DictWriter(f, fieldnames=HEADER)
    if fresh:
        w.writeheader()
        f.flush()

    n_rows = 0
    for name, path in targets:
        A, load_s = load_mtx(path)
        print(f"=== {name} {A.shape} nnz={A.nnz} load={load_s:.2f}s ===",
              file=sys.stderr)
        # rep 在外层: 同一 rep 内 N 与方法相邻执行, 消除机器状态漂移
        acc = {(d, m): [] for d in args.dimN for m in args.methods}
        failed = set()
        for _rep in range(max(1, args.reps)):
            for dimN in args.dimN:
                for method in args.methods:
                    key = (dimN, method)
                    if key in failed:
                        continue
                    _reset_tr_cache()
                    _reset_mp_cache()
                    _reset_libra_cache()
                    gc.collect()
                    torch.cuda.empty_cache()
                    try:
                        rec = run_one(name, A, load_s, method, dimN, device)
                    except Exception as exc:  # missing MP bins / OOM
                        print(f"  [SKIP] {name} {method} N={dimN}: {exc}",
                              file=sys.stderr)
                        failed.add(key)
                        continue
                    acc[key].append(rec)
        for dimN in args.dimN:
            for method in args.methods:
                lst = acc[(dimN, method)]
                if not lst:
                    continue
                best = _median_rec(lst)
                n_rows += 1
                w.writerow(best)
                f.flush()      # 即时落盘, 进程被杀也不丢已完成行
                print(f"  [OK] {name:12s} {method:6s} N={dimN:<3d} "
                      f"prep={best['prep_s']:>8s}s cold={best['cold_ms']:>9s}ms "
                      f"hot={best['hot_ms']:>9s}ms "
                      f"kernel={best['kernel_ms']:>8s}ms "
                      f"e2e={best['e2e_s']:>9s}s "
                      f"(n={len(lst)})", file=sys.stderr)
        del A
        gc.collect()
        torch.cuda.empty_cache()

    f.close()
    print(f"[done] {n_rows} rows -> {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
