#!/usr/bin/env python3
"""2-layer GCN training loop (fwd + bwd + Adam) for kernel comparison.

ref (torch.sparse.mm) / TR-SPMM / Libra / MP-SpMM share one loop; only the
SpMM backend changes.  Protocol:
    Linear -> SpMM(A) -> ReLU  |  Linear -> SpMM(A)
    backward calls SpMM(A^T) once; A and A^T are preprocessed separately.
Main metric: avg_epoch_ms (post-warmup, includes fwd+bwd+Adam).

Usage:
    CUDA_VISIBLE_DEVICES=2 python gnn_train.py --methods tr libra --matrices ddi
"""
from __future__ import annotations

import argparse
import csv
import gc
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.io import mmread

ROOT = Path(__file__).resolve().parent
# 数据根目录可迁移 (A100/Vast.ai 上用 DR_REAL_DIR 指向矩阵所在处).
# MP 官方预处理产物也会写在该目录的 dataset_mp_processed/ 下.
REAL_DIR = Path(os.environ.get("DR_REAL_DIR", "/data1/datasets/real_matrices"))
TR_MAIN = ROOT / "TR-source"   # 主代码 (DR-SpMM 主实现, 23 参数 + 零拷贝 + 缓存)
BASE_DIR = ROOT / "baseline"
LIBRA_PKG = BASE_DIR / "libra"
MP_PKG = BASE_DIR / "mp"
GNN_DATA_ROOT = ROOT / "data_gnn"
RESULTS_DIR = ROOT / "result"

# TR-source 是主代码; TR-abl-source 仅用于消融实验 (独立进程加载),
# 所有 baseline 源码统一放在 baseline/ 下.
for _p in (str(LIBRA_PKG), str(MP_PKG),
           str(TR_MAIN), str(TR_MAIN / "SpMM")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matching_utils as _MU        # noqa: E402
import TR_SPMM as _TRSPMM           # noqa: E402

DIM_N = 128
WARMUP = 5
EPOCHS = 20
LR = 1e-3
DT = 16        # dense_threshold
TM = 16        # tc_min_cols

# Libra 的 C++ wrapper 自带 benchmark 式自测: 每次调用先跑 10 次内部 warmup,
# 结束后再完整重算一遍. 在训练/端到端循环里这些都属于白跑的重复计算
# (Yeast 上 ~40 ms/次, 而真实 kernel 仅 2 ms), 与"一次 SpMM 调用"的语义不符.
# 这里关闭, 使三家 baseline 的调用边界一致; 需要复现官方 benchmark 行为时可覆盖.
os.environ.setdefault("LIBRA_INNER_WARMUP", "0")
os.environ.setdefault("LIBRA_EXTRA_PASS", "0")

# DTC: 官方 kernel 每次调用内部跑 EXE_TIME=2 遍取事件平均, 且每次在 cwd
# 追加写计时 CSV. 端到端链式调用中一次 SpMM 只应执行一遍 kernel, 与其余
# baseline 的调用边界一致; CSV 写盘移出计时窗口. 复现官方行为可覆盖.
os.environ.setdefault("DTC_EXE_TIME", "1")
os.environ.setdefault("DTC_LOG_CSV", "0")

_FWD = 0
_BWD = 0


def _reset_cnt():
    global _FWD, _BWD
    _FWD = _BWD = 0


class _SpMMFn(torch.autograd.Function):
    """forward = SpMM(A), backward = SpMM(A^T).

    必须定义在模块级: 若每次调用都在函数内新建 Function 子类, 会重复付出
    类创建 + autograd 分派缓存的代价 (实测 ddi 上 3 ms/次 vs 0.42 ms).
    """
    @staticmethod
    def forward(ctx, S, backend):
        global _FWD
        _FWD += 1
        ctx.backend = backend
        with torch.no_grad():
            return backend.spmm_A(S)

    @staticmethod
    def backward(ctx, dY):
        global _BWD
        _BWD += 1
        return ctx.backend.spmm_AT(dY.contiguous()), None


def spmm_op(S, backend):
    return _SpMMFn.apply(S, backend)


class GCN2(nn.Module):
    def __init__(self, dim=DIM_N):
        super().__init__()
        self.lin1 = nn.Linear(dim, dim, bias=True)
        self.lin2 = nn.Linear(dim, dim, bias=True)

    def forward(self, x, backend):
        h = F.relu(spmm_op(self.lin1(x), backend))
        return spmm_op(self.lin2(h), backend)


# =====================================================================
# Backends
# =====================================================================
def _csr_to_torch_sparse(csr, device):
    coo = csr.tocoo()
    i = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long)
    v = torch.tensor(coo.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(i, v, coo.shape).coalesce().to(device)


class RefBackend:
    name = "ref"

    def __init__(self):
        self.A = self.AT = None
        self.prep_A_s = self.prep_AT_s = 0.0

    def preprocess(self, A, device, assume_undirected=False):
        t = time.perf_counter()
        self.A = _csr_to_torch_sparse(A, device)
        self.prep_A_s = time.perf_counter() - t
        t = time.perf_counter()
        if assume_undirected:
            self.AT, self.prep_AT_s = self.A, 0.0
        else:
            self.AT = _csr_to_torch_sparse(A.T.tocsr(), device)
            self.prep_AT_s = time.perf_counter() - t

    def spmm_A(self, X):
        return torch.sparse.mm(self.A, X)

    def spmm_AT(self, dY):
        return torch.sparse.mm(self.AT, dY)


class TRBackend:
    """TR 主代码 (TR-source): match_2to4 once per (A, A^T), then epoches=1 per call."""
    name = "tr"

    def __init__(self):
        self.prep_A_s = self.prep_AT_s = 0.0
        self._pA = self._pAT = None
        self.M = self.K = 0
        self._last_kernel_ms = float("nan")

    @staticmethod
    def _pack(mat):
        rp = mat.indptr.astype(np.int32)
        ci = mat.indices.astype(np.int32)
        vals = mat.data.astype(np.float32)
        packed = _MU.match_2to4(rp, ci, vals, window_size=16,
                                dense_threshold=DT, t_max=8, tc_min_cols=TM)
        # 一次性转到 half: 若放在每次 forward 里, Yeast 上要多付约 25 ms/次
        packed["sptc_value"] = packed["sptc_value"].to(torch.float16)
        packed["tc_value"] = packed["tc_value"].to(torch.float16)
        return packed

    def _forward(self, packed, X, M, K):
        sw = packed["sptc_window_offset"]
        tcw = packed["tc_window_offset"]
        nsb = int(sw[-1].item())
        ntb = int(tcw[-1].item())
        dt = 16
        if ntb > 0:
            dt = int(packed["tc_col_old"].size(0) // ntb)
        out, t_ms = _TRSPMM.forward(
            sw, packed["sptc_value"], packed["sptc_metadata"],
            packed["sptc_col_old"],
            tcw, packed["tc_offset"], packed["tc_local_id"], packed["tc_col_old"],
            packed["tc_value"],
            X.to(torch.float16).contiguous(),
            16, int(X.size(1)), M, K, int(sw.size(0) - 1), nsb, ntb, dt,
            1,    # epoches: one call
            0,    # mode: full
            0,    # warmup
            0,    # ablation
            0,    # chunk_size: auto
        )
        self._last_kernel_ms = float(t_ms.item() if torch.is_tensor(t_ms) else t_ms)
        return out

    def preprocess(self, A, device, assume_undirected=False):
        t = time.perf_counter()
        self._pA = self._pack(A)
        self.prep_A_s = time.perf_counter() - t
        t = time.perf_counter()
        if assume_undirected:
            self._pAT, self.prep_AT_s = self._pA, 0.0
        else:
            self._pAT = self._pack(A.T.tocsr())
            self.prep_AT_s = time.perf_counter() - t
        self.M, self.K = A.shape

    def spmm_A(self, X):
        # GPU 直通: 输入/输出都留在显存, 避免大矩阵的 D2H/H2D 往返
        return self._forward(self._pA, X.detach(), self.M, self.K)

    def spmm_AT(self, dY):
        return self._forward(self._pAT, dY.detach(), self.K, self.M)


class LibraBackend:
    """Libra TCU/SPTC/CUDA three-route pipeline (needs npz graph files)."""
    name = "libra"

    def __init__(self):
        self.prep_A_s = self.prep_AT_s = 0.0
        self._iA = self._iAT = None
        self.n = 0
        self.dimN = DIM_N
        self._last_kernel_ms = float("nan")

    @staticmethod
    def _write_npz(A, tag):
        d = GNN_DATA_ROOT / tag
        p = d / f"{tag}.npz"
        if p.exists():
            return tag          # 复用已落盘的图文件, 避免把写盘算进 prep
        coo = A.tocoo()
        d.mkdir(parents=True, exist_ok=True)
        np.savez(p,
                 src_li=coo.row.astype(np.int64), dst_li=coo.col.astype(np.int64),
                 values=coo.data.astype(np.float32),
                 num_nodes_src=np.int64(A.shape[0]), num_nodes_dst=np.int64(A.shape[1]))
        return tag

    def _build(self, A, tag):
        from tr_csr_fp16.mdataset2_online import GCN_dataset_tcu_sptc_cuda_online
        os.environ["TR_DATA_ROOT"] = str(GNN_DATA_ROOT)
        return GCN_dataset_tcu_sptc_cuda_online(
            tag, self.dimN, 8, 32, 32, 3, tag, 16, 16,
            tc_min_util=0.0, sptc_min_util=0.5, sptc_threshold=12)

    def preprocess(self, A, device, assume_undirected=False):
        # tag 必须随矩阵变化: _write_npz 会复用同名文件, 固定 tag 会让第二个
        # 矩阵读到上一个矩阵的图
        tag = f"e2e_{A.shape[0]}_{A.shape[1]}_{A.nnz}"
        t = time.perf_counter()
        self._iA = self._build(A, self._write_npz(A, tag))
        self.n = self._iA.num_nodes_ori
        self.prep_A_s = time.perf_counter() - t
        t = time.perf_counter()
        if assume_undirected:
            self._iAT, self.prep_AT_s = self._iA, 0.0
        else:
            tagt = f"{tag}_T"
            self._iAT = self._build(A.T.tocsr(), self._write_npz(A.T.tocsr(), tagt))
            self.prep_AT_s = time.perf_counter() - t

    def _call(self, info, X):
        from tr_csr_fp16 import test_tr_csr_online
        info.x = X.detach().to(torch.float16)   # 保持 device, GPU 直通
        out, t_ms, _, _ = test_tr_csr_online.kernel_tcu_sptc_cuda_online(info, 1)
        self._last_kernel_ms = float(t_ms.item() if torch.is_tensor(t_ms) else t_ms)
        return out[: self.n]

    def spmm_A(self, X):
        return self._call(self._iA, X)

    def spmm_AT(self, dY):
        return self._call(self._iAT, dY)


class MPBackend:
    """MP-SpMM: 官方 CLI 基准程序 -> pybind 封装 (baseline/mp/MP_SPMM_WRAP).

    预处理默认现场执行官方 impl-iterative-2-4 (从 mtx 生成 <name>_data.bin),
    五段计时 (read/pack/match/padding/unwrap) 全部计入 prep_s —— 与 TR/Libra
    的口径一致, 也不再依赖外部预生成产物. DR_MP_PREP=reuse 可退回旧开发模式
    (直接读 DR_MP_BIN_DIR 下既有 bin, 不计官方预处理).
    对称矩阵的 A^T == A, 共用同一份 packed (与 TR/Libra 的
    --assume-undirected 语义一致).
    """
    name = "mp"
    MP_DIR = Path(os.environ.get("DR_MP_BIN_DIR", "/data1/datasets/mp_processed_2-4"))
    # 匹配策略: adjacent 与既有 e2e 结果同口径 (/data1 的 bin 即
    # 0.Adjacent-matching 产物, md5 已验证可现场精确复现);
    # DR_MP_MATCHING=iterative 可切换到 1.Iterative-matching.
    MP_MATCHING = os.environ.get("DR_MP_MATCHING", "adjacent")
    _MP_TOOLS = {
        "adjacent": ("impl-adjacent-matching-2-4", "0.Adjacent-matching"),
        "iterative": ("impl-iterative-2-4", "1.Iterative-matching"),
    }
    MP_PREP_BIN = MP_PKG / "preprocessing" / _MP_TOOLS[MP_MATCHING][0]
    # 官方预处理的输出布局 (相对 dataset_path)
    MP_OFFICIAL_SUB = Path("dataset_mp_processed/2-4") / _MP_TOOLS[MP_MATCHING][1]

    def __init__(self):
        self.prep_A_s = self.prep_AT_s = 0.0
        self._pa = self._pat = None
        self._m = self._k = 0
        self._sym = False
        self.dataset = None

    def set_dataset(self, name):
        self.dataset = name

    @staticmethod
    def _read_bin(path):
        import struct
        with open(path, "rb") as f:
            def ri():
                return struct.unpack("<i", f.read(4))[0]
            nr, nc, nnz = ri(), ri(), ri()
            n_off = ri()
            off = np.frombuffer(f.read(n_off * 4), dtype=np.int32).copy()
            n_a = ri()
            a = np.frombuffer(f.read(n_a * 8), dtype=np.float64).copy()
            n_m = ri()
            meta = np.frombuffer(f.read(n_m * 4), dtype=np.int32).copy()
            n_c = ri()
            col = np.frombuffer(f.read(n_c * 4), dtype=np.int32).copy()
        return nr, nc, off, a, meta, col

    @staticmethod
    def _prepare(path):
        import MP_SPMM_WRAP
        nr, nc, off, a, meta, col = MPBackend._read_bin(path)
        a[np.isnan(a)] = 0.0
        col = col.astype(np.int32)
        col[col == -1] = 0
        off_t = torch.from_numpy(off)
        a_t = torch.from_numpy(a.astype(np.float16))
        col_t = torch.from_numpy(col)
        packed = MP_SPMM_WRAP.pack_metadata(torch.from_numpy(meta), off_t)
        packed_t = torch.from_numpy(np.asarray(packed, dtype=np.uint32).view(np.int32).copy())
        return dict(off=off_t, a=a_t, meta=packed_t, col=col_t, rows=nr, cols=nc)

    @staticmethod
    def _official_prep(name, timeout=3600):
        """现场执行 MP 官方预处理, 返回 (bin 路径, 耗时 s).

        官方口径 = stderr 五段之和: read_matrix + pack_columns_into_units +
        match_units + padding + unwrap_metadata. 程序按 cwd 相对路径读
        ../../path.txt 取 dataset_path, 这里动态生成指向 REAL_DIR; 产物按
        匹配策略写在 REAL_DIR/dataset_mp_processed/2-4/<策略目录>/<name>_data.bin.
        """
        binp = MPBackend.MP_PREP_BIN
        if not binp.exists():
            raise RuntimeError(
                f"MP 官方预处理二进制缺失: {binp} "
                "(在 baseline/mp/preprocessing/ 下编译, 见 build.sh; "
                f"当前匹配策略 {MPBackend.MP_MATCHING})")
        path_cfg = BASE_DIR / "path.txt"
        path_cfg.write_text(f"dataset_path = {REAL_DIR}\n")
        r = subprocess.run([str(binp), name], cwd=str(binp.parent),
                           capture_output=True, text=True, timeout=timeout)
        ts = [float(x) for x in re.findall(r"\[([0-9.]+) ms\]", r.stderr)]
        out = REAL_DIR / MPBackend.MP_OFFICIAL_SUB / f"{name}_data.bin"
        if r.returncode != 0 or not ts or not out.exists():
            raise RuntimeError(
                f"MP official preprocessing failed rc={r.returncode} "
                f"stderr_tail={r.stderr[-400:]}")
        print(f"    [mp-prep] {name}: official 5-stage total = "
              f"{sum(ts) / 1e3:.3f}s", file=sys.stderr)
        return out, sum(ts) / 1e3

    def preprocess(self, A, device, assume_undirected=False):
        if not self.dataset:
            raise RuntimeError("MPBackend needs dataset name")
        mode = os.environ.get("DR_MP_PREP", "official")
        if mode == "reuse":
            # 旧开发模式: 只读既有 bin, prep 仅含读盘 + 元数据打包
            p = self.MP_DIR / f"{self.dataset}_data.bin"
            if not p.exists():
                raise RuntimeError(f"missing MP preprocessed file {p}")
            official_s = 0.0
        else:
            p, official_s = self._official_prep(self.dataset)
        t = time.perf_counter()
        self._pa = self._prepare(p)
        self.prep_A_s = official_s + (time.perf_counter() - t)
        self._m, self._k = A.shape

        t = time.perf_counter()
        if assume_undirected:
            self._pat, self.prep_AT_s = self._pa, 0.0
        else:
            pt = self.MP_DIR / f"{self.dataset}_T_data.bin"
            if pt.exists():
                self._pat = self._prepare(pt)
            else:
                # 对称矩阵 A^T == A; 非对称且缺 A^T 产物时直接报错, 不做静默近似
                sym = (A.nnz == A.T.nnz) and ((A != A.T).nnz == 0)
                if not sym:
                    raise RuntimeError(f"{self.dataset} is non-symmetric and "
                                       f"{pt.name} is missing")
                self._pat = self._pa
            self.prep_AT_s = time.perf_counter() - t

    @staticmethod
    def _call(p, X, M, K):
        import MP_SPMM_WRAP
        # rhs 零拷贝: 必须是 half 且连续 (wrapper 直接把它当 B 矩阵用)
        return MP_SPMM_WRAP.forward(p["off"], p["a"], p["meta"], p["col"],
                                    X.detach().to(torch.float16).contiguous(),
                                    M, K, int(X.size(1)))

    def spmm_A(self, X):
        return MPBackend._call(self._pa, X, self._m, self._k)

    def spmm_AT(self, dY):
        return MPBackend._call(self._pat, dY, self._k, self._m)


class DTCBackend:
    """DTC-SpMM (ASPLOS'24) 复现, 自 TR-SPMM_zzx/gnn_e2e.py 的 run_dtc 移植.

    流程: TCA 行重排 (纯 CPU, datasketch; 耗时计入预处理, 大图上可达
    数百秒) + DTCSpMM.preprocess_gpu (GPU 元数据构建) +
    run_DTCSpMM dense-TC kernel.

    口径说明:
      * 官方 kernel 的 valuesA 恒为 1 (无权图 SpMM): 加权矩阵上按二值
        邻接计算, 与旧图口径一致; 数值校验时用二值 A 做参考.
      * kernel 输入输出为 fp32 (TF32 路径), 输出行在重排坐标系下,
        用 reorder_id (new→old) 反置换回原始节点序以支持两层链式
        前向与 A^T 反向; 反置换 kernel 也计入调用墙钟.
      * TCA 重排随 nnz 平方级增长, 大矩阵上常见超时; DTC_PREP_TIMEOUT
        (默认 3600 s, 0 = 不限时) 超时抛 TimeoutError, 由上层记 SKIP/TO.
    """
    name = "dtc"

    def __init__(self):
        self.prep_A_s = self.prep_AT_s = 0.0
        self._pA = self._pAT = None
        self._last_kernel_ms = float("nan")

    @staticmethod
    def _exeplan(dimN):
        # kernel 内部按维度选择守卫: float4_split 需 dim>=128, float2_split >=64
        if dimN >= 128:
            return "float4_split"
        if dimN >= 64:
            return "float2_split"
        return "float_nonsplit"

    @staticmethod
    def _build(A):
        import DTCSpMM
        from tr_csr_fp16 import tr_tca_reorder
        rows, cols = A.shape
        t0 = time.perf_counter()
        new_ptr, new_idx, reorder_id = tr_tca_reorder.apply_tca_reorder(
            A.indptr.astype(np.int32), A.indices.astype(np.int32),
            rows, thres=16, verbose=False)
        reorder_s = time.perf_counter() - t0
        nnz = int(len(new_idx))
        # [fix] 官方 DTC_dataset 会把行数补齐到 BLK_H=16 的倍数: kernel 每个
        # window 固定写 16 行输出, 未补齐时末窗会越界写 2.5KB 到相邻显存
        # (症状: 随分配布局而定的确定性乱值). indptr 追加空行, 输出再切片.
        rows_pad = (rows + 15) // 16 * 16
        if rows_pad > rows:
            new_ptr = np.concatenate(
                [new_ptr, np.full(rows_pad - rows, int(new_ptr[-1]),
                                  dtype=np.int32)])
        rp_t = torch.from_numpy(
            np.ascontiguousarray(new_ptr, dtype=np.int32)).cuda()
        ci_t = torch.from_numpy(
            np.ascontiguousarray(new_idx, dtype=np.int32)).cuda()
        bp = torch.zeros((rows_pad + 15) // 16, dtype=torch.int32,
                         device="cuda")
        ec = torch.zeros(nnz, dtype=torch.int32, device="cuda")
        er = torch.zeros(nnz, dtype=torch.int32, device="cuda")
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        rw, trid, ttid, toff, sax, _ = DTCSpMM.preprocess_gpu(
            ci_t, rp_t, rows_pad, 16, 8, bp, ec, er)
        torch.cuda.synchronize()
        prep = reorder_s + (time.perf_counter() - t1)
        rid = torch.from_numpy(
            np.ascontiguousarray(reorder_id, dtype=np.int64)).cuda()
        return dict(rw=rw, ttid=ttid, toff=toff, sax=sax,
                    rows=rows, rows_pad=rows_pad, nnz=nnz, rid=rid), prep

    @staticmethod
    def _run(p, X):
        import DTCSpMM
        Xf = X.detach().to(torch.float32).contiguous()
        out, kms = DTCSpMM.run_DTCSpMM(
            Xf, p["rw"], p["ttid"], p["toff"], p["sax"],
            p["rows_pad"], p["nnz"], DTCBackend._exeplan(Xf.size(1)))
        out = out[: p["rows"]]           # 去掉补齐的空行
        # 输出行在重排坐标系 (rid[new]=old), scatter 回原始节点序:
        # out_orig[rid[i]] = out_new[i] (index_select 是 gather, 方向相反)
        out_orig = torch.zeros_like(out)
        out_orig[p["rid"]] = out
        return out_orig, float(kms.item())

    def preprocess(self, A, device, assume_undirected=False):
        timeout = float(os.environ.get("DTC_PREP_TIMEOUT", "3600"))

        def _alarm(signum, frame):
            raise TimeoutError(
                f"DTC preprocessing exceeded DTC_PREP_TIMEOUT={timeout}s "
                "(TCA 重排为主, 大图常见)")

        old_handler = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout if timeout > 0 else 0)
        try:
            self._pA, prep = self._build(A)
            self.prep_A_s = prep
            if assume_undirected:
                self._pAT, self.prep_AT_s = self._pA, 0.0
            else:
                self._pAT, self.prep_AT_s = self._build(A.T.tocsr())
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old_handler)

    def spmm_A(self, X):
        out, kms = self._run(self._pA, X)
        self._last_kernel_ms = kms
        return out

    def spmm_AT(self, dY):
        out, kms = self._run(self._pAT, dY)
        self._last_kernel_ms = kms
        return out


BACKENDS = {"ref": RefBackend, "tr": TRBackend, "libra": LibraBackend,
            "mp": MPBackend, "dtc": DTCBackend}


# =====================================================================
# Data / correctness
# =====================================================================
def load_mtx(path):
    t = time.perf_counter()
    mat = mmread(str(path))
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    mat.sum_duplicates()
    mat.sort_indices()
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    return mat, time.perf_counter() - t


def discover(names=None):
    if names:
        return [(n, REAL_DIR / n / f"{n}.mtx") for n in names]
    return [(d.name, d / f"{d.name}.mtx") for d in sorted(REAL_DIR.iterdir())
            if d.is_dir() and (d / f"{d.name}.mtx").exists()]


@torch.no_grad()
def check_vs_ref(backend, A, device, dimn, binary=False,
                 tol_abs=1e-1, tol_rel=5e-3):
    # binary=True: DTC 官方 kernel 的 valuesA 恒为 1, 加权矩阵上按二值
    # 邻接计算, 参考解须同样二值化.
    if binary:
        A = A.copy()
        A.data = np.ones_like(A.data)
    n, k = A.shape
    X = torch.randn(k, dimn, device=device, dtype=torch.float32)
    Y_ref = torch.sparse.mm(_csr_to_torch_sparse(A, device), X)
    Y = backend.spmm_A(X).float()
    ea = float((Y - Y_ref).abs().max().item())
    ra = ea / (float(Y_ref.abs().max().item()) + 1e-8)

    dY = torch.randn(n, dimn, device=device, dtype=torch.float32)
    dX_ref = torch.sparse.mm(_csr_to_torch_sparse(A.T.tocsr(), device), dY)
    dX = backend.spmm_AT(dY).float()
    eb = float((dX - dX_ref).abs().max().item())
    rb = eb / (float(dX_ref.abs().max().item()) + 1e-8)
    ok = (ea < tol_abs or ra < tol_rel) and (eb < tol_abs or rb < tol_rel)
    return ok, ea, eb


# =====================================================================
# Main
# =====================================================================
def run_one(name, A, load_s, method, args, device):
    n, k = A.shape
    if n != k:
        raise RuntimeError(f"non-square {A.shape}")
    backend = BACKENDS[method]()
    if hasattr(backend, "set_dataset"):
        backend.set_dataset(name)
    backend.preprocess(A, device, assume_undirected=args.assume_undirected)

    if args.check and method != "libra":
        # Libra: 官方 online pipeline 未提供数值校验接口 (其 mma fragment
        # 布局与输出映射不自洽), 仅用于性能对比, 故跳过数值检查.
        ok, ea, eb = check_vs_ref(backend, A, device, args.dimN,
                                  binary=(method == "dtc"))
        print(f"  [check] {name} {method} fwd={ea:.3e} bwd={eb:.3e} "
              f"{'OK' if ok else 'FAIL'}", file=sys.stderr)
        if not ok:
            raise RuntimeError(f"numeric check failed fwd={ea:.3e} bwd={eb:.3e}")

    model = GCN2(args.dimN).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    X = torch.randn(n, args.dimN, device=device, dtype=torch.float32)
    y = torch.randn(n, args.dimN, device=device, dtype=torch.float32)

    def one_epoch():
        opt.zero_grad(set_to_none=True)
        out = model(X, backend)
        loss = F.mse_loss(out, y)
        loss.backward()
        opt.step()
        return loss

    for _ in range(args.warmup):
        one_epoch()
    torch.cuda.synchronize()

    _reset_cnt()
    one_epoch()
    torch.cuda.synchronize()
    if _FWD != 2 or _BWD != 2:
        raise RuntimeError(f"SpMM count fwd={_FWD} bwd={_BWD} (expect 2/2)")

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.epochs):
        one_epoch()
    torch.cuda.synchronize()
    avg_ms = (time.perf_counter() - t0) / args.epochs * 1e3

    return {
        "dataset": name, "rows": n, "cols": k, "nnz": int(A.nnz),
        "method": method, "load_s": f"{load_s:.4f}",
        "prep_A_s": f"{backend.prep_A_s:.4f}", "prep_AT_s": f"{backend.prep_AT_s:.4f}",
        "warmup": args.warmup, "epochs": args.epochs,
        "avg_epoch_ms": f"{avg_ms:.4f}", "fwd_spmm": 2, "bwd_spmm": 2,
    }


HEADER = ["dataset", "rows", "cols", "nnz", "method", "load_s",
          "prep_A_s", "prep_AT_s", "warmup", "epochs", "avg_epoch_ms",
          "fwd_spmm", "bwd_spmm"]


def main():
    global DIM_N
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrices", nargs="+", default=["ddi"])
    ap.add_argument("--methods", nargs="+", default=["ref", "tr", "libra"],
                    choices=["ref", "tr", "libra", "mp", "dtc"])
    ap.add_argument("--dimN", type=int, default=DIM_N)
    ap.add_argument("--warmup", type=int, default=WARMUP)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--assume-undirected", action="store_true")
    ap.add_argument("--check", action="store_true", default=False)
    ap.add_argument("--csv", type=str, default=str(RESULTS_DIR / "gnn_train_128.csv"))
    ap.add_argument("--resume", action="store_true",
                    help="保留 CSV 中已完成的 (dataset, method), 只补跑缺的")
    args = ap.parse_args()
    device = "cuda"

    DIM_N = args.dimN
    LibraBackend.dimN = args.dimN

    mats = discover(args.matrices)
    rows = []
    done = set()
    if args.resume and Path(args.csv).exists():
        for r in csv.DictReader(open(args.csv)):
            if r.get("avg_epoch_ms"):
                done.add((r["dataset"], r["method"]))
                rows.append([r[h] for h in HEADER])
        print(f"[resume] 已有 {len(done)} 条结果", file=sys.stderr)
    for name, path in mats:
        if not path.exists():
            print(f"[SKIP] {name}: {path} missing", file=sys.stderr)
            continue
        A, load_s = load_mtx(path)
        print(f"=== {name} {A.shape} nnz={A.nnz} load={load_s:.2f}s ===", file=sys.stderr)
        if name == "web-BerkStan" and args.assume_undirected:
            raise RuntimeError("web-BerkStan is directed; --assume-undirected forbidden")
        # 换矩阵前释放上一矩阵的预打包缓存, 否则多个 packed 同时驻留会 OOM
        try:
            _TRSPMM.clear_prep_cache()
        except AttributeError:
            pass
        torch.cuda.empty_cache()
        for m in args.methods:
            if (name, m) in done:
                print(f"[skip] {name} {m}: 已完成", file=sys.stderr)
                continue
            try:
                rec = run_one(name, A, load_s, m, args, device)
                rows.append([rec[h] for h in HEADER])
                print(f"[OK] {name} {m}: avg_epoch={rec['avg_epoch_ms']}ms "
                      f"prepA={rec['prep_A_s']}s prepAT={rec['prep_AT_s']}s", file=sys.stderr)
            except Exception as e:  # noqa: BLE001
                print(f"[FAIL] {name} {m}: {e}", file=sys.stderr)
                rows.append([name, A.shape[0], A.shape[1], A.nnz, m] + [""] * 8)
            gc.collect()
            torch.cuda.empty_cache()
        del A
        gc.collect()
        torch.cuda.empty_cache()

        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(HEADER)
            w.writerows(rows)

    print(f"[done] {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
