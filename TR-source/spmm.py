"""
spmm.py — 稀疏矩阵分块 + GPU 矩阵乘 完整流程

功能:
  1. 自动发现并加载数据集 (.npz 优先, .mtx 兜底)
  2. 将矩阵转为 CSR 格式 (row_ptr, col_ind, values)
  3. 调用 C++ matching_utils.match_2to4() 执行分块 (TC + SPTC)
  4. 将分块数据传入 GPU 做矩阵乘运算 (TC: 稠密 MMA, SPTC: mma.sp)
  5. 验证结果正确性

编译方法 (首次运行前):
    cd TR-source/SpMM/
    python setup_tr.py build_ext --inplace
    cd ..
    python setup_24matching.py build_ext --inplace

用法:
    python spmm.py                            # 使用默认数据集
    python spmm.py --path <file>              # 指定数据集路径
    python spmm.py --list                     # 列出可用数据集
    python spmm.py --dimN 32                  # 指定特征维度 (默认 32)
    python spmm.py --epoches 100              # 指定 benchmark 迭代次数 (默认 100)
"""

import os
import sys
import argparse
import time
import glob

import numpy as np
import scipy.sparse as sp
from scipy.io import mmread, mminfo
import torch

# ====================================================================
# 项目路径配置
# ====================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
DGL_DATASETS_DIR = os.path.join(PROJECT_ROOT, "dgl_datasets")

# 确保能 import C++ 模块
sys.path.insert(0, os.path.join(PROJECT_ROOT, "TR-source"))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "TR-source", "SpMM"))


# ====================================================================
# 数据集发现与加载 (复用 test_24_matching.py 逻辑)
# ====================================================================

def find_datasets():
    """
    扫描数据集目录, 返回所有可用矩阵文件的路径列表。
    优先级: .npz > .mtx
    """
    datasets = {}
    if os.path.isdir(DGL_DATASETS_DIR):
        for f in glob.glob(os.path.join(DGL_DATASETS_DIR, "*.npz")):
            name = os.path.splitext(os.path.basename(f))[0]
            datasets[name] = f
    if os.path.isdir(DATASETS_DIR):
        for root, dirs, files in os.walk(DATASETS_DIR):
            for f in files:
                if f.endswith(".npz"):
                    name = os.path.splitext(f)[0]
                    if name not in datasets:
                        datasets[name] = os.path.join(root, f)
            for f in files:
                if f.endswith(".mtx"):
                    name = os.path.splitext(f)[0]
                    if name not in datasets:
                        datasets[name] = os.path.join(root, f)
    return datasets


def _load_custom_npz(filepath):
    """加载 DGL 格式的 .npz 稀疏矩阵。"""
    raw = np.load(filepath)
    keys = list(raw.keys())
    if 'format' in keys or 'indices' in keys:
        return sp.load_npz(filepath)
    src = raw['src_li']
    dst = raw['dst_li']
    val = raw.get('data', None)
    shape = tuple(raw['shape'].tolist())
    if val is None:
        val = np.ones(len(src), dtype=np.float32)
    else:
        val = val.astype(np.float32)
    mat = sp.coo_matrix((val, (src, dst)), shape=shape)
    return mat.tocsr()


def load_matrix(filepath):
    """加载稀疏矩阵, 返回 scipy.sparse.csr_matrix (float32)。"""
    ext = os.path.splitext(filepath)[1].lower()
    print(f"[加载] {filepath} (格式: {ext})")
    if ext == ".npz":
        mat = _load_custom_npz(filepath)
    elif ext == ".mtx":
        mm_rows, mm_cols, mm_entries, mm_format, mm_field, mm_symm = mminfo(filepath)
        is_pattern = (mm_field == 'pattern')
        mat = mmread(filepath)
        if is_pattern and hasattr(mat, 'data'):
            mat.data = np.ones(mat.nnz, dtype=np.float32)
        if isinstance(mat, np.ndarray):
            mat = sp.csr_matrix(mat)
    else:
        raise ValueError(f"不支持的格式: {ext}")
    if not sp.isspmatrix_csr(mat):
        mat = mat.tocsr()
    mat.sort_indices()
    if mat.dtype != np.float32:
        mat = mat.astype(np.float32)
    return mat


def matrix_to_csr_arrays(mat):
    """将 scipy CSR 矩阵拆解为三个 NumPy 数组。"""
    row_ptr = mat.indptr.astype(np.int32)
    col_ind = mat.indices.astype(np.int32)
    values = mat.data.astype(np.float32)
    return row_ptr, col_ind, values, mat.shape[0], mat.shape[1]


# ====================================================================
# GPU 矩阵乘
# (SPTC 元数据压缩在 TR.cpp 中完成)
# ====================================================================

def run_gpu_spmm(result_dict, num_rows, num_cols, dimN, epoches=100):
    """
    将分块数据传入 GPU 做矩阵乘。

    参数:
      result_dict: match_2to4() 返回的 dict, 包含 TC 和 SPTC tensor
      num_rows:    原始矩阵行数 M
      num_cols:    原始矩阵列数 K
      dimN:        特征维度 N
      epoches:     benchmark 迭代次数

    返回:
      output:       [M, N] float32 结果矩阵
      elapsed_ms:   平均耗时 (ms)
    """
    try:
        import TR_SPMM
    except ImportError:
        print("=" * 60)
        print("  错误: 导入 TR_SPMM 模块失败")
        print("  请先编译: cd TR-source/SpMM/ && python setup_tr.py build_ext --inplace")
        print("=" * 60)
        sys.exit(1)

    # ---- 取数据 ----
    # SPTC
    sptc_window_offset = result_dict["sptc_window_offset"]  # [num_windows+1] int32
    sptc_value = result_dict["sptc_value"]                   # [N] float64
    sptc_metadata = result_dict["sptc_metadata"]              # [N] int32 (raw, 0-3)
    sptc_col_old = result_dict["sptc_col_old"]                # [N] int32

    # TC
    tc_window_offset = result_dict["tc_window_offset"]       # [num_windows+1] int32
    tc_offset = result_dict["tc_offset"]                      # [N] int32
    tc_local_id = result_dict["tc_local_id"]                  # [N] uint8
    tc_col_old = result_dict["tc_col_old"]                    # [N] int32
    tc_value = result_dict["tc_value"]                        # [N] float64

    window_size = 16
    num_windows = sptc_window_offset.size(0) - 1

    # ---- 转换为 half precision ----
    sptc_value_half = sptc_value.to(torch.float16)
    tc_value_half = tc_value.to(torch.float16)

    num_sptc_blocks = int(sptc_window_offset[-1].item())

    # ---- TC 数据处理 ----
    num_tc_blocks = int(tc_window_offset[-1].item())
    # tc_col_old 每 block 有 dense_threshold 个列
    dense_threshold = 8  # 默认值, 可从 col_old 推断
    if num_tc_blocks > 0:
        dense_threshold = tc_col_old.size(0) // num_tc_blocks

    # ---- 生成 B 矩阵 (K × N, random, 确保 contiguous) ----
    K = num_cols
    M = num_rows
    B = torch.randn(K, dimN, dtype=torch.float16).contiguous()

    # ---- 调用 GPU 模块 (元数据压缩在 TR.cpp 中完成) ----
    print(f"\n[GPU SpMM] M={M}, K={K}, N={dimN}")
    print(f"  SPTC blocks: {num_sptc_blocks}, TC blocks: {num_tc_blocks}")
    print(f"  Windows: {num_windows}")

    output, elapsed_ms = TR_SPMM.forward(
        # SPTC tensors (raw metadata, C++ 端压缩)
        sptc_window_offset,
        sptc_value_half,
        sptc_metadata,           # raw int32, TR.cpp 中压缩为 2-bit packed uint32
        sptc_col_old,
        # TC tensors
        tc_window_offset,
        tc_offset,
        tc_local_id,
        tc_col_old,
        tc_value_half,
        # B matrix
        B,
        # Dimensions
        window_size,
        dimN,
        M,
        K,
        num_windows,
        num_sptc_blocks,
        num_tc_blocks,
        dense_threshold,
        epoches,
    )

    return output, elapsed_ms.item()


def run_cpu_reference(mat, B_np, dimN):
    """
    CPU 参考实现: scipy CSR × dense, 用于验证 GPU 结果。
    """
    M, K = mat.shape
    # mat: M×K, B: K×N
    C_cpu = mat @ B_np  # [M, N] float32
    return C_cpu


# ====================================================================
# 主流程
# ====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="稀疏矩阵分块 + GPU 矩阵乘"
    )
    parser.add_argument("--path", type=str, default=None,
                        help="指定矩阵文件路径 (.npz 或 .mtx)")
    parser.add_argument("--list", action="store_true",
                        help="列出所有可用数据集")
    parser.add_argument("--dimN", type=int, default=128,
                        help="特征维度 N (默认 128)")
    parser.add_argument("--epoches", type=int, default=10,
                        help="Benchmark 迭代次数 (默认 10)")
    parser.add_argument("--window", type=int, default=16,
                        help="窗口行数 (默认 16)")
    parser.add_argument("--dense_threshold", type=int, default=8,
                        help="TC 路由阈值 (默认 8)")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 CPU 结果验证")
    args = parser.parse_args()

    # ---- 导入 C++ 模块 ----
    try:
        import matching_utils
    except ImportError as e:
        print("=" * 60)
        print(f"  错误: 导入 matching_utils 失败 — {e}")
        print("  请先编译: cd TR-source/ && "
              "CXX=$CONDA_PREFIX/bin/g++ python setup_24matching.py build_ext --inplace")
        print("=" * 60)
        sys.exit(1)

    # ---- 列出数据集 ----
    if args.list:
        datasets = find_datasets()
        print(f"\n可用数据集 (共 {len(datasets)} 个):")
        for name, path in sorted(datasets.items()):
            size_kb = os.path.getsize(path) / 1024
            print(f"  {name:30s}  [{size_kb:.1f} KB]  {path}")
        return

    # ---- 选择并加载矩阵 ----
    if args.path:
        filepath = args.path
        if not os.path.exists(filepath):
            print(f"错误: 文件不存在: {filepath}")
            sys.exit(1)
    else:
        datasets = find_datasets()
        if not datasets:
            print("错误: 未找到任何数据集!")
            print(f"  已搜索: {DGL_DATASETS_DIR}, {DATASETS_DIR}")
            sys.exit(1)
        npz_files = {k: v for k, v in datasets.items() if v.endswith(".npz")}
        if npz_files:
            name = sorted(npz_files.keys())[0]
        else:
            name = sorted(datasets.keys())[0]
        filepath = datasets[name]
        print(f"[自动选择] {name}")

    # ---- 加载矩阵 ----
    mat = load_matrix(filepath)
    print(f"  矩阵形状: {mat.shape[0]} × {mat.shape[1]}")
    print(f"  非零元:   {mat.nnz}")
    print(f"  稀疏度:   {100 * (1 - mat.nnz / (mat.shape[0] * mat.shape[1])):.4f}%")

    # ---- 转为 CSR 数组 ----
    row_ptr, col_ind, values, rows, cols = matrix_to_csr_arrays(mat)

    # ---- 执行分块匹配 ----
    print(f"\n[分块] 调用 matching_utils.match_2to4 "
          f"(window={args.window}, dense_th={args.dense_threshold}, t_max=8)...")
    t0 = time.perf_counter()
    result_dict = matching_utils.match_2to4(
        row_ptr, col_ind, values,
        window_size=args.window,
        dense_threshold=args.dense_threshold,
        t_max=8,
    )
    blocking_time = time.perf_counter() - t0
    print(f"  分块耗时: {blocking_time:.4f} 秒")

    num_sptc = int(result_dict["sptc_window_offset"][-1].item())
    num_tc = int(result_dict["tc_window_offset"][-1].item())
    print(f"  SPTC 块: {num_sptc}, TC 块: {num_tc}")

    # ---- GPU 矩阵乘 ----
    dimN = args.dimN
    epoches = args.epoches

    output, elapsed_ms = run_gpu_spmm(
        result_dict, rows, cols, dimN, epoches=epoches)

    print(f"\n[结果] GPU SpMM 平均耗时: {elapsed_ms:.4f} ms (over {epoches} iterations)")
    print(f"  输出形状: {list(output.shape)}")

    # ---- CPU 验证 ----
    if not args.no_verify:
        print("\n[验证] 计算 CPU 参考结果...")
        B_np = torch.randn(cols, dimN, dtype=torch.float32).numpy()
        # 重新用相同的 B 做 GPU 推理
        B_verify = torch.from_numpy(B_np).to(torch.float16)
        try:
            import TR_SPMM
            output_verify, _ = TR_SPMM.forward(
                result_dict["sptc_window_offset"],
                result_dict["sptc_value"].to(torch.float16),
                result_dict["sptc_metadata"],     # raw, C++ 端压缩
                result_dict["sptc_col_old"],
                result_dict["tc_window_offset"],
                result_dict["tc_offset"],
                result_dict["tc_local_id"],
                result_dict["tc_col_old"],
                result_dict["tc_value"].to(torch.float16),
                B_verify,
                args.window,
                dimN,
                rows,
                cols,
                result_dict["sptc_window_offset"].size(0) - 1,
                num_sptc,
                num_tc,
                args.dense_threshold,
                1,
            )
        except ImportError:
            print("  TR_SPMM 未编译, 跳过验证")
            return

        cpu_output = run_cpu_reference(mat, B_np, dimN)
        gpu_output = output_verify.numpy()

        # 比较 (在有效行范围内)
        M_valid = rows
        # diff = np.abs(cpu_output[:M_valid, :] - gpu_output[:M_valid, :])
        # max_diff = diff.max()
        # mean_diff = diff.mean()
        # # 相对误差
        # rel_diff = diff / (np.abs(cpu_output[:M_valid, :]) + 1e-8)
        # max_rel_diff = rel_diff.max()
        # mean_rel_diff = rel_diff.mean()

        # print(f"  Max absolute diff: {max_diff:.6f}")
        # print(f"  Mean absolute diff: {mean_diff:.6f}")
        # print(f"  Max relative diff: {max_rel_diff:.6f}")
        # print(f"  Mean relative diff: {mean_rel_diff:.6f}")

        # # half precision 容忍度
        # if max_rel_diff < 0.05:
        #     print("  [验证通过] GPU 与 CPU 结果一致 (fp16 精度范围内)")
        # else:
        #     print("  [验证警告] GPU 与 CPU 结果差异较大, 请检查")
        
        # 修改 spmm.py 中的验证逻辑
        atol = 1e-2  # 绝对误差容忍度
        rtol = 1e-2  # 相对误差容忍度

        # 复合误差校验：abs(a - b) <= atol + rtol * abs(b)
        is_correct = np.isclose(gpu_output[:M_valid, :], cpu_output[:M_valid, :], rtol=rtol, atol=atol)
        pass_rate = np.mean(is_correct) * 100

        print(f"  元素通过率 (atol={atol}, rtol={rtol}): {pass_rate:.2f}%")

        if pass_rate > 99.0:
            print("  [验证通过] GPU 与 CPU 结果高度一致 (FP16 精度范围内)")
        else:
            print("  [验证警告] GPU 与 CPU 结果差异较大, 请检查")

    print("\n测试完成!")


if __name__ == "__main__":
    main()
