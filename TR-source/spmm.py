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
# print(PROJECT_ROOT)
DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
DGL_DATASETS_DIR = os.path.join(PROJECT_ROOT, "dgl_datasets")

# 确保能 import C++ 模块（本目录自身编译的版本，勿指向主工程）
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.join(_THIS_DIR, "SpMM"))


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
        # mm_rows, mm_cols, mm_entries, mm_format, mm_field, mm_symm = mminfo(filepath)
        # is_pattern = (mm_field == 'pattern')
        mat = mmread(filepath)

        if isinstance(mat, np.ndarray):
            mat = sp.csr_matrix(mat)
            
        if hasattr(mat, 'sum_duplicates'):
            mat.sum_duplicates()
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

def run_gpu_spmm(result_dict, num_rows, num_cols, dimN, epoches=100, mode=0, ablation=0, chunk=0):
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
        print("  请先编译: cd SpMM/ && python setup_tr.py build_ext --inplace")
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
    dense_threshold = 16  # 回退值; 正常由 tc_col_old 每块列数推断 (下行)
    if num_tc_blocks > 0:
        dense_threshold = tc_col_old.size(0) // num_tc_blocks

    # ---- 生成 B 矩阵 (K × N, random, 确保 contiguous) ----
    K = num_cols
    M = num_rows
    B = torch.randn(K, dimN, dtype=torch.float16).contiguous()

    # ---- 调用 GPU 模块 (元数据压缩在 TR.cpp 中完成) ----
    # print(f"\n[GPU SpMM] M={M}, K={K}, N={dimN}")
    # print(f"  SPTC blocks: {num_sptc_blocks}, TC blocks: {num_tc_blocks}")
    # print(f"  Windows: {num_windows}")

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
        mode,
        10,          # warmup 迭代数
        ablation,
        chunk,       # E-chunk 扫描: 0=auto, -1=整窗单CTA, >0=固定 chunk
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

def find_preprocessed_matrices(batch_dir: str) -> list:
    """
    扫描预处理后的目录，返回 [(matrix_name, mtx_path), ...] 列表。
    
    期望目录结构:
      batch_dir/
      ├── ash85/ash85.mtx
      ├── bcsstk01/bcsstk01.mtx
      └── ...
    """
    results = []
    if not os.path.isdir(batch_dir):
        return results
    for entry in sorted(os.listdir(batch_dir)):
        subdir = os.path.join(batch_dir, entry)
        if os.path.isdir(subdir):
            mtx_path = os.path.join(subdir, entry + ".mtx")
            if os.path.isfile(mtx_path):
                results.append((entry, mtx_path))
            else:
                # 兼容：子目录内文件名与目录名不一致
                candidates = glob.glob(os.path.join(subdir, "*.mtx"))
                if candidates:
                    name = os.path.splitext(os.path.basename(candidates[0]))[0]
                    results.append((name, candidates[0]))
    return results


def benchmark_single_matrix(filepath, matrix_name, args, matching_utils):
    """
    对单个矩阵执行完整的 分块 + GPU SpMM 流程。
    
    返回 dict: {matrix_name, rows, cols, nnz, blocking_time_s, elapsed_ms, status}
    失败时 status != "ok"，耗时字段为 None。
    """
    result = {
        "matrix_name": matrix_name,
        "rows": None,
        "cols": None,
        "nnz": None,
        "blocking_time_s": None,
        "elapsed_ms": None,
        "status": "ok",
    }
    try:
        # ---- 加载 ----
        mat = load_matrix(filepath)
        result["rows"] = mat.shape[0]
        result["cols"] = mat.shape[1]
        result["nnz"] = mat.nnz
        print(f"  [{matrix_name}] shape={mat.shape}, nnz={mat.nnz}")

        # ---- CSR ----
        row_ptr, col_ind, values, rows, cols = matrix_to_csr_arrays(mat)

        # ---- 分块 ----
        t0 = time.perf_counter()
        result_dict = matching_utils.match_2to4(
            row_ptr, col_ind, values,
            window_size=args.window,
            dense_threshold=args.dense_threshold,
            t_max=8,
        )
        blocking_time = time.perf_counter() - t0
        result["blocking_time_s"] = round(blocking_time, 6)
        print(f"  [{matrix_name}] 分块耗时: {blocking_time:.4f}s")

        # ---- GPU SpMM ----
        output, elapsed_ms = run_gpu_spmm(
            result_dict, rows, cols,
            dimN=args.dimN, epoches=args.epoches, mode=args.mode, ablation=args.ablation,
            chunk=args.chunk,
        )
        result["elapsed_ms"] = round(elapsed_ms, 4)
        print(f"  [{matrix_name}] GPU SpMM: {elapsed_ms:.4f} ms")


    except Exception as e:
        result["status"] = f"error: {e}"
        print(f"  [{matrix_name}] ❌ 失败: {e}")

    return result

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
    parser.add_argument("--epoches", type=int, default=50,
                        help="Benchmark 迭代次数 (默认 50)")
    parser.add_argument("--window", type=int, default=16,
                        help="窗口行数 (默认 16)")
    parser.add_argument("--dense_threshold", type=int, default=16,
                        help="TC 残余列打包门槛 (8 或 16, 默认 16)")
    parser.add_argument("--mode", type=int, default=0,
                        help="诊断模式: 0=Full(双流), 1=TC-only, 2=SPTC-only (默认 0)")
    parser.add_argument("--ablation", type=int, default=0,
                        help="消融: 0=baseline, 1=no-A(SPTC双缓冲), 2=no-B(SPTC全N), 3=no-C(SPTC宽B), "
                             "4=no-D(TC稠密tile), 5=no-E(chunking), 6=no-F(单写者写回), "
                             "7=no-G(TC全N), 8=no-H(TC宽B), 9=no-I(m16n8k16), 10=no-AB(双流)")
    parser.add_argument("--chunk", type=int, default=0,
                        help="E-chunk 扫描: 0=自适应(auto), -1=整窗单CTA(同 no-E), "
                             "16/32/64/128/256=固定每CTA块数 (TC/SPTC 共用)")
    parser.add_argument("--no-verify", action="store_true",
                        help="跳过 CPU 结果验证")
    parser.add_argument("--batch-dir", type=str, default=None,
                    help="预处理矩阵集根目录，遍历所有子目录中的 .mtx")
    parser.add_argument("--csv", type=str, default=None,
                    help="批量结果输出 CSV 路径（默认: <batch-dir>/benchmark_results.csv）")
    args = parser.parse_args()

    # ---- 导入 C++ 模块 ----
    try:
        import matching_utils
    except ImportError as e:
        print("=" * 60)
        print(f"  错误: 导入 matching_utils 失败 — {e}")
        print("  请先编译: cd SpMM/ && "
              "CXX=$CONDA_PREFIX/bin/g++ python setup_24matching.py build_ext --inplace")
        print("=" * 60)
        sys.exit(1)


    # ================================================================
    # 批量模式
    # ================================================================
    if args.batch_dir:
        import csv

        matrices = find_preprocessed_matrices(args.batch_dir)
        if not matrices:
            print(f"错误: 在 {args.batch_dir} 中未找到任何 .mtx 文件")
            sys.exit(1)

        print(f"[批量模式] 共 {len(matrices)} 个矩阵, "
            f"dimN={args.dimN}, epoches={args.epoches}")
        print("=" * 70)

        all_results = []
        for idx, (name, path) in enumerate(matrices):
            print(f"\n[{idx+1}/{len(matrices)}] {name}")
            print("-" * 50)
            res = benchmark_single_matrix(path, name, args, matching_utils)
            all_results.append(res)

        # ---- 写 CSV ----
        csv_path = args.csv or os.path.join(args.batch_dir, "benchmark_results.csv")
        fieldnames = [
            "matrix_name", "rows", "cols", "nnz",
            "blocking_time_s", "elapsed_ms", "status",
        ]
        if not args.no_verify:
            fieldnames.append("verify_pass_rate")

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)

        # ---- 汇总 ----
        ok_results = [r for r in all_results if r["status"] == "ok"]
        fail_count = len(all_results) - len(ok_results)
        print("\n" + "=" * 70)
        print(f"[批量完成] 成功: {len(ok_results)}, 失败: {fail_count}, "
            f"总计: {len(all_results)}")
        # if ok_results:
        #     avg_block = np.mean([r["blocking_time_s"] for r in ok_results])
        #     avg_spmm = np.mean([r["elapsed_ms"] for r in ok_results])
        #     med_spmm = np.median([r["elapsed_ms"] for r in ok_results])
        #     print(f"  平均分块耗时: {avg_block:.4f} s")
        #     print(f"  平均 GPU SpMM: {avg_spmm:.4f} ms")
        #     print(f"  中位 GPU SpMM: {med_spmm:.4f} ms")
        print(f"  CSV 已保存: {csv_path}")
        return
    # ================================================================
    # 批量模式
    # ================================================================


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

    # ---- per-window 块分布统计 (straggler 分析) ----
    try:
        sptc_wo = result_dict["sptc_window_offset"].numpy()
        tc_wo = result_dict["tc_window_offset"].numpy()
        sptc_per_win = np.diff(sptc_wo)      # 每窗口 SPTC 块数
        tc_per_win = np.diff(tc_wo)          # 每窗口 TC 块数

        def dist_summary(name, arr):
            if arr.size == 0 or arr.sum() == 0:
                print(f"  [{name}] 无块")
                return
            arr = arr.astype(np.int64)
            nz = arr[arr > 0]
            print(f"  [{name}] 非空窗口数={nz.size}/{arr.size}, "
                  f"均值={arr.mean():.2f}, "
                  f"最大={arr.max()}, "
                  f"P99={np.percentile(arr, 99):.0f}, "
                  f"P90={np.percentile(arr, 90):.0f}, "
                  f"中位={np.median(arr):.0f}")
            # 最大窗口的贡献占比: 最重窗口块数 / 总块数
            print(f"          最大窗口占比={arr.max() / max(arr.sum(), 1) * 100:.2f}%")

        print("  [Per-Window 块分布]")
        dist_summary("SPTC", sptc_per_win)
        dist_summary("TC", tc_per_win)
    except Exception as e:
        print(f"  (per-window 统计失败: {e})")

    # ---- GPU 矩阵乘 ----
    dimN = args.dimN
    epoches = args.epoches

    output, elapsed_ms = run_gpu_spmm(
        result_dict, rows, cols, dimN, epoches=epoches, mode=args.mode, ablation=args.ablation,
        chunk=args.chunk)

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
                1,           # epoches
                0,           # mode
                10,          # warmup
                args.ablation,
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