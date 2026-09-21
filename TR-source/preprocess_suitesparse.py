#!/usr/bin/env python3
"""
SuiteSparse MTX 批量预处理脚本
=================================
功能：遍历 SuiteSparse 目录，将所有矩阵的值统一为 1~10 随机整数，输出标准 MTX。

预处理规则：
  1. 数值：全部替换为 1~10 的随机整数（覆盖原值 / 为 pattern 矩阵生成值）
  2. 对称性：不展开对称矩阵，保留 symmetric/Hermitian 标记
  3. 结构：保留原始条目数（nnz line 中的数字），不做去重、不消除零、不排序
  4. 索引：保持 1-based（MTX 标准）

输出：每个矩阵生成一个预处理后的 .mtx 文件，所有论文代码均可直接读取。

用法：
  python preprocess_suitesparse.py \
      --input_dir  /path/to/SSget   \
      --output_dir /path/to/output  \
      --seed 42                     \
      --workers 8
"""

import os
import sys
import glob
import random
import logging
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# ============================================================
#  日志
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================
#  单矩阵预处理核心逻辑
# ============================================================
def preprocess_single_mtx(input_path: str, output_path: str, seed: int) -> dict:
    """
    逐行解析 MTX 文件，将数值替换为 1~10 随机整数。

    返回一个 info dict，用于汇总统计。
    """
    rng = random.Random(seed)
    info = {
        "input": input_path,
        "output": output_path,
        "status": "ok",
        "rows": 0,
        "cols": 0,
        "nnz": 0,
        "is_symmetric": False,
        "original_type": "unknown",
    }

    try:
        with open(input_path, "r", encoding="utf-8", errors="replace") as fin:
            # ---- 第 1 行：MatrixMarket 头 ----
            header = fin.readline()
            if not header.startswith("%%MatrixMarket"):
                # 有些文件第一行是注释，尝试跳过
                while header.startswith("%"):
                    header = fin.readline()
                if not header.startswith("%%MatrixMarket"):
                    raise ValueError(f"找不到 %%MatrixMarket 头: {input_path}")

            header_lower = header.lower()

            # 解析矩阵属性
            # 格式: %%MatrixMarket matrix coordinate <field> <symmetry>
            tokens = header_lower.split()
            # tokens 示例: ['%%matrixmarket', 'matrix', 'coordinate', 'real', 'general']

            # --- field 类型 ---
            is_pattern = "pattern" in tokens
            is_complex = "complex" in tokens
            is_real = "real" in tokens or "integer" in tokens or "double" in tokens

            if is_pattern:
                info["original_type"] = "pattern"
            elif is_complex:
                info["original_type"] = "complex"
            elif is_real:
                info["original_type"] = "real/integer"
            else:
                info["original_type"] = "unknown"

            # --- 对称性 ---
            is_symmetric = "symmetric" in tokens or "hermitian" in tokens
            is_skew = "skew" in tokens
            info["is_symmetric"] = is_symmetric

            # ---- 构建输出头 ----
            # 输出统一为 real 类型（因为值是整数），保留原始对称性
            if is_symmetric:
                sym_str = "symmetric"
            elif is_skew:
                sym_str = "skew-symmetric"
            else:
                sym_str = "general"

            out_header = f"%%MatrixMarket matrix coordinate real {sym_str}\n"

            # ---- 收集注释行 ----
            comments = []
            line = fin.readline()
            while line.startswith("%"):
                comments.append(line)
                line = fin.readline()

            # ---- 此时 line 应该是 size 行 ----
            # 可能 line 是空的（如果前面注释读多了），跳过空行
            while line.strip() == "":
                line = fin.readline()

            size_parts = line.strip().split()
            if len(size_parts) < 3:
                raise ValueError(f"size 行格式错误: '{line.strip()}' in {input_path}")

            nr, nc, nnz = int(size_parts[0]), int(size_parts[1]), int(size_parts[2])
            info["rows"] = nr
            info["cols"] = nc
            info["nnz"] = nnz

            # ---- 写出 ----
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as fout:
                # 头
                fout.write(out_header)
                # 添加预处理标记注释
                fout.write("% Preprocessed: values randomized to integers in [1,10]\n")
                fout.write(f"% Original type: {info['original_type']}\n")
                # 原始注释（可选，保留以追溯来源）
                for c in comments:
                    fout.write(c)
                # size 行
                fout.write(f"{nr} {nc} {nnz}\n")

                # ---- 逐行处理数据 ----
                count = 0
                for data_line in fin:
                    stripped = data_line.strip()
                    if not stripped:
                        continue

                    parts = stripped.split()
                    if len(parts) < 2:
                        # 可能是行末空行或格式异常，跳过
                        logger.warning(f"  跳过异常行: '{stripped}' in {input_path}")
                        continue

                    row_idx = parts[0]
                    col_idx = parts[1]
                    rand_val = rng.randint(1, 10)

                    fout.write(f"{row_idx} {col_idx} {rand_val}\n")
                    count += 1

                if count != nnz:
                    logger.warning(
                        f"  条目数不匹配: 文件头声明 nnz={nnz}, "
                        f"实际写出 {count} 条 ({input_path})"
                    )
                    info["nnz_actual"] = count

    except Exception as e:
        info["status"] = f"error: {e}"
        logger.error(f"处理失败: {input_path} -> {e}")

    return info


# ============================================================
#  目录扫描：找到所有 .mtx 文件
# ============================================================
def find_mtx_files(input_dir: str) -> list:
    """
    扫描 SuiteSparse 目录，返回 (mtx_path, matrix_name) 列表。

    SuiteSparse 常见目录结构：
      input_dir/
      ├── matrix_name_1/
      │   ├── matrix_name_1.mtx          ← 标准情况
      │   └── ...
      ├── matrix_name_2/
      │   ├── matrix_name_2.mtx
      │   └── ...
      └── ...

    也兼容：
      - .mtx 直接在 input_dir 下
      - .mtx.gz 压缩文件（跳过，提示用户先解压）
      - 矩阵名和目录名不一致的情况
    """
    mtx_files = []

    # 情况 1：子目录结构
    for entry in sorted(os.listdir(input_dir)):
        subdir = os.path.join(input_dir, entry)
        if os.path.isdir(subdir):
            # 在子目录中找 .mtx 文件
            candidates = glob.glob(os.path.join(subdir, "*.mtx"))
            if not candidates:
                # 有些矩阵的文件名不带 .mtx 后缀，尝试所有文件
                candidates = [
                    os.path.join(subdir, f)
                    for f in os.listdir(subdir)
                    if not f.endswith(".gz")
                    and not f.endswith(".png")
                    and not f.endswith(".html")
                    and not f.startswith(".")
                    and os.path.isfile(os.path.join(subdir, f))
                ]
            if candidates:
                # 优先选择与目录同名的 .mtx
                preferred = os.path.join(subdir, entry + ".mtx")
                if preferred in candidates:
                    mtx_files.append((preferred, entry))
                else:
                    # 取第一个 .mtx
                    mtx_files.append((candidates[0], entry))

    # 情况 2：.mtx 直接在 input_dir 下
    direct_mtx = glob.glob(os.path.join(input_dir, "*.mtx"))
    for f in direct_mtx:
        name = os.path.splitext(os.path.basename(f))[0]
        mtx_files.append((f, name))

    # 检查是否有未解压的 .mtx.gz
    gz_files = glob.glob(os.path.join(input_dir, "**", "*.mtx.gz"), recursive=True)
    if gz_files:
        logger.warning(
            f"发现 {len(gz_files)} 个 .mtx.gz 压缩文件，请先解压后再运行本脚本。"
        )

    return mtx_files


# ============================================================
#  批量处理主函数
# ============================================================
def batch_preprocess(input_dir: str, output_dir: str, seed: int, workers: int):
    """批量预处理所有 MTX 文件。"""

    logger.info(f"扫描输入目录: {input_dir}")
    mtx_files = find_mtx_files(input_dir)
    logger.info(f"找到 {len(mtx_files)} 个矩阵")

    if not mtx_files:
        logger.error("未找到任何 .mtx 文件，请检查输入目录。")
        return

    # 构建任务列表
    tasks = []
    for mtx_path, matrix_name in mtx_files:
        # 输出路径：output_dir/matrix_name/matrix_name.mtx
        out_dir = os.path.join(output_dir, matrix_name)
        out_path = os.path.join(out_dir, matrix_name + ".mtx")
        # 每个矩阵用不同的 seed（基于全局 seed + 矩阵名 hash）
        mat_seed = seed + hash(matrix_name) % (2**31)
        tasks.append((mtx_path, out_path, mat_seed, matrix_name))

    # 统计
    stats = {
        "total": len(tasks),
        "success": 0,
        "failed": 0,
        "symmetric": 0,
        "general": 0,
        "pattern": 0,
        "real": 0,
        "complex": 0,
    }

    failed_list = []

    logger.info(f"开始预处理（workers={workers}）...")

    if workers <= 1:
        # 单进程模式（方便调试）
        for i, (inp, outp, s, name) in enumerate(tasks):
            logger.info(f"[{i+1}/{len(tasks)}] {name}")
            info = preprocess_single_mtx(inp, outp, s)
            _update_stats(stats, info, failed_list)
    else:
        # 多进程模式
        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_name = {}
            for inp, outp, s, name in tasks:
                f = executor.submit(preprocess_single_mtx, inp, outp, s)
                future_to_name[f] = name

            for i, future in enumerate(as_completed(future_to_name)):
                name = future_to_name[future]
                try:
                    info = future.result()
                except Exception as e:
                    info = {"status": f"exception: {e}", "original_type": "unknown",
                            "is_symmetric": False}
                    logger.error(f"[{i+1}/{len(tasks)}] {name} 异常: {e}")

                _update_stats(stats, info, failed_list)

                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    logger.info(
                        f"进度: {i+1}/{len(tasks)} "
                        f"(成功={stats['success']}, 失败={stats['failed']})"
                    )

    # ---- 输出汇总 ----
    logger.info("=" * 60)
    logger.info("预处理完成！")
    logger.info(f"  总计:     {stats['total']}")
    logger.info(f"  成功:     {stats['success']}")
    logger.info(f"  失败:     {stats['failed']}")
    logger.info(f"  对称矩阵: {stats['symmetric']}")
    logger.info(f"  一般矩阵: {stats['general']}")
    logger.info(f"  原始类型 → pattern: {stats['pattern']}, "
                f"real: {stats['real']}, complex: {stats['complex']}")
    logger.info(f"  输出目录: {output_dir}")
    logger.info("=" * 60)

    if failed_list:
        logger.warning(f"以下 {len(failed_list)} 个矩阵处理失败:")
        for name, reason in failed_list:
            logger.warning(f"  - {name}: {reason}")

    # 保存处理报告
    report_path = os.path.join(output_dir, "preprocess_report.txt")
    with open(report_path, "w") as f:
        f.write(f"SuiteSparse Preprocessing Report\n")
        f.write(f"{'='*50}\n")
        f.write(f"Total:      {stats['total']}\n")
        f.write(f"Success:    {stats['success']}\n")
        f.write(f"Failed:     {stats['failed']}\n")
        f.write(f"Symmetric:  {stats['symmetric']}\n")
        f.write(f"General:    {stats['general']}\n")
        f.write(f"Pattern:    {stats['pattern']}\n")
        f.write(f"Real:       {stats['real']}\n")
        f.write(f"Complex:    {stats['complex']}\n")
        f.write(f"\nFailed matrices:\n")
        for name, reason in failed_list:
            f.write(f"  {name}: {reason}\n")
    logger.info(f"报告已保存到: {report_path}")


def _update_stats(stats, info, failed_list):
    """更新统计信息。"""
    if info["status"] == "ok":
        stats["success"] += 1
        if info.get("is_symmetric"):
            stats["symmetric"] += 1
        else:
            stats["general"] += 1

        orig = info.get("original_type", "unknown")
        if orig == "pattern":
            stats["pattern"] += 1
        elif orig in ("real/integer",):
            stats["real"] += 1
        elif orig == "complex":
            stats["complex"] += 1
    else:
        stats["failed"] += 1
        name = os.path.basename(os.path.dirname(info.get("input", "")))
        failed_list.append((name, info["status"]))


# ============================================================
#  CLI 入口
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description="SuiteSparse MTX 批量预处理：值随机化为 1~10 整数",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python preprocess_suitesparse.py \\
      --input_dir  /data/SSget \\
      --output_dir /data/preprocessed \\
      --seed 42 \\
      --workers 8
        """,
    )
    parser.add_argument(
        "--input_dir", "-i",
        type=str,
        required=True,
        help="SuiteSparse 根目录（包含各矩阵子目录）",
    )
    parser.add_argument(
        "--output_dir", "-o",
        type=str,
        required=True,
        help="预处理后 MTX 的输出目录",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="随机种子（默认 42），确保可复现",
    )
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=4,
        help="并行进程数（默认 4）",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="仅扫描并列出矩阵，不执行预处理",
    )

    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        logger.error(f"输入目录不存在: {args.input_dir}")
        sys.exit(1)

    if args.dry_run:
        logger.info(f"扫描目录: {args.input_dir}")
        mtx_files = find_mtx_files(args.input_dir)
        logger.info(f"找到 {len(mtx_files)} 个矩阵:")
        for path, name in mtx_files[:20]:
            logger.info(f"  {name}: {path}")
        if len(mtx_files) > 20:
            logger.info(f"  ... 还有 {len(mtx_files) - 20} 个")
        return

    batch_preprocess(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        seed=args.seed,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()