#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析 DR 方法相对各 baseline 的加速比分布。

数据文件（result/ 目录）：
  - results_DR_128.csv         : 我们的方法，内核耗时 = elapsed_ms
  - result_libra_128.csv       : baseline Libra，内核耗时 = elapsed_ms
  - compare_n128.csv           : baseline cuSPARSE = cusparse_kernel_ms
                                 baseline MP      = mp_best_ms
  - compare_dtc_rode_n128.csv  : baseline DTC  = dtc_kernel_ms
                                 baseline RODE = rode_kernel_ms

关联方式：按矩阵名（matrix_name / dataset）对齐。
加速比   ：speedup = baseline_kernel_ms / our_elapsed_ms（>1 表示我们更快）。

分桶（闭开区间约定，覆盖所有取值、无重叠）：
  <1x        : speedup <  1.0
  [1x, 1.5x) : 1.0 <= speedup < 1.5
  [1.5x, 2x) : 1.5 <= speedup < 2.0
  >=2x       : speedup >= 2.0

输出：
  1) 每个 baseline 的：各分桶占比(%)、平均加速比、最大加速比（汇总表）
  2) result/speedup_analysis.csv：逐矩阵的每个 baseline 加速比明细
  3) result/speedup_summary.csv ：汇总表
"""

import os

import pandas as pd

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result")

# 我们的方法
OUR_FILE = "results_DR_128.csv"
OUR_KEY = "matrix_name"
OUR_TIME = "elapsed_ms"

# baseline 定义：名称 -> (文件名, 关联键列, 内核耗时列)
BASELINES = [
    ("Libra",     "result_libra_128.csv",      "matrix_name", "elapsed_ms"),
    ("cuSPARSE",  "compare_n128.csv",          "dataset",     "cusparse_kernel_ms"),
    ("MP",        "compare_n128.csv",          "dataset",     "mp_best_ms"),
    ("DTC",       "compare_dtc_rode_n128.csv", "dataset",     "dtc_kernel_ms"),
    ("RODE",      "compare_dtc_rode_n128.csv", "dataset",     "rode_kernel_ms"),
]

# 分桶边界： (label, 下界, 上界)，下界闭、上界开；上界 None 表示 +inf
BUCKETS = [
    ("<1x",        None, 1.0),
    ("[1x, 1.5x)", 1.0,  1.5),
    ("[1.5x, 2x)", 1.5,  2.0),
    (">=2x",       2.0,  None),
]


def bucket_of(speedup):
    """返回 speedup 所属分桶的下标。"""
    for i, (_, lo, hi) in enumerate(BUCKETS):
        if (lo is None or speedup >= lo) and (hi is None or speedup < hi):
            return i
    return None  # 理论上不可达（各桶已覆盖所有非负数）


def main():
    # 读取我们的方法，按矩阵名去重（保留第一个）
    our = pd.read_csv(os.path.join(RESULT_DIR, OUR_FILE))
    our = our.drop_duplicates(subset=OUR_KEY, keep="first").set_index(OUR_KEY)
    print(f"我们的方法：{OUR_FILE}，共 {len(our)} 个矩阵")

    detail = pd.DataFrame(index=our.index)
    detail["ours_elapsed_ms"] = our[OUR_TIME]

    rows = []  # 汇总表数据
    for name, filename, key_col, time_col in BASELINES:
        base = pd.read_csv(os.path.join(RESULT_DIR, filename))
        base = base.drop_duplicates(subset=key_col, keep="first") \
                   .set_index(key_col)[[time_col]] \
                   .rename(columns={time_col: "base_time"})
        # 与我们的方法按矩阵名对齐（内连接），并转成数值
        merged = our[[OUR_TIME]].join(base, how="inner")
        merged[OUR_TIME] = pd.to_numeric(merged[OUR_TIME], errors="coerce")
        merged["base_time"] = pd.to_numeric(merged["base_time"], errors="coerce")
        merged = merged[(merged[OUR_TIME] > 0) & (merged["base_time"] >= 0)] \
                 .dropna()

        speedup = merged["base_time"] / merged[OUR_TIME]
        detail[name] = speedup

        n = len(speedup)
        avg = speedup.mean()
        maxv = speedup.max()

        counts = [int((speedup.map(bucket_of) == i).sum()) for i in range(len(BUCKETS))]
        pct = [f"{c / n * 100:.1f}%" for c in counts] if n else ["-"] * len(BUCKETS)

        print(f"\n[{name}] 匹配 {n} 个矩阵 "
              + " | ".join(f"{b[0]}={p}" for b, p in zip(BUCKETS, pct))
              + f" | 平均 {avg:.3f}x | 最大 {maxv:.3f}x")

        rows.append({
            "baseline": name,
            "num_matrices": n,
            **{b[0]: f"{c / n * 100:.1f}%" for b, c in zip(BUCKETS, counts)},
            "avg_speedup": avg,
            "max_speedup": maxv,
        })

    # ---------- 打印汇总表 ----------
    summary = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("加速比分布汇总表")
    print("=" * 90)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("=" * 90)
    print("注：各桶边界为闭开区间 <1x / [1x,1.5x) / [1.5x,2x) / >=2x。")

    # ---------- 输出 csv ----------
    detail = detail.sort_index()
    detail.index.name = "matrix_name"
    detail.to_csv(os.path.join(RESULT_DIR, "speedup_analysis.csv"),
                  float_format="%.6f")
    summary.to_csv(os.path.join(RESULT_DIR, "speedup_summary.csv"), index=False)

    print(f"\n明细已写入：{os.path.join(RESULT_DIR, 'speedup_analysis.csv')}")
    print(f"汇总已写入：{os.path.join(RESULT_DIR, 'speedup_summary.csv')}")


if __name__ == "__main__":
    main()
