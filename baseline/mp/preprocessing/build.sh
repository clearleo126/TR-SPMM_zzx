#!/usr/bin/env bash
# 编译 MP-SpMM 官方预处理工具 (纯 CPU, 只依赖 g++ 与 libmmio).
# gnn_train.py / gnn_e2e.py 的 MPBackend 会现场执行匹配程序,
# 从 <DR_REAL_DIR>/<name>/<name>.mtx 生成 2:4 结构化的 <name>_data.bin,
# 并把五段耗时 (read/pack/match/padding/unwrap) 计入 MP 预处理.
#
# 默认匹配策略是 adjacent (0.Adjacent-matching, 与既有 e2e 结果同口径);
# iterative (1.Iterative-matching) 也一并编出, 供 DR_MP_MATCHING=iterative.
#
# 用法: bash build.sh   (在本目录下执行)
set -e
cd "$(dirname "$0")"
g++ -O3 -o impl-adjacent-matching-2-4 impl-adjacent-matching-2-4.cpp libmmio/mmio.c
echo "built: impl-adjacent-matching-2-4"
g++ -O3 -o impl-iterative-2-4 impl-iterative-2-4.cpp libmmio/mmio.c
echo "built: impl-iterative-2-4"
# 2.Maximum-matching 本流程未用到, 需要时自行编译:
# g++ -O3 -o max_match-2-4 max_match-2-4.cpp libmmio/mmio.c
