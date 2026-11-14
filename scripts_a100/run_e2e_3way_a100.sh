#!/bin/bash
# 端到端 4 家对比 (A100, CUDA_VISIBLE_DEVICES=0), N=128, 11 矩阵.
# 与任务给定命令一致, 仅把 DR_REAL_DIR 改到本机可写目录 (原 /data1 不存在),
# 并清理 trae agent 注入的 LD_LIBRARY_PATH (否则 MP 的 g++ 预处理程序找不到
# 系统 libstdc++ 的 GLIBCXX_3.4.26).
set -u
REPO=/root/TR-SPMM_zzx
cd "$REPO" || exit 1
mkdir -p result
cp -f result/e2e_a100_n128_4way.csv result/e2e_a100_n128_4way.bak.csv 2>/dev/null || true

export PATH=/root/venv39/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export CUDA_VISIBLE_DEVICES=0
export DR_REAL_DIR=/dev/shm/real_matrices
export TR_FUSED=1
export DTC_PREP_TIMEOUT=1500
export DTC_REORDER_ID_DIR=$PWD/baseline/dtc/reordering/asplos_dtc_dataset
unset DTC_NO_REORDER

setsid nohup python gnn_e2e.py \
  --methods tr mp libra \
  --dimN 128 \
  --out result/e2e_a100_n128_4way.csv \
  > result/e2e_a100_n128_4way.log 2>&1 &
echo "PID=$!"
