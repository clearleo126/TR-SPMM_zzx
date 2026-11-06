#!/usr/bin/env bash
# 钉死口径的端到端复跑 (论文 fig:end-to-end 用)
#   - e2e = prep + cold + hot (不含 matrix load, 见 gnn_e2e.py 头注释)
#   - 同一进程内 rep -> N -> method 交错, 3 个 N 同机配对
#   - reps=2 取中位(两者取较慢), MP 固定 adjacent + 官方五段预处理现场复现
#   - protein/reddit 的 Libra 预处理 15~30 min/次, 降为 1 rep (其余 2 rep)
set -u
cd /home/zhangzhixuan/DR_SPMM_ABL
PY=/home/zhangzhixuan/miniconda3/envs/libra/bin/python
export CUDA_VISIBLE_DEVICES=2
export DR_MP_PREP=official
export DR_MP_MATCHING=adjacent
OUT=result/e2e_pinned_n32_128_256.csv

# phase 1: tr + mp（全 8 矩阵, 3 个 N, 2 reps; MP 官方只到 N=128）
for M in YeastH OVCAR-8H Yeast DD web-BerkStan reddit ddi protein; do
  echo "=== P1 $M $(date) ==="
  timeout 5400 $PY gnn_e2e.py --matrices "$M" --methods tr mp --dimN 32 128 --reps 2 --append --out $OUT
  echo "=== P1 $M rc=$? $(date) ==="
done

# phase 2: libra 便宜矩阵（3 个 N, 2 reps）
for M in DD OVCAR-8H Yeast YeastH web-BerkStan ddi; do
  echo "=== P2 $M $(date) ==="
  timeout 3600 $PY gnn_e2e.py --matrices "$M" --methods libra --dimN 32 128 256 --reps 2 --append --out $OUT
  echo "=== P2 $M rc=$? $(date) ==="
done

# phase 3: libra 大图（预处理 15~30 min/次, 只跑 1 rep）
for M in protein reddit; do
  echo "=== P3 $M $(date) ==="
  timeout 7200 $PY gnn_e2e.py --matrices "$M" --methods libra --dimN 32 128 256 --reps 1 --append --out $OUT
  echo "=== P3 $M rc=$? $(date) ==="
done
echo "=== PINNED ALL DONE $(date) ==="
