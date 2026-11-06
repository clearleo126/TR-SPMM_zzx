#!/usr/bin/env bash
# 端到端四方对比 (RTX 3090, GPU2), dimN=128, 11 矩阵
#   phase A : tr + mp           -> result/e2e_4way_n128_trmp.csv   (已完成)
#   phase B1: dtc 全部 11 矩阵  -> result/e2e_4way_n128_dtclibra.csv
#   phase B2: libra 仅补旧表缺的 4 个矩阵 (单独限时, 避免拖垮排队)
# 逐行即时落盘 (gnn_e2e.py --append), 被杀不丢已完成行.
# DTC 预处理超时 3600s (与论文协议一致), 超时由 backend 抛 TimeoutError 记 SKIP.
set -u
cd /home/zhangzhixuan/DR_SPMM_ABL
PY=/home/zhangzhixuan/miniconda3/envs/libra/bin/python
export CUDA_VISIBLE_DEVICES=2
export DTC_PREP_TIMEOUT=3600
MATS="ddi DD ogbn_arxiv web-BerkStan Yeast OVCAR-8H igb_small YeastH protein ogbn_products reddit"
OUTB=result/e2e_4way_n128_dtclibra.csv

for M in $MATS; do
  echo "=== B1 dtc: $M $(date) ==="
  timeout 4200 $PY gnn_e2e.py --matrices "$M" --methods dtc --reps 1 --append --out $OUTB
  echo "=== B1 $M rc=$? $(date) ==="
done

for M in ogbn_arxiv ogbn_products igb_small reddit; do
  echo "=== B2 libra: $M $(date) ==="
  timeout 2400 $PY gnn_e2e.py --matrices "$M" --methods libra --reps 1 --append --out $OUTB
  echo "=== B2 $M rc=$? $(date) ==="
done
echo "=== ALL DONE $(date) ==="
