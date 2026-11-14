#!/bin/bash
# 现场生成 DTC 官方 TCA 重排缓存 (libMHCUDA + CUDA), 生成阶段上限 3000s,
# 记录实际墙钟 -> <name>.reorder_seconds.txt; 然后跑 DTC 端到端 (1500s 上限).
# 全部由脚本记录到 result/reorder_gen_status.txt, 无需人工监视.
set -u
REPO=/root/TR-SPMM_zzx
RD=$REPO/baseline/dtc/reordering
ST=$REPO/result/reorder_gen_status.txt
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64
export TCA_GPU_MAX_ROWS=4000000
export PATH=/root/venv39/bin:$PATH

echo "=== reorder gen start $(date -Is) ===" >> "$ST"
cd "$RD" || exit 1
for n in DD ogbn_arxiv Yeast OVCAR-8H igb_small YeastH; do
  echo "$(date -Is) $n start" >> "$ST"
  t0=$(date +%s)
  timeout 3000 /root/venv39/bin/python TCA_reorder_gpu.py --dataset "$n" \
      > "$REPO/result/tca_gen_$n.log" 2>&1
  rc=$?
  t1=$(date +%s)
  s=$((t1 - t0))
  if [ "$rc" -eq 0 ] && [ -f "asplos_dtc_dataset/$n.reorder_id.npz" ]; then
    echo "$s" > "asplos_dtc_dataset/$n.reorder_seconds.txt"
    echo "$(date -Is) $n OK rc=$rc gen_s=$s" >> "$ST"
  else
    echo "$(date -Is) $n FAIL rc=$rc gen_s=$s (no artifact)" >> "$ST"
  fi
done

cd "$REPO" || exit 1
export CUDA_VISIBLE_DEVICES=0
export DR_REAL_DIR=/dev/shm/real_matrices
export TR_FUSED=1
export DTC_PREP_TIMEOUT=1500
export DTC_REORDER_ID_DIR=$RD/asplos_dtc_dataset
unset DTC_NO_REORDER
echo "$(date -Is) dtc e2e start" >> "$ST"
/root/venv39/bin/python gnn_e2e.py --methods dtc --dimN 128 --append \
  --out result/e2e_a100_n128_4way.csv > result/e2e_a100_n128_4way_dtc.log 2>&1
echo "$(date -Is) dtc e2e done rc=$?" >> "$ST"
/root/venv39/bin/python /root/summarize_csv.py result/e2e_a100_n128_4way.csv >> "$ST" 2>&1
echo "=== all done $(date -Is) ===" >> "$ST"
