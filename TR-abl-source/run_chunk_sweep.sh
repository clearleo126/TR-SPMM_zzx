#!/usr/bin/env bash
# ============================================================================
# run_chunk_sweep.sh — [E] chunk 尺寸扫描实验
#
# 目的: E 优化 (窗口内分块) 的净收益 = straggler 均衡收益 − 跨 chunk 原子
#       写回开销. chunk 大小是核心参数, 本脚本扫描固定 chunk ∈ {16,32,64,
#       128,256} 与 auto (现状自适应) / whole (整窗单 CTA, 即 no-E) 对比.
#
# 用法:
#   bash run_chunk_sweep.sh [MATRIX_DIR] [DIMN] [EPOCHES]
#   默认: MATRIX_DIR=/root/autodl-tmp/real_matrices, DIMN=128, EPOCHES=50
#
# 输出:
#   log/    res_chunk_<cfg>_dimN${DIMN}.log
#   result/ results_chunk_<cfg>_dimN${DIMN}.csv
#   result/ chunk_sweep_dimN${DIMN}.csv  (汇总: 各矩阵 × 各配置耗时与倍数)
# ============================================================================
set -u
cd "$(dirname "$0")"

MATRIX_DIR=${1:-/root/autodl-tmp/real_matrices}
DIMN=${2:-128}
EPOCHES=${3:-50}

# 配置: 名称:chunk 值
#   auto=0 自适应 (现状 baseline 逻辑: 均窗<32→32, >512→128, 其余 64)
#   whole=-1 整窗单 CTA (每窗口一个 CTA, 无跨 chunk 原子写)
#   16/32/64/128/256 固定 chunk
CONFIGS=(
  "auto:0"
  "whole:-1"
  "c16:16"
  "c32:32"
  "c64:64"
  "c128:128"
  "c256:256"
)

mkdir -p log result

echo "======================================================================"
echo " [E] chunk 尺寸扫描: dir=${MATRIX_DIR} dimN=${DIMN} epoches=${EPOCHES}"
echo " 配置: ${CONFIGS[*]}"
echo "======================================================================"

for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%:*}"
  chunk="${cfg##*:}"
  log="log/res_chunk_${name}_dimN${DIMN}.log"
  csv="result/results_chunk_${name}_dimN${DIMN}.csv"

  echo ">>> [chunk=${name}] 开始: $(date '+%F %T')"
  python spmm.py --batch-dir "${MATRIX_DIR}" \
                 --dimN "${DIMN}" \
                 --epoches "${EPOCHES}" \
                 --ablation 0 \
                 --chunk "${chunk}" \
                 --no-verify \
                 --csv "${csv}" \
                 > "${log}" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo ">>> [chunk=${name}] FAILED (exit=$rc), 详见 ${log}"
  else
    echo ">>> [chunk=${name}] DONE: $(date '+%F %T')"
  fi
done

# ---- 汇总 ----
python - "$DIMN" <<'PYEOF'
import glob, os, sys
import numpy as np
import pandas as pd

dimn = sys.argv[1]
cfg_order = ["auto", "whole", "c16", "c32", "c64", "c128", "c256"]
data = {}
for f in sorted(glob.glob(f"result/results_chunk_*_dimN{dimn}.csv")):
    cfg = os.path.basename(f).replace("results_chunk_", "").replace(f"_dimN{dimn}.csv", "")
    data[cfg] = pd.read_csv(f).set_index("matrix_name").elapsed_ms

if not data:
    print("无结果可汇总"); sys.exit(0)

idx = data["auto"].index if "auto" in data else data[list(data)[0]].index
out = pd.DataFrame(index=idx)
for cfg in cfg_order:
    if cfg in data:
        out[cfg] = data[cfg].reindex(idx)
        if "auto" in data:
            out[f"{cfg}/auto"] = (data[cfg].reindex(idx) / data["auto"].reindex(idx)).round(3)

path = f"result/chunk_sweep_dimN{dimn}.csv"
out.to_csv(path, float_format="%.4f")
print(f"\n汇总已写入: {path}")

# 每矩阵最优 chunk
if "auto" in data:
    print("\n== 每矩阵最优 chunk (相对 auto) ==")
    for m in idx:
        ratios = {c: out.loc[m, f"{c}/auto"] for c in cfg_order
                  if c != "auto" and f"{c}/auto" in out.columns and pd.notna(out.loc[m, f"{c}/auto"])}
        best = min(ratios, key=ratios.get)
        print(f"  {m:<16} best={best:<5} ({ratios[best]:.3f}x vs auto)")
PYEOF

echo ""
echo "======================================================================"
echo "扫描完成 $(date '+%F %T')"
echo "======================================================================"
