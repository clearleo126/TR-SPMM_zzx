#!/usr/bin/env bash
# ============================================================================
# run_ablation.sh — DR-SPMM 消融实验 (A~I + 双流) 一键脚本
#
# 用法:
#   bash run_ablation.sh [MATRIX_DIR] [DIMN] [EPOCHES]
#   默认: MATRIX_DIR=/root/autodl-tmp/real_matrices, DIMN=128, EPOCHES=50
#
# 输出:
#   log/  res_abl*_dimN${DIMN}.log      每个配置的完整运行日志
#   result/ results_abl*_dimN${DIMN}.csv 每个配置的耗时表
#   result/ ablation_summary_dimN${DIMN}.csv  汇总对比表 (analyze 阶段生成)
#
# 说明:
#   - 顺序执行, 避免 GPU 争用导致计时失真
#   - 每个配置运行前打印配置名, 结束打印 DONE_<NAME>
#   - 任一配置失败会记录 FAILED_<NAME> 并继续跑后面的配置
# ============================================================================
set -u
cd "$(dirname "$0")"

MATRIX_DIR=${1:-/root/autodl-tmp/real_matrices}
DIMN=${2:-128}
EPOCHES=${3:-50}

# 消融配置: 名称:编号
CONFIGS=(
  "baseline:0"
  "no_A:1"
  "no_B:2"
  "no_C:3"
  "no_D:4"
  "no_E:5"
  "no_F:6"
  "no_G:7"
  "no_H:8"
  "no_I:9"
  "no_streams:10"
)

mkdir -p log result

echo "======================================================================"
echo " DR-SPMM 消融实验: matrix_dir=${MATRIX_DIR} dimN=${DIMN} epoches=${EPOCHES}"
echo " 共 ${#CONFIGS[@]} 个配置"
echo "======================================================================"

for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%:*}"
  abl="${cfg##*:}"
  log="log/res_abl_${name}_dimN${DIMN}.log"
  csv="result/results_abl_${name}_dimN${DIMN}.csv"

  echo ""
  echo ">>> [$name] ablation=${abl} 开始: $(date '+%F %T')"
  python spmm.py --batch-dir "${MATRIX_DIR}" \
                 --dimN "${DIMN}" \
                 --epoches "${EPOCHES}" \
                 --ablation "${abl}" \
                 --no-verify \
                 --csv "${csv}" \
                 > "${log}" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo ">>> [$name] FAILED (exit=$rc), 详见 ${log}"
  else
    echo ">>> [$name] DONE: $(date '+%F %T')  csv=${csv}"
  fi
done

# ---- 汇总表 ----
python - "$DIMN" <<'PYEOF'
import glob, os, sys
import pandas as pd

dimn = sys.argv[1]
rows = {}
for f in sorted(glob.glob(f"result/results_abl_*_dimN{dimn}.csv")):
    cfg = os.path.basename(f).replace("results_abl_", "").replace(f"_dimN{dimn}.csv", "")
    df = pd.read_csv(f)
    for _, r in df.iterrows():
        rows.setdefault(r["matrix_name"], {})[cfg] = r["elapsed_ms"]

if not rows:
    print("无结果可汇总")
    sys.exit(0)

order = ["baseline","no_A","no_B","no_C","no_D","no_E","no_F","no_G","no_H","no_I","no_streams"]
out = pd.DataFrame(rows).T
out = out.reindex(columns=[c for c in order if c in out.columns])
# 相对 baseline 的倍数列
for c in out.columns:
    if c != "baseline":
        out[f"{c}_vs_base"] = (out[c] / out["baseline"]).round(3)
path = f"result/ablation_summary_dimN{dimn}.csv"
out.to_csv(path, float_format="%.4f")
print(f"\n汇总已写入: {path}")
print(out.to_string(float_format=lambda x: f"{x:.4f}"))
PYEOF

echo ""
echo "======================================================================"
echo "全部完成 $(date '+%F %T')"
echo "======================================================================"
