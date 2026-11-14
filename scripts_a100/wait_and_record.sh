#!/bin/bash
# 由脚本自行记录进度与最终结果 (无需 agent 监视).
#   每 5 分钟: result/e2e_a100_n128_4way_progress.txt
#   进程结束: result/e2e_a100_n128_4way_status.txt
set -u
REPO=/root/TR-SPMM_zzx
PID="${1:?pid}"
PROG="$REPO/result/e2e_a100_n128_4way_progress.txt"
OUT="${2:-$REPO/result/e2e_a100_n128_4way_status.txt}"
CSV="$REPO/result/e2e_a100_n128_4way.csv"
echo "=== watch start $(date -Is) pid=$PID ===" >> "$PROG"
while kill -0 "$PID" 2>/dev/null; do
  n=$(( $(wc -l < "$CSV" 2>/dev/null || echo 0) - 1 ))
  cur=$(grep -E "^=== " "$REPO/result/e2e_a100_n128_4way.log" 2>/dev/null | tail -1)
  echo "$(date -Is) rows=$n $cur" >> "$PROG"
  sleep 300
done
{
  echo "=== finished $(date -Is) pid=$PID ==="
  /root/venv39/bin/python /root/summarize_csv.py "$CSV"
  echo "--- log tail ---"
  tail -40 "$REPO/result/e2e_a100_n128_4way.log"
} > "$OUT" 2>&1
echo "=== watch end $(date -Is) ===" >> "$PROG"
