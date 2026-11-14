#!/usr/bin/env python3
"""把旧 A100 结果里的 4 个 DTC timeout 行并入本次结果 (只搬 status, 不搬数字).

本次 DTC 以 1500s 上限跑时, web-BerkStan/protein/ogbn_products/reddit 会在
CPU TCA 重排上超时。这 4 行在旧结果里也是 timeout 且无任何指标数字, 因此按用户
要求把这 4 行以 status=timeout 追加到新 CSV, 保证 44 行完整。

用法:
  python merge_old_dtc_timeouts.py <old.csv> <new.csv>
"""
import csv
import sys

HEADER = ["dataset", "rows", "cols", "nnz", "method", "dimN", "load_s",
          "prep_s", "cold_ms", "hot_ms", "chain_2call_ms", "kernel_ms",
          "e2e_s", "e2e_incl_load_s", "status"]
NEED = ["web-BerkStan", "protein", "ogbn_products", "reddit"]

old_p = sys.argv[1]
new_p = sys.argv[2]
old = {(r["dataset"], r["method"]): r for r in csv.DictReader(open(old_p))}
rows = list(csv.DictReader(open(new_p)))
have = {(r["dataset"], r["method"]) for r in rows}
for n in NEED:
    if (n, "dtc") in have:
        print(f"skip {n}: already present")
        continue
    o = old[(n, "dtc")]
    assert o["status"] == "timeout", (n, o["status"])
    assert not o["prep_s"] and not o["cold_ms"], (n, "old row carries numbers")
    rows.append({"dataset": n, "rows": o["rows"], "cols": o["cols"],
                 "nnz": o["nnz"], "method": "dtc", "dimN": "128",
                 "load_s": "", "prep_s": "", "cold_ms": "", "hot_ms": "",
                 "chain_2call_ms": "", "kernel_ms": "", "e2e_s": "",
                 "e2e_incl_load_s": "", "status": "timeout"})
with open(new_p, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=HEADER)
    w.writeheader()
    w.writerows(rows)
print(f"rows now {len(rows)}")
