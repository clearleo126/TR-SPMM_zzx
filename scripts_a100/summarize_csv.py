#!/usr/bin/env python3
"""只读汇总: 不改 CSV, 生成每家成功/失败计数与 e2e_s 表."""
import csv
import sys
from collections import defaultdict

CSV = sys.argv[1] if len(sys.argv) > 1 else "result/e2e_a100_n128_4way.csv"
rows = list(csv.DictReader(open(CSV)))
methods = ["tr", "mp", "libra", "dtc"]
print(f"rows={len(rows)}")
for m in methods:
    rs = [r for r in rows if r["method"] == m]
    if not rs:
        continue
    ok = sum(1 for r in rs if r.get("status") == "ok")
    to = sum(1 for r in rs if r.get("status") == "timeout")
    fa = sum(1 for r in rs if r.get("status") == "fail")
    print(f"{m:6s} total={len(rs):2d} ok={ok:2d} timeout={to:2d} fail={fa:2d}")
print("\n dataset            method   e2e_s     status")
for r in sorted(rows, key=lambda x: (x["dataset"], methods.index(x["method"]))):
    e2e = r["e2e_s"]
    e2e = f"{float(e2e):.3f}" if e2e else "-"
    print(f" {r['dataset']:18s} {r['method']:6s} {e2e:>9s} {r.get('status','')}")
