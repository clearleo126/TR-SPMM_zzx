import matplotlib.pyplot as plt
import numpy as np

# ===== 数据 =====
data = 'mpi1'
x = [8, 16, 32, 64, 128]   # 横坐标：threshold group size
size_tc = [0.2231, 0.2119, 0.2062, 0.2085, 0.2191]
size_cc = [0.2112, 0.2081, 0.2072, 0.2092, 0.2122]

# ===== 画图 =====
fig, ax = plt.subplots(figsize=(5, 2))

ax.plot(
    x, size_tc,
    marker='o', linewidth=2.0, markersize=5,
    label='Group size: Ts'
)

ax.plot(
    x, size_cc,
    marker='s', linewidth=2.0, markersize=5,
    label='Group size: Cs'
)

# ===== 坐标轴 =====
# ax.set_xlabel('Threshold Group Size Index', fontsize=11)
# ax.set_ylabel('Normalized Runtime Overhead', fontsize=11)
ax.set_xticks(x)

# y 轴留一点 margin，避免线贴边
ymin = min(size_tc + size_cc)
ymax = max(size_tc + size_cc)
ax.set_ylim(ymin * 0.98, ymax * 1.02)

# ===== 网格 =====
ax.grid(axis='y', linestyle='--', alpha=0.5)

# # ===== 图例 =====
# ax.legend(
#     loc='best',
#     fontsize=10,
#     frameon=False
# )


ax.grid(axis='y', linestyle='--', alpha=0.5)

# ===== 保存 =====
plt.tight_layout()
plt.savefig(
    '/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/atomic/size.png',
    dpi=800
)
plt.show()