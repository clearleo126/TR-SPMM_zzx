import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/data_access/spmm/result1.csv')
df[['cuda', 'hybrid']] = df[['cuda', 'hybrid']] / 1_000_000
# ---------- 核心逻辑：计算提升比例并排序 ----------
# 避免除零报错，设置一个很小的 epsilon 防止 TCU Util 为 0
epsilon = 1e-5
df['improve_ratio'] = df['cuda'] / (df['hybrid'] + epsilon)
df['improve_ratio1'] = (df['cuda'] - df['hybrid']) / (df['cuda'] + epsilon)

print(f"\n最大提升比: {round(df['improve_ratio1'].max(), 4)}")
print(f"算术平均提升比: {round(df['improve_ratio1'].mean(), 4)}")
print(f"几何平均提升比: {round(gmean(df['improve_ratio1'][df['improve_ratio1'] > 0] + 1) - 1, 4)}")
print(f"几何平均提升比: {round(gmean(df['improve_ratio1']), 4)}")


# 选取提升比例最高的前 10 个矩阵
df_top10 = df.sort_values(by='improve_ratio', ascending=False).head(10).reset_index(drop=True)

# ---------- 绘图 ----------
x = np.arange(len(df_top10))
hybrid_access = df_top10['hybrid'].values*2
cuda_access = df_top10['cuda'].values*2 - hybrid_access*2

fig, ax = plt.subplots(figsize=(4, 2))

# bar1 = ax.bar(x, hybrid_access,  color='green', label='Hybrid Computation',width=0.45, edgecolor='black')
# bar2 = ax.bar(x, cuda_access, bottom=hybrid_access, color='lightgreen', label='CUDA core-only',width=0.45, edgecolor='black')

bar1 = ax.bar(x, hybrid_access,  color='green',width=0.45, edgecolor='black')
bar2 = ax.bar(x, cuda_access, bottom=hybrid_access, color='lightgreen', width=0.45, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(df_top10['dataSet'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Memory (MB)', fontsize=12)
# ax.set_title('Top 10 Matrices by Hybrid vs. TCU Utilization Gain', fontsize=14)
# ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/data_access/spmm/spmm_top6.png', dpi=800)
plt.show()