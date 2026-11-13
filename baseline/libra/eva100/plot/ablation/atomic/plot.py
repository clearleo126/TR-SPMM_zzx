import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/atomic/filter_h100_spmm_fp16_result_new_0217_128_overhead.csv')
df_pos = df[df['percent'] > 0]

# 2. 取 percent 最小的 20 个
df_filtered = df_pos.nsmallest(20, 'percent')

# print(f"最大: {round(df_filtered['percent'].max(), 4)}")
print(f"算术平均: {round(df_filtered['percent'].mean(), 4)}")
print(f"几何平均提升比: {round(gmean(df_filtered['percent']), 4)}")


# 选取提升比例最高的前 10 个矩阵
df_top10 = df_filtered.sort_values(by='percent', ascending=True).head(10).reset_index(drop=True)

# ---------- 绘图 ----------
x = np.arange(len(df_top10))
atomic = df_top10['percent'].values

fig, ax = plt.subplots(figsize=(4, 2))

# bar1 = ax.bar(x, hybrid_access,  color='green', label='Hybrid Computation',width=0.45, edgecolor='black')
# bar2 = ax.bar(x, cuda_access, bottom=hybrid_access, color='lightgreen', label='CUDA core-only',width=0.45, edgecolor='black')

bar1 = ax.bar(x, atomic,  color='orchid',width=0.45, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(df_top10['dataSet'], rotation=35, ha='right', fontsize=9)
# ax.set_ylabel('Memory (MB)', fontsize=12)
# ax.set_title('Top 10 Matrices by Hybrid vs. TCU Utilization Gain', fontsize=14)
# ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/atomic/atomic.png', dpi=800)
plt.show()