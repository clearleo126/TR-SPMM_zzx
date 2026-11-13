import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/sddmm/result.csv')
df = df[df['hybrid'] < 100].reset_index(drop=True)
# ---------- 核心逻辑：计算提升比例并排序 ----------
# 避免除零报错，设置一个很小的 epsilon 防止 TCU Util 为 0
epsilon = 1e-5
df['improve_ratio'] = (df['hybrid'] - df['tcu']) / (df['tcu'] + epsilon)
df['improve_ratio1'] = (df['hybrid']) / (df['tcu'] + epsilon)

print(f"\n最大提升比: {round(df['improve_ratio1'].max(), 4)}")
print(f"算术平均提升比: {round(df['improve_ratio1'].mean(), 4)}")
print(f"几何平均提升比: {round(gmean(df['improve_ratio1'][df['improve_ratio1'] > 0] + 1) - 1, 4)}")

# 选取提升比例最高的前 10 个矩阵
df_top10 = df.sort_values(by='improve_ratio', ascending=False).head(10).reset_index(drop=True)

# ---------- 绘图 ----------
x = np.arange(len(df_top10))
tcu_util = df_top10['tcu'].values
hybrid_util = df_top10['hybrid'].values - tcu_util

fig, ax = plt.subplots(figsize=(4, 2))

# bar1 = ax.bar(x, tcu_util, color='orange', label='TCU-only',width=0.45, edgecolor='black')
# bar2 = ax.bar(x, hybrid_util, bottom=tcu_util, color='orangered', label='Hybrid Computation',width=0.45, edgecolor='black')

bar1 = ax.bar(x, tcu_util, color='orange',width=0.45, edgecolor='black')
bar2 = ax.bar(x, hybrid_util, bottom=tcu_util, color='orangered', width=0.45, edgecolor='black')

ax.set_xticks(x)
ax.set_xticklabels(df_top10['dataSet'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Utilization (%)', fontsize=12)
# ax.set_title('Top 10 Matrices by Hybrid vs. TCU Utilization Gain', fontsize=14)
# ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
# ax.set_ylim(0, 100)  # 强制纵坐标最大为 100
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/sddmm/sddmm_top6.png', dpi=800)
plt.show()