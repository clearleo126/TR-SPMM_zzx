import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/spmm/result.csv')
df = df[df['hybrid'] < 100].reset_index(drop=True)
# ---------- 核心逻辑：计算提升比例并排序 ----------
# 避免除零报错，设置一个很小的 epsilon 防止 TCU Util 为 0
epsilon = 1e-5
df['improve_ratio'] = (df['hybrid'] - df['tcu']) / (df['tcu'] + epsilon)

# 选取提升比例最高的前 10 个矩阵
df_top10 = df.sort_values(by='improve_ratio', ascending=False).head(15).reset_index(drop=True)

# ---------- 绘图 ----------
x = np.arange(len(df_top10))
tcu_util = df_top10['tcu'].values
hybrid_util = df_top10['hybrid'].values - tcu_util


fig, ax = plt.subplots(figsize=(7, 2.5))

bar1 = ax.bar(x, tcu_util, color='orange', label='TCU-only',width=0.45)
bar2 = ax.bar(x, hybrid_util, bottom=tcu_util, color='orangered', label='Hybrid Computation',width=0.45)

ax.set_xticks(x)
ax.set_xticklabels(df_top10['dataSet'], rotation=35, ha='right', fontsize=9)
ax.set_ylabel('Utilization (%)', fontsize=12)
# ax.set_title('Top 10 Matrices by Hybrid vs. TCU Utilization Gain', fontsize=14)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.5)
# ax.set_ylim(0, 100)  # 强制纵坐标最大为 100
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/spmm/spmm_top10.png', dpi=800)
plt.show()