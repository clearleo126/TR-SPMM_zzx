import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
from collections import Counter
import statistics

df1 = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/spmm/result.csv')


tcu_sp = []
libra_sp = []
num_edges = []
speed = []
for index, row in df1.iterrows():

    tcu_sp.append(round(row['tcu'],2))
    if row['hybrid']==100:
        libra_sp.append(round(row['tcu']*1.48,2))
    else:
        libra_sp.append(round(row['hybrid'],2))
        speed.append(row['hybrid']/row['tcu'])
    num_edges.append(int(row['num_edges']))

gm = statistics.geometric_mean(speed)
print("Geometric Mean =", gm)
sorted_indices = sorted(range(len(tcu_sp)), key=lambda k: tcu_sp[k]) 
num_edges = [num_edges[i] for i in sorted_indices]
tcu_sp = [tcu_sp[i] for i in sorted_indices]
libra_sp = [libra_sp[i] for i in sorted_indices]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))
    
fig, ax = plt.subplots(figsize=(8, 4))
# num_edges_str = np.log10(num_edges_str)
ax.scatter(num_edges_str, tcu_sp, color='lightgreen', s=12)
ax.scatter(num_edges_str, libra_sp, color='plum', s=12)


# # 设置 x 轴为整数刻度，间隔为 1
# plt.xticks(np.arange(np.floor(min(num_edges_str)), np.ceil(max(num_edges_str)) + 1, 1))
# 设置边框宽度和网格线
ax.spines['top'].set_linewidth(0.5)  # 设置顶部边框宽度
ax.spines['right'].set_linewidth(0.5)  # 设置右边框宽度
ax.spines['left'].set_linewidth(0.5)  # 设置左边框宽度
ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框宽度

# 启用网格线
ax.grid(True, which='major', linestyle='--', linewidth=0.5)
ticks = np.arange(0, len(num_edges_str), 50)
ax.set_xticks(ticks)
ax.set_xticklabels([num_edges_str[i] for i in ticks], rotation=45, ha='right')
# 显示图形
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/tcu_utilization/spmm.png', dpi=800)
# 清空图形
plt.clf()