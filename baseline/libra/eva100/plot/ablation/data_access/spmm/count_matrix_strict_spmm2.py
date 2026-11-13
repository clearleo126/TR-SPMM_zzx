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


df1 = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/data_access/spmm/result1.csv')


cuda_sp = []
libra_sp = []
num_edges = []

for index, row in df1.iterrows():
    
    ratio = (row['cuda'] - row['hybrid'])/row['cuda']
    libra_sp.append(round(ratio,2))
    num_edges.append(int(row['num_edges']))

sorted_indices = sorted(range(len(libra_sp)), key=lambda k: libra_sp[k]) 
num_edges = [num_edges[i] for i in sorted_indices]
libra_sp = [libra_sp[i] for i in sorted_indices]

num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))
    
fig, ax = plt.subplots(figsize=(8, 4))

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
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/data_access/spmm.png', dpi=800)
# 清空图形
plt.clf()