#绘制Libra，cusaprse, sputnik, Rode的图
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)
import itertools


dataset = ['pubmed', 'Reddit', 'IGB_small']
baseline = ['DGL-fp16', 'PCGCN', 'TC-GNN', 'GNNAdvisor', 'PyG']



libra_fp16_128 = [15.0393, 9.1876, 5.6129, ]
libra_tf32_128 = [20.9806, 13.91, 6.7284,  ]
tcgnn_128 =[21.5388, 34, 80,               ]
dgl_128 = [21.1727, 17.5586, 7.6347,       ]
pyg_128 = [40.4372, 35, 39.2397,           ]
advisor_128 = [21.2809, 25.9637, 7.2208,   ]
pcgcn_128 =[100.9248, 50, 50.1456,         ]

libra_fp16_speedup= []
pcgcn_speedup = []
tcgnn_speedup = []
advisor_speedup = []
pyg_speedup = []

i=0
for dgl_sub in dgl_128:
    libra_fp16_speedup.append(round((dgl_sub/libra_fp16_128[i]),4))
    pcgcn_speedup.append(round((dgl_sub/pcgcn_128[i]),4))
    tcgnn_speedup.append(round((dgl_sub/tcgnn_128[i]),4))
    advisor_speedup.append(round((dgl_sub/advisor_128[i]),4))
    pyg_speedup.append(round((dgl_sub/pyg_128[i]),4))
    i+=1


#循环6次，每个数据集一个图，创建6个图

ind = np.arange(3)  # 柱状图的 x 坐标位置
width = 0.16  # 柱状图的宽度

# 绘制柱状图
fig, ax = plt.subplots(figsize=(5.5, 2.6))

# 每组柱状图的纹理样式
patterns = ['/', 'x', '-', '\\', '|-']

#绘制Libra-FP16
bar1 = ax.bar(ind - 2*width, libra_fp16_speedup, width, label='Libra', hatch=patterns[0], color='lightskyblue', edgecolor='black', linewidth=1)

#绘制advisor
bar1 = ax.bar(ind - width, advisor_speedup, width, label='advisor', hatch=patterns[1], color='lightcoral', edgecolor='black', linewidth=1)

#绘制TC-GNN
bar1 = ax.bar(ind, tcgnn_speedup, width, label='TC-GNN', hatch=patterns[2], color='moccasin', edgecolor='black', linewidth=1)

#绘制pyg
bar1 = ax.bar(ind + width, pyg_speedup, width, label='pyg', hatch=patterns[3], color='lightyellow', edgecolor='black', linewidth=1)

#绘制PCGCN
bar1 = ax.bar(ind + 2*width, pcgcn_speedup, width, label='PCGCN', hatch=patterns[4], color='lightgreen', edgecolor='black', linewidth=1)

# 设置边框宽度和网格线
ax.spines['top'].set_linewidth(0.5)  # 设置顶部边框宽度
ax.spines['right'].set_linewidth(0.5)  # 设置右边框宽度
ax.spines['left'].set_linewidth(0.5)  # 设置左边框宽度
ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框宽度
ax.axhline(y=1, color='dimgray', linestyle='--', linewidth=1.5)
# 启用网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/h100-gcn-new' +'.png', dpi=800)

# 清空图形
plt.clf()
    
    
#fp16 加速比
print("fp16:")
print(libra_fp16_speedup)
print("dgl ", stats.gmean(libra_fp16_speedup))
print("max: ", max(libra_fp16_speedup))
print()
