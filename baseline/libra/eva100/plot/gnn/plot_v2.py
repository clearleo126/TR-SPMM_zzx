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



dataset = [ 'IGB_small', 'reddit', 'amazon']
baseline = ['DGL-fp16', 'DGL-tf32', 'TC-GNN', 'GNNAdvisor', 'PyG']

# libra_fp16_128 = [14.2035, 8.922, 4.458]
# libra_tf32_128 = [16.6063, 9.1434, 4.7825]
# tcgnn_128 =[20.0014, 700, 286.5995]
# dgl_128 = [17.4165, 14.7682, 6.0157]
# pyg_128 = [28.0065, 100000, 13.4948]
# advisor_128 = [21.019, 24.08, 6.4492]

libra_fp16_128 = [17.3301, 13.1667, 6.8544]
libra_tf32_128 = [22.4692, 15.0057, 7.7372]
tcgnn_128 =[25.9081, 700, 297.7326]
dgl_128 = [23.1234, 16.2014, 7.8822]
pyg_128 = [61.3963, 100000, 34.9828]
advisor_128 = [24.5883, 33.0699, 8.6901]


libra_fp16_256 = [23.0146, 23.0164, 9.6939]
libra_tf32_256 = [32.9706, 31.0262, 12.0279]
tcgnn_256 =[39.5111, 700, 209.7029]
dgl_256 = [31.7599, 23.0827, 11.0496]
pyg_256 = [124.0033, 100000, 76.8821]
advisor_256 = [35.5332, 50.9425, 12.8341]


libra_fp16_speedup= []
libra_tf32_speedup= []
tcgnn_speedup = []
pyg_speedup = []
advisor_speedup = []

libra_fp16_speedup_256 = []
libra_tf32_speedup_256 = []
tcgnn_speedup_256 = []
pyg_speedup_256 = []
advisor_speedup_256 = []

i=0
for dgl_sub in dgl_128:
    libra_fp16_speedup.append(round((dgl_sub/libra_fp16_128[i]),4))
    libra_tf32_speedup.append(round((dgl_sub/libra_tf32_128[i]),4))
    tcgnn_speedup.append(round((dgl_sub/tcgnn_128[i]),4))
    pyg_speedup.append(round((dgl_sub/pyg_128[i]),4))
    advisor_speedup.append(round((dgl_sub/advisor_128[i]),4))
    i+=1
    
i=0
for dgl_sub in dgl_256:
    libra_fp16_speedup_256.append(round((dgl_sub/libra_fp16_256[i]),4))
    libra_tf32_speedup_256.append(round((dgl_sub/libra_tf32_256[i]),4))
    tcgnn_speedup_256.append(round((dgl_sub/tcgnn_256[i]),4))
    pyg_speedup_256.append(round((dgl_sub/pyg_256[i]),4))
    advisor_speedup_256.append(round((dgl_sub/advisor_256[i]),4))
    i+=1


cur =0
#循环6次，每个数据集一个图，创建6个图
for data in dataset:
    ind = np.arange(2)  # 柱状图的 x 坐标位置
    width = 0.16  # 柱状图的宽度

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(4, 2))

    # 每组柱状图的纹理样式
    patterns = ['/', 'x', '-', '\\', '|-']

    #绘制Libra-FP16
    temp = []
    temp.append(libra_fp16_speedup[cur])
    temp.append(libra_fp16_speedup_256[cur])
    bar1 = ax.bar(ind - 2*width, temp, width, label='Libra-FP16',  color='cornflowerblue', edgecolor='black', linewidth=1)
    
    #绘制Libra-TF32
    temp = []
    temp.append(libra_tf32_speedup[cur])
    temp.append(libra_tf32_speedup_256[cur])
    bar1 = ax.bar(ind - width, temp, width, label='Libra-TF32',  color='lightskyblue', edgecolor='black', linewidth=1)
    
    #绘制TC-GNN
    temp = []
    temp.append(tcgnn_speedup[cur])
    temp.append(tcgnn_speedup_256[cur])
    bar1 = ax.bar(ind, temp, width, label='TC-GNN',  color='lightcoral', edgecolor='black', linewidth=1)
    
    #绘制Advisor
    temp = []
    temp.append(advisor_speedup[cur])
    temp.append(advisor_speedup_256[cur])
    bar1 = ax.bar(ind + width, temp, width, label='GNNAdvisor', color='lightgreen', edgecolor='black', linewidth=1)
    
    #绘制PyG
    temp = []
    temp.append(pyg_speedup[cur])
    temp.append(pyg_speedup_256[cur])
    bar1 = ax.bar(ind + 2*width, temp, width, label='PyG', color='lightyellow', edgecolor='black', linewidth=1)

    ax.xaxis.set_visible(False)
    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/h100-gcn-' + data +'.png', dpi=800)
    cur += 1
    # 清空图形
    plt.clf()
    

#fp16 加速比
print("fp16:")
print(libra_fp16_speedup)
print("dgl ", stats.gmean(libra_fp16_speedup))
print("dl: ", max(libra_fp16_speedup))
print()
  
# #fp16 加速比
# print("fp16:")
# result_tcgnn = [a / b for a in tcgnn_128 for b in libra_fp16_128]
# result_dgl = [a / b for a in dgl_128 for b in libra_fp16_128]
# result_pyg = [a / b for a in pyg_128 for b in libra_fp16_128]
# print("tcgnn: ", stats.gmean(result_tcgnn))
# print("dgl ", stats.gmean(result_dgl))
# print("pyg: ", stats.gmean(result_pyg))

# print("tcgnn: ", max(result_tcgnn))
# print("dgl ", max(result_dgl))
# print("pyg: ", max(result_pyg))
# print()

# print('tf32:')
# result_tcgnn = [a / b for a in tcgnn_128 for b in libra_tf32_128]
# result_dgl = [a / b for a in dgl_128 for b in libra_tf32_128]
# result_pyg = [a / b for a in pyg_128 for b in libra_tf32_128]
# print("tcgnn: ", stats.gmean(result_tcgnn))
# print("dgl ", stats.gmean(result_dgl))
# print("pyg: ", stats.gmean(result_pyg))

# print("tcgnn: ", max(result_tcgnn))
# print("dgl ", max(result_dgl))
# print("pyg: ", max(result_pyg))