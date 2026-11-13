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
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/h100_spmm_fp16_result_128.csv')
df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n32.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）


dataset = [ 'IGB_small', 'reddit', 'amazon']
baseline = ['DGL-fp16', 'DGL-tf32', 'TC-GNN', 'GNNAdvisor', 'PyG']

# libra_fp16_128 = [14.2035, 8.922, 4.458]
# libra_tf32_128 = [16.6063, 9.1434, 4.7825]
# tcgnn_128 =[12.1785, 700, 188.1937]
# dgl_128= [10.9531, 10.9531, 4.1267]
# pyg_128= [24.6922, 100000, 13.3843]
# advisor_128 = [10.004, 13.2158, 3.9526]

libra_fp16_128 = [15.0393, 9.1876, 5.6129]
libra_tf32_128 = [20.9806, 13.91, 6.7284]
tcgnn_128 =[21.5388, 700, 191.2491]
dgl_128 = [21.1727, 17.5586, 7.6347]
pyg_128 = [68.4372, 100000, 39.2397]
advisor_128 = [21.2809, 25.9637, 7.2208]



libra_fp16_256 = [25.4393, 16.6223, 9.0486]
libra_tf32_256 = [37.8102, 29.3845, 12.2252]
tcgnn_256 =[37.3408, 700, 136.0393]
dgl_256 = [33.9075, 26.7331, 12.2775]
pyg_256 = [100000, 100000, 100000]
advisor_256 = [38.6019, 45.6943, 13.0477]





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
    bar1 = ax.bar(ind - 2*width, temp, width, label='Libra-FP16', color='cornflowerblue', edgecolor='black', linewidth=1)
    
    #绘制Libra-TF32
    temp = []
    temp.append(libra_tf32_speedup[cur])
    temp.append(libra_tf32_speedup_256[cur])
    bar1 = ax.bar(ind - width, temp, width, label='Libra-TF32', color='lightskyblue', edgecolor='black', linewidth=1)
    
    #绘制TC-GNN
    temp = []
    temp.append(tcgnn_speedup[cur])
    temp.append(tcgnn_speedup_256[cur])
    bar1 = ax.bar(ind, temp, width, label='TC-GNN',  color='lightcoral', edgecolor='black', linewidth=1)
    
    #绘制Advisor
    temp = []
    temp.append(advisor_speedup[cur])
    temp.append(advisor_speedup_256[cur])
    bar1 = ax.bar(ind + width, temp, width, label='GNNAdvisor', hatch=patterns[3], color='lightgreen', edgecolor='black', linewidth=1)
    
    #绘制PyG
    temp = []
    temp.append(pyg_speedup[cur])
    temp.append(pyg_speedup_256[cur])
    bar1 = ax.bar(ind + 2*width, temp, width, label='PyG', color='lightyellow', edgecolor='black', linewidth=1)

    ax.xaxis.set_visible(False)
    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/4090-gcn-' + data +'.png', dpi=800)
    cur += 1
    # 清空图形
    plt.clf()
    
    
#fp16 加速比
print("fp16:")
print(libra_fp16_speedup)
print("dgl ", stats.gmean(libra_fp16_speedup))
print("dl: ", max(libra_fp16_speedup))
print()

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