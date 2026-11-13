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
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n128.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）


dataset = [ 'IGB_small', 'LastFM', 'twitter', 'MOOC', 'reddit', 'yeast']
baseline = ['DGL-fp16', 'DGL-tf32', 'TC-GNN', 'PyG']
# libra_fp16 = [15.8707, 18.4264, 7.4958, 8.3653, 13.9579, 26.2983]
# libra_tf32 = [19.2429, 20.2455, 9.056, 8.5135, 17.7885, 100000]
libra_fp16_32 = [18.7667, 20.8038,8.1727, 8.8427, 20.0664, 28.3901]
libra_tf32_32 = [22.1137, 22.4091, 10.292, 10.0799, 24.5673, 30.809]
tcgnn_32 =[36.9453, 25.1481, 9.8215, 322.7239, 700, 31.4962]
dgl_32 = [32.0777, 27.5137, 11.7637, 10.1547, 107.5903, 36.5581]
pyg_32 = [62.869, 30.3577, 16.9257, 9.249, 100000, 51.5764]


libra_fp16_128 = [29.8316, 25.4624, 10.7019, 8.9523, 51.5471, 38.0307]
libra_tf32_128 = [41.3659, 32.4935, 15.7395, 9.9359, 64.7263, 47.2839]
tcgnn_128 =[70.3981, 35.832, 14.1485, 362.7268, 10000, 44.4689]
dgl_128 = [53.8704,41.2508, 18.3224, 15.0385, 137.0785,58.4269]
pyg_128 = [10000, 54.9958, 39.937, 19.5126, 100000, 115.1821]

libra_fp16_speedup= []
libra_tf32_speedup= []
tcgnn_speedup = []
pyg_speedup = []

libra_fp16_speedup_128 = []
libra_tf32_speedup_128 = []
tcgnn_speedup_128 = []
pyg_speedup_128 = []
i=0
for dgl_sub in dgl_32:
    libra_fp16_speedup.append(round((dgl_sub/libra_fp16_32[i]),4))
    libra_tf32_speedup.append(round((dgl_sub/libra_tf32_32[i]),4))
    tcgnn_speedup.append(round((dgl_sub/tcgnn_32[i]),4))
    pyg_speedup.append(round((dgl_sub/pyg_32[i]),4))
    i+=1

i=0
for dgl_sub in dgl_128:
    libra_fp16_speedup_128.append(round((dgl_sub/libra_fp16_128[i]),4))
    libra_tf32_speedup_128.append(round((dgl_sub/libra_tf32_128[i]),4))
    tcgnn_speedup_128.append(round((dgl_sub/tcgnn_128[i]),4))
    pyg_speedup_128.append(round((dgl_sub/pyg_128[i]),4))
    i+=1
    
mycolor = {'DGL-fp16':'royalblue', 'TC-GNN':'limegreen', 
           'DGL-tf32':'cornflowerblue', 'PyG':'orchid'}
cur =0
#循环6次，每个数据集一个图，创建6个图
for data in dataset:
    res = {}
    res['Category'] = []
    res['Type'] = []
    res['Value'] = []
    for i in range(4):
        res['Category'].append(data)
    for i in range(4):
        res['Category'].append(data+'-1')
            
    for base in baseline:
        res['Type'].append(base)
    for base in baseline:
        res['Type'].append(base)

    res['Value'].append(libra_fp16_speedup[cur])
    res['Value'].append(libra_tf32_speedup[cur])
    res['Value'].append(tcgnn_speedup[cur])
    res['Value'].append(pyg_speedup[cur])

    res['Value'].append(libra_fp16_speedup_128[cur])
    res['Value'].append(libra_tf32_speedup_128[cur])
    res['Value'].append(tcgnn_speedup_128[cur])
    res['Value'].append(pyg_speedup_128[cur])
    
    # 创建DataFrame
    df = pd.DataFrame(res)
    sns.set_style("darkgrid")
    # 绘制多组柱状图
    plt.figure(figsize=(5, 3))
    sns.barplot(x='Category', y='Value', hue='Type', data=df,  palette=mycolor, linewidth=0.6, legend=False, gap=0, width=0.8)

    # 添加标题和标签
    # plt.title('Complex Multiple Group Barplot')
    plt.xlabel('Category')
    # plt.ylabel('Valu1111')

    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/agnn/agnn-' + data +'.png', dpi=800)
    cur += 1
    # 清空图形
    plt.clf()
    
    
#fp16 加速比
print("fp16:")
result_tcgnn = [a / b for a in tcgnn_32 for b in libra_fp16_32]
result_dgl = [a / b for a in dgl_32 for b in libra_fp16_32]
result_pyg = [a / b for a in pyg_32 for b in libra_fp16_32]
print("tcgnn: ", stats.gmean(result_tcgnn))
print("dgl ", stats.gmean(result_dgl))
print("pyg: ", stats.gmean(result_pyg))
print("tcgnn: ", max(result_tcgnn))
print("dgl ", max(result_dgl))
print("pyg: ", max(result_pyg))
print()
print("tf32:")
result_tcgnn = [a / b for a in tcgnn_32 for b in libra_tf32_32]
result_dgl = [a / b for a in dgl_32 for b in libra_tf32_32]
result_pyg = [a / b for a in pyg_32 for b in libra_tf32_32]
print("tcgnn: ", stats.gmean(result_tcgnn))
print("dgl ", stats.gmean(result_dgl))
print("pyg: ", stats.gmean(result_pyg))
print("tcgnn: ", max(result_tcgnn))
print("dgl ", max(result_dgl))
print("pyg: ", max(result_pyg))