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
libra_fp16_128 = [17.7692, 20.5478, 7.8526, 5.8638, 13.3562, 29.8782]
libra_tf32_128 = [25.0478, 26.0406, 11.3981, 6.9627, 15.9122, 36.3158]
tcgnn_128 =[28.4862, 27.3911, 11.3857, 67.0915, 700, 36.4211]
dgl_128 = [25.7447, 27.5137, 11.7637, 7.4853, 19.5577, 36.5581]
pyg_128 = [62.869, 30.3577, 16.9257, 9.249, 100000, 51.5764]


libra_fp16_256 = [23.4792, 24.0776, 9.997, 6.5683, 23.124, 35.6808]
libra_tf32_256 = [37.5832, 35.5525, 16.1627, 10.2649, 32.2543, 50.8517]
tcgnn_256 =[36.3353, 36.9758, 15.6566,70.5753, 700, 49.2078]
dgl_256 = [38.4639, 38.9706, 16.8093, 11.1258, 34.0359, 51.9079]
pyg_256 = [127.0546, 44.5112, 26.9331, 14.3598, 100000, 81.2535]


libra_fp16_speedup= []
libra_tf32_speedup= []
tcgnn_speedup = []
pyg_speedup = []

libra_fp16_speedup_256 = []
libra_tf32_speedup_256 = []
tcgnn_speedup_256 = []
pyg_speedup_256 = []

i=0
for dgl_sub in dgl_128:
    libra_fp16_speedup.append(round((dgl_sub/libra_fp16_128[i]),4))
    libra_tf32_speedup.append(round((dgl_sub/libra_tf32_128[i]),4))
    tcgnn_speedup.append(round((dgl_sub/tcgnn_128[i]),4))
    pyg_speedup.append(round((dgl_sub/pyg_128[i]),4))
    i+=1
    
i=0
for dgl_sub in dgl_256:
    libra_fp16_speedup_256.append(round((dgl_sub/libra_fp16_256[i]),4))
    libra_tf32_speedup_256.append(round((dgl_sub/libra_tf32_256[i]),4))
    tcgnn_speedup_256.append(round((dgl_sub/tcgnn_256[i]),4))
    pyg_speedup_256.append(round((dgl_sub/pyg_256[i]),4))
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
    # for i in range(4):
    #     res['Category'].append(data+'-2')
            
    for base in baseline:
        res['Type'].append(base)
    for base in baseline:
        res['Type'].append(base)
    # for base in baseline:
    #     res['Type'].append(base)

    res['Value'].append(libra_fp16_speedup[cur])
    res['Value'].append(libra_tf32_speedup[cur])
    res['Value'].append(tcgnn_speedup[cur])
    res['Value'].append(pyg_speedup[cur])

    res['Value'].append(libra_fp16_speedup_256[cur])
    res['Value'].append(libra_tf32_speedup_256[cur])
    res['Value'].append(tcgnn_speedup_256[cur])
    res['Value'].append(pyg_speedup_256[cur])

    # res['Value'].append(libra_fp16_speedup[cur])
    # res['Value'].append(libra_tf32_speedup[cur])
    # res['Value'].append(tcgnn_speedup[cur])
    # res['Value'].append(pyg_speedup[cur])
    
    # 创建DataFrame
    df = pd.DataFrame(res)
    sns.set_style("darkgrid")
    # 绘制多组柱状图
    plt.figure(figsize=(4, 2))
    sns.barplot(x='Category', y='Value', hue='Type', data=df,  palette=mycolor, linewidth=0.6, legend=False, gap=0, width=0.6)

    # 添加标题和标签
    # plt.title('Complex Multiple Group Barplot')
    plt.xlabel('Category')
    # plt.ylabel('Value')

    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/gcn-' + data +'.png', dpi=800)
    cur += 1
    # 清空图形
    plt.clf()
    
    
#fp16 加速比
print("fp16:")
result_tcgnn = [a / b for a in tcgnn_128 for b in libra_fp16_128]
result_dgl = [a / b for a in dgl_128 for b in libra_fp16_128]
result_pyg = [a / b for a in pyg_128 for b in libra_fp16_128]
print("tcgnn: ", stats.gmean(result_tcgnn))
print("dgl ", stats.gmean(result_dgl))
print("pyg: ", stats.gmean(result_pyg))

print("tcgnn: ", max(result_tcgnn))
print("dgl ", max(result_dgl))
print("pyg: ", max(result_pyg))
print()

print('tf32:')
result_tcgnn = [a / b for a in tcgnn_128 for b in libra_tf32_128]
result_dgl = [a / b for a in dgl_128 for b in libra_tf32_128]
result_pyg = [a / b for a in pyg_128 for b in libra_tf32_128]
print("tcgnn: ", stats.gmean(result_tcgnn))
print("dgl ", stats.gmean(result_dgl))
print("pyg: ", stats.gmean(result_pyg))

print("tcgnn: ", max(result_tcgnn))
print("dgl ", max(result_dgl))
print("pyg: ", max(result_pyg))