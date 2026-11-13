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
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/h100_spmm_fp16_result_32.csv')
df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n32.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）


dataset = [ 'IGB_small', 'reddit', 'amazon']
baseline = ['DGL-fp16', 'DGL-tf32', 'TC-GNN', 'PyG']

libra_fp16_32 = [18.364, 20.0386, 12.0976]
libra_tf32_32 = [21.4698, 24.36, 12.1427]
tcgnn_32 =[36.3716, 700, 700]
dgl_32 = [31.4121, 107.5515, 20.8632]
pyg_32 = [65.1989, 100000, 39.2367]


libra_fp16_128 = [29.5261, 51.2812, 13.23]
libra_tf32_128 = [39.8019, 64.2323, 15.3166]
tcgnn_128 =[68.9749, 700, 700]
dgl_128 = [52.3586, 136.4583, 24.9658]
pyg_128 = [100000, 100000, 122.6329]






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


cur =0
#循环6次，每个数据集一个图，创建6个图
for data in dataset:
    ind = np.arange(2)  # 柱状图的 x 坐标位置
    width = 0.2  # 柱状图的宽度

    # 绘制柱状图
    fig, ax = plt.subplots(figsize=(4, 2))

    # 每组柱状图的纹理样式
    patterns = ['/', 'x', '-', '\\', '|-']

    #绘制Libra-FP16
    temp = []
    temp.append(libra_fp16_speedup[cur])
    temp.append(libra_fp16_speedup_128[cur])
    bar1 = ax.bar(ind - width - width/2, temp, width, label='Libra-FP16', hatch=patterns[0], color='cornflowerblue', edgecolor='black', linewidth=1)
    
    #绘制Libra-TF32
    temp = []
    temp.append(libra_tf32_speedup[cur])
    temp.append(libra_tf32_speedup_128[cur])
    bar1 = ax.bar(ind - width/2, temp, width, label='Libra-TF32', hatch=patterns[1], color='lightskyblue', edgecolor='black', linewidth=1)
    
    #绘制TC-GNN
    temp = []
    temp.append(tcgnn_speedup[cur])
    temp.append(tcgnn_speedup_128[cur])
    bar1 = ax.bar(ind+width/2, temp, width, label='TC-GNN', hatch=patterns[2], color='lightcoral', edgecolor='black', linewidth=1)
    
    
    #绘制PyG
    temp = []
    temp.append(pyg_speedup[cur])
    temp.append(pyg_speedup_128[cur])
    bar1 = ax.bar(ind + width + width/2, temp, width, label='PyG', hatch=patterns[4], color='lightyellow', edgecolor='black', linewidth=1)

    ax.xaxis.set_visible(False)
    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/agnn/h100-agnn-' + data +'.png', dpi=800)
    cur += 1
    # 清空图形
    plt.clf()
    
print("fp16:")
print(libra_fp16_speedup)
print("dgl ", stats.gmean(libra_fp16_speedup))
print("dl: ", max(libra_fp16_speedup))
print()
# #fp16 加速比
# print("fp16:")
# result_tcgnn = [a / b for a in tcgnn_32 for b in libra_fp16_32]
# result_dgl = [a / b for a in dgl_32 for b in libra_fp16_32]
# result_pyg = [a / b for a in pyg_32 for b in libra_fp16_32]
# print("tcgnn: ", stats.gmean(result_tcgnn))
# print("dgl ", stats.gmean(result_dgl))
# print("pyg: ", stats.gmean(result_pyg))

# print("tcgnn: ", max(result_tcgnn))
# print("dgl ", max(result_dgl))
# print("pyg: ", max(result_pyg))
# print()

# print('tf32:')
# result_tcgnn = [a / b for a in tcgnn_32 for b in libra_tf32_32]
# result_dgl = [a / b for a in dgl_32 for b in libra_tf32_32]
# result_pyg = [a / b for a in pyg_32 for b in libra_tf32_32]
# print("tcgnn: ", stats.gmean(result_tcgnn))
# print("dgl ", stats.gmean(result_dgl))
# print("pyg: ", stats.gmean(result_pyg))

# print("tcgnn: ", max(result_tcgnn))
# print("dgl ", max(result_dgl))
# print("pyg: ", max(result_pyg))