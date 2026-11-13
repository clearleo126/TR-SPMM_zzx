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


dataset = [ 'IGB_meduium', 'Amazonproducts', 'ogb', 'DGraphFin']
baseline = ['DGL-fp16', 'DGL-tf32', 'DTC']


libra_tcu_fp16_128 = [1056.58, 1411.67, 1182.29, 754.63]
libra_cuda_fp16_128 = [1306.77, 2636.77, 1857.86, 602.47]

libra_tcu_tf32_128 = [803.23, 1176.49, 906.3, 646.65]
libra_cuda_tf32_128 = [793.8, 1870, 969.90, 499.006]

# libra_fp16_128 = [1306.77, 2636.77, 1857.86, 754.63]
# libra_tf32_128 = [793.8,1870, 969.90, 646.65]
libra_fp16_128 = [1306.77, 2636.77, 1857.86, 754.63]
libra_tf32_128 = [803.23, 1870, 969.90, 646.65]
dtc_128 =[611.7, 1020.1, 773.4, 329.4119]



#循环6次，每个数据集一个图，创建6个图

ind = np.arange(4)  # 柱状图的 x 坐标位置
width = 0.26  # 柱状图的宽度

# 绘制柱状图
fig, ax = plt.subplots(figsize=(6, 2))

# 每组柱状图的纹理样式
patterns = ['/', 'x', '\\', '|-']

#绘制Libra-FP16
bar1 = ax.bar(ind - width, libra_fp16_128, width, label='Libra-FP16', hatch=patterns[0], color='cornflowerblue', edgecolor='black', linewidth=1)

#绘制Libra-TF32
bar1 = ax.bar(ind, libra_tf32_128, width, label='Libra-TF32', hatch=patterns[1], color='lightskyblue', edgecolor='black', linewidth=1)

#绘制TC-GNN
bar1 = ax.bar(ind + width, dtc_128, width, label='TC-GNN', hatch=patterns[2], color='lemonchiffon', edgecolor='black', linewidth=1)


ax.xaxis.set_visible(False)
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/h100-gcn-huge' +'.png', dpi=800)

# 清空图形
plt.clf()
    
    
#fp16 加速比
print("fp16:")
result_dtc = [a / b for a in libra_fp16_128 for b in dtc_128]
print(result_dtc)
print("dtc: ", stats.gmean(result_dtc))


print("dtc: ", max(result_dtc))

print()

print('tf32:')
print(result_dtc)
result_dtc = [a / b for a in libra_tf32_128 for b in dtc_128]

print("dtc: ", stats.gmean(result_dtc))


print("dtc: ", max(result_dtc))
