import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
import matplotlib.patches as patches

# input_libra = 'h100/spmm/fp16/h100_spmm_fp16_result_128.csv'
# df1 = pd.read_csv('/home/shijinliang/module/Libra/res/' + input_libra)
# df1 = df1[['dataSet', 'speedup']]
df2 = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
df3 = pd.read_csv('/home/shijinliang/module/Libra/data_filter.csv')
df2 = pd.merge(df2, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
# df = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
# df = df[df['speedup'] > 1.1]
# df['row_number'] = df.reset_index().index


df2['1'] = df2[['1']].sum(axis=1)
# df['3-8'] = 1-df[['1-2']].sum(axis=1)
num_rows = df2.shape[0]
increment_list = [i for i in range(num_rows)]

data1 = df2['1'].astype(float).values.tolist()
data1.sort(reverse=True)

# sns.set_style("darkgrid")
plt.figure(figsize=(5, 5))  # 设置宽度为 10，高度为 6
#sns.lineplot(x=increment_list, y=data1, color='royalblue', linewidth=2.6)
# sns.scatterplot(x=increment_list, y=data1, label='Libra-FP16', color='royalblue', s=18, legend=False, edgecolor='none')

cuda = len([x for x in data1 if x > 0.9])
tcu = len([x for x in data1 if x < 0.15])
print(len([x for x in data1 if x > 0.9]))
print(len([x for x in data1 if x < 0.2]))



# # 绘制颜色带，指定在 x 轴的 10 到 20 之间填充颜色
# plt.axvspan(0, cuda, color='lightgreen', alpha=1)  # alpha 控制透明度
# plt.axvspan(cuda, num_rows-tcu, color='lightskyblue', alpha=1)  # alpha 控制透明度
# plt.axvspan(num_rows-tcu, num_rows, color='orange', alpha=0.8)  # alpha 控制透明度

sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(5, 5))
# ax.add_patch(patches.Rectangle((0, 0), cuda, 1, facecolor='none', hatch='/', edgecolor='green'))
# ax.add_patch(patches.Rectangle((cuda, 0), num_rows-tcu-cuda, 1, facecolor='none', hatch='x', edgecolor='lightskyblue'))
# ax.add_patch(patches.Rectangle((num_rows-tcu, 0), tcu, 1, facecolor='none', hatch='\\', edgecolor='orange'))

# 绘制散点图
sns.scatterplot(x=increment_list, y=data1, label='Libra-FP16', color='royalblue', s=18, legend=False, edgecolor='none')
sns.scatterplot(x=increment_list, y=data1, label='Libra-FP16', color='royalblue', s=18, legend=False, edgecolor='none')
sns.scatterplot(x=increment_list, y=data1, label='Libra-FP16', color='royalblue', s=18, legend=False, edgecolor='none')
# plt.ylim(0, 1)
# plt.xlim(0, num_rows)
# 设置轴标签
plt.xlabel('Matrices')
plt.ylabel('Percentage')
# 显示图形
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/motivaton/distribution.png', dpi=800)
# 清空图形
plt.clf()