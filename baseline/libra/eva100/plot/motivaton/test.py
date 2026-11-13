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


# 假设 df2['1'] 已经按降序排列
cuda_threshold = 0.9  # CUDA 的阈值
tcu_threshold = 0.13  # TCU 的阈值

# 计算区间
cuda_range = [x > cuda_threshold for x in data1]  # CUDA 区间
tcu_range = [x < tcu_threshold for x in data1]  # TCU 区间
middle_range = [(x <= cuda_threshold) & (x >= tcu_threshold) for x in data1]  # 中间的3个区间

# # 绘制不同区间的散点图
# sns.set_style("darkgrid")
fig, ax = plt.subplots(figsize=(5, 5))

# 绘制 CUDA 区间的点，颜色为绿色
sns.scatterplot(x=np.array(increment_list)[cuda_range], 
                y=np.array(data1)[cuda_range], 
                label='CUDA', color='green', s=20, edgecolor='none',legend=False,)

# 绘制中间的点，颜色为蓝色
sns.scatterplot(x=np.array(increment_list)[middle_range], 
                y=np.array(data1)[middle_range], 
                label='Middle', color='blue', s=20, edgecolor='none',legend=False,)

# 绘制 TCU 区间的点，颜色为橙色
sns.scatterplot(x=np.array(increment_list)[tcu_range], 
                y=np.array(data1)[tcu_range], 
                label='TCU', color='orange', s=20, edgecolor='none',legend=False,)

# 设置轴标签
plt.xlabel('Matrices')
plt.ylabel('Percentage')

# 显示图形
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/motivaton/distribution.png', dpi=800)

# 清空图形
plt.clf()
