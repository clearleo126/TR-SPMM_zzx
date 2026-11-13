import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns


input_libra = 'h100/spmm/fp16/h100_spmm_fp16_result_128.csv'
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/' + input_libra)
df1 = df1[['dataSet', 'speedup']]
df2 = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
df = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
df = df[df['speedup'] > 1.1]
df['row_number'] = df.reset_index().index
# temp = df[df['speedup'] >= 1.1][['density']]
df['1-2'] = df[['1', '2']].sum(axis=1)
df['3-8'] = 1-df[['1-2']].sum(axis=1)
# df['3-6'] = df[['3', '4', '5', '6']].sum(axis=1)
# df['7-8'] = 1-df[['1-2', '3-6']].sum(axis=1)
num_rows = df.shape[0]
increment_list = [i for i in range(num_rows)]

data1 = df['1-2'].astype(float).values.tolist()
data2 = df['3-8'].astype(float).values.tolist()
# data3 = df['7-8'].astype(float).values.tolist()
# data1_2 = [x + y for x, y in zip(data1, data2)]
# sns.barplot(x='row_number', y='1-2', data=df, color='blue', label='1-2')
# sns.barplot(x='row_number', y='3-6', data=df, color='orange', label='3-6', bottom=df['1-2'])
# sns.barplot(x='row_number', y='7-8', data=df, color='green', label='7-8', bottom=df['1-2']+df['3-6'])

# sns.barplot(x=increment_list, y=data1, color='blue', label='1-2',  edgecolor='none')
# sns.barplot(x=increment_list, y=data2, color='orange', label='3-8', bottom=data1,  edgecolor='none')

sns.lineplot(x=increment_list, y=data1, color='blue', label='1-2',linewidth=0.9)
sns.lineplot(x=increment_list, y=data2, color='orange', label='3-8',linewidth=0.9)
# sns.barplot(x=increment_list, y=data3, color='green', label='7-8', bottom=data1_2)
# 添加标题和标签
# plt.title('NNZ 4-5')
plt.xlabel('Matrices')
plt.ylabel('Nnz of 1-D column vector')

# 显示图形
plt.savefig('eva100/plot/ablation/tcu_cuda_density_spmm_fp16_128.png', dpi=800)
# 清空图形
plt.clf()