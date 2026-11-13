import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns


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

desired_row = df2[df2['dataSet'] == 'pkustk01']


data1 = df2['1'].astype(float).values.tolist()
data1.sort(reverse=True)

for index, row in desired_row.iterrows():
    value = row['1']
new_index = data1.index(value)
print(new_index)
# sns.set_style("darkgrid")
plt.figure(figsize=(5, 5))  # 设置宽度为 10，高度为 6
#sns.lineplot(x=increment_list, y=data1, color='royalblue', linewidth=2.6)
sns.scatterplot(x=increment_list, y=data1, label='Libra-FP16', color='royalblue', s=18, legend=False, edgecolor='none')

print(len([x for x in data1 if x > 0.9]))
print(len([x for x in data1 if x < 0.1]))
# plt.title('NNZ 4-5')
plt.xlabel('Matrices')
plt.ylabel('Percentage')

# 显示图形
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/motivaton/distribution.png', dpi=800)
# 清空图形
plt.clf()