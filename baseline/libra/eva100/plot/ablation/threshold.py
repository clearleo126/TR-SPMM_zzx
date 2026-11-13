import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

res = []
mydataset = ['Zd_Jac3', 'rma10', 'ex19']
dataset = ['SiO2', 'pkustk05', 'Ga3As3H12', 'Zd_Jac2','gupta2']
dataset = ['ex19', '2D_27628_bjtcai', '3D_28984_Tetra', 'gupta1','std1_Jac2']
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_new_128.csv')
cur = 0
for index, row in df1.iterrows():
    # if cur < 10 and row['density'] ==3 :
    #     print(row['dataSet'])
    if row['dataSet'] in dataset:
        print(row['dataSet'])
        temp = []
        temp.append(row['spmm_cuda'] / row['1'])
        temp.append(row['spmm_cuda'] / row['2'])
        temp.append(row['spmm_cuda'] / row['3'])
        temp.append(row['spmm_cuda'] / row['4'])
        temp.append(row['spmm_cuda'] / row['5'])
        temp.append(row['spmm_cuda'] / row['6'])
        temp.append(row['spmm_cuda'] / row['7'])
        temp.append(row['spmm_cuda'] / row['8'])
        res.append(temp)
        cur +=1
print(res)
xticks_labels = ['1', '2', '3', '4', '5', '6', '7', '8']
# Normalize data by row (using MinMaxScaler)
scaler = MinMaxScaler(feature_range=(0.5, 1.5))  # 归一化到 [0, 2] 范围
data_normalized = np.array([scaler.fit_transform(np.array(row).reshape(-1, 1)).flatten() for row in res])
# Create a heatmap
plt.figure(figsize=(12, 3))
sns.heatmap(res, annot=True,  fmt=".2f", cmap="Blues", cbar_kws={'label': 'Speedup'}, square=True, xticklabels=xticks_labels, linewidths=0.1)

# Add labels (example, adjust according to your axes)
plt.xlabel('ngs')
plt.ylabel('dw')
plt.title('Heatmap Example')

plt.savefig('/home/shijinliang/module/Libra/eva100/plot/ablation/preprocess' +'.png', dpi=800)
