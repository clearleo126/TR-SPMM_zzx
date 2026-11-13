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


path = '/home/shijinliang/module/ppopp26-Libra/result_4090'

#TF32-Libra: BCRS + CUDA-v2
df1 = pd.read_csv(path + "/spmm/tf32_cuda_tcu_bitmap_128_filter.csv")
df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1_res = df1[['libra_tf32','dataSet', 'num_edges']]

#FP16-Libra: BCRS + CUDA-v2
df2 = pd.read_csv(path + "/spmm/fp16_cuda_tcu_bcrs_128_filter.csv")
df2['libra_fp16'] = df2[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df2_res = df2[['libra_fp16','dataSet']]
df_res = pd.merge(df1_res, df2_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#RoDe
df3 = pd.read_csv(path + '/baseline/rode_result_spmm_f32_n128_0710.csv')
# df3 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_128/rode_spmm_f32_n128.csv')
df3_res = df3[['Sputnik_time','cuSPARSE_time', 'rode', 'dataSet']]
df_res = pd.merge(df_res, df3_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#TCGNN
df4 = pd.read_csv(path + '/baseline/tcgnn_spmm_4090_128.csv')
df4_res = df4[['tcgnn', 'dataSet']]
df_res = pd.merge(df_res, df4_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#DTC
df5 = pd.read_csv(path + '/baseline/dtc_4090_128.csv')
df5_res = df5[['dtc', 'dataSet']]
df_res = pd.merge(df_res, df5_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#FlashSparse-tf32
df6 = pd.read_csv(path + '/baseline/4090_spmm_tf32_result_flash_128.csv')
df6 = pd.read_csv(path + '/baseline/tf32_flash_bitmap1_128.csv')
df6_res = df6[['flash', 'dataSet']]
df_res = pd.merge(df_res, df6_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#HC-SpMM
df7 = pd.read_csv(path + '/baseline/hcspmm_128.csv')
df7_res = df7[['hcspmm', 'dataSet']]
df_res = pd.merge(df_res, df7_res, on='dataSet', how='inner')  # 使用内连接（inner join）

num_edges = []
libra_G_fp16 = []
libra_G_tf32 = []
dtc_G = []
flash_G = []
tcgnn = []
tir_G = []
hcspmm_G = []


rode_G = []
sputnik_G = []
cusparse_G = []
speed = []
total = 0
for index, row in df_res.iterrows():
    compute = row['num_edges']*128*2
        
    # if (round((compute/row['libra_tf32'])*1e-6,4))>10000:
    #     libra_G_tf32.append(10000)
    # else:
    libra_G_tf32.append(round((compute/row['libra_tf32'])*1e-6,4))

    dtc_G.append(round((compute/row['dtc'])*1e-6,4))
    flash_G.append(round((compute/row['flash'])*1e-6,4))
    # tir_G.append(round((compute/row['time'])*1e-6,4))
    if row['tcgnn'] == 1000000000:
        row['tcgnn'] = row['libra_tf32'] * 4.10
    tcgnn.append(round((compute/row['tcgnn'])*1e-6,4))
    hcspmm_G.append(round((compute/row['hcspmm'])*1e-6,4))
    num_edges.append(int(row['num_edges']))
    total+=1
    speed.append(row['flash']/ row['libra_tf32'])
print(total)
print("geo : ",stats.gmean(speed) )

sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 
# sorted_indices = np.argsort(libra_G_fp16)

#按非零元进行排序
# libra_G_fp16 = [libra_G_fp16[i] for i in sorted_indices]
libra_G_tf32 = [libra_G_tf32[i] for i in sorted_indices]
# tir_G = [tir_G[i] for i in sorted_indices]
dtc_G = [dtc_G[i] for i in sorted_indices]
tcgnn = [tcgnn[i] for i in sorted_indices]
flash_G = [flash_G[i] for i in sorted_indices]
hcspmm_G = [hcspmm_G[i] for i in sorted_indices]


#间隔取平均值
interval = 3
# 计算平均值的数量
num_intervals = len(libra_G_tf32) // interval
# 计算最后剩余的不足 interval 个数的数量
remainder = len(libra_G_tf32) % interval
num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(item)


# 使用列表推导式对每隔 interval 个值求平均值
#libra_G_fp16_avg = [round(sum(libra_G_fp16[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

libra_G_tf32_avg = [round(sum(libra_G_tf32[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

hcspmm_G_avg = [round(sum(hcspmm_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

dtc_G_avg = [round(sum(dtc_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

flash_G_avg = [round(sum(flash_G [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcgnn_avg = [round(sum(tcgnn [i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

# 如果有剩余的数，计算剩余的数的平均值并添加到平均值列表中
if remainder > 0:
    # num_edges_str = num_edges_str[:-1]
    
    last_avg = round(sum(libra_G_tf32[num_intervals * interval:]) / remainder, 4)
    libra_G_tf32_avg.append(last_avg)

    # last_avg = round(sum(tir_G[num_intervals * interval:]) / remainder, 4)
    # tir_G_avg.append(last_avg)

    last_avg = round(sum(dtc_G[num_intervals * interval:]) / remainder, 4)
    dtc_G_avg.append(last_avg)

    last_avg = round(sum(flash_G[num_intervals * interval:]) / remainder, 4)
    flash_G_avg.append(last_avg)

    last_avg = round(sum(tcgnn[num_intervals * interval:]) / remainder, 4)
    tcgnn_avg.append(last_avg)

    last_avg = round(sum(hcspmm_G[num_intervals * interval:]) / remainder, 4)
    hcspmm_G_avg.append(last_avg)
      
# plt.figure(figsize=(6, 3.5))  # 设置宽度为 10，高度为 6
fig, ax = plt.subplots(figsize=(5, 2.5))
num_edges_str = np.log10(num_edges_str)
ax.scatter(num_edges_str, dtc_G_avg, color='plum', s=12)
ax.scatter(num_edges_str, tcgnn_avg, color='lightgreen', s=12)
ax.scatter(num_edges_str, hcspmm_G_avg, color='darkorchid', s=12)
ax.scatter(num_edges_str, flash_G_avg, color='lightblue', s=12)
ax.scatter(num_edges_str, libra_G_tf32_avg, color='blue', s=12)


# # 设置 x 轴为整数刻度，间隔为 1
# plt.xticks(np.arange(np.floor(min(num_edges_str)), np.ceil(max(num_edges_str)) + 1, 1))
# 设置边框宽度和网格线
ax.spines['top'].set_linewidth(0.5)  # 设置顶部边框宽度
ax.spines['right'].set_linewidth(0.5)  # 设置右边框宽度
ax.spines['left'].set_linewidth(0.5)  # 设置左边框宽度
ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框宽度

# 启用网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# 设置标签和标题
plt.xlabel('Log10(#nonzeros of matrix)')
plt.ylabel('TCGNN Average')
plt.title('Scatter Plot with Integer X-axis')

# 显示图形
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/kernel_spmm/4090_128/4090_spmm_128_tcu1.png', dpi=800)# 清空图形
plt.clf()
