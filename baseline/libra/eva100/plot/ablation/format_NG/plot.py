import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
def mean(input_list):
    return round((sum(input_list) / len(input_list)),1)

df1 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16.csv')
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n128.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
df3 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16_sgt.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
df4 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/data_square.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）


num_edges = []
tcu_ori_G = []
tcu_part_G = []
tcu_bianry_G = []
tcu_libra_G = []
sputnik_G = []
cusparse_G = []
rode_G = []

total = 0
NG_effctive = 0
NG_speedup = []
format_effctive = 0
format_speedup= []
tcu_effctive = 0
tcu_speedup= []

for index, row in df_res.iterrows():
    compute = row.iloc[2]*128*2
    if row['spmm_tcu'] < row['spmm_tcu_binary'] and row['spmm_tcu_binary'] < row['spmm_tcu_sgt'] :
        tcu_ori_G.append(round((compute/row['spmm_tcu_sgt'])*1e-6,4))
        tcu_part_G.append(round((compute/row['spmm_tcu_part'])*1e-6,4))
        tcu_bianry_G.append(round((compute/row['spmm_tcu_binary'])*1e-6,4))
        tcu_libra_G.append(round((compute/row['spmm_tcu'])*1e-6,4))
        sputnik_G.append(round((compute/row['Sputnik_time'])*1e-6,4))
        cusparse_G.append(round((compute/row['cuSPARSE_time'])*1e-6,4))
        rode_G.append(round((compute/row['ours_time'])*1e-6,4))
        
        num_edges.append(int(row.iloc[2]))
        if row['spmm_tcu'] < row['spmm_tcu_binary']:
            NG_effctive+=1
            NG_speedup.append(round((row['spmm_tcu_binary']/row['spmm_tcu']),4))
        if row['spmm_tcu_binary'] < row['spmm_tcu_sgt'] :
            format_effctive+=1
            format_speedup.append(round((row['spmm_tcu_sgt']/row['spmm_tcu_binary']),4))
        if  row['spmm_tcu'] < row['Sputnik_time'] :
            tcu_effctive+=1
            tcu_speedup.append(round(( row['Sputnik_time']/ row['spmm_tcu']),4))
        total+=1
print("NG 有效矩阵个数: " + str(NG_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(NG_speedup)) + '; 最高加速比: ' + str(max(NG_speedup)) )
print("Format 有效矩阵个数: " + str(format_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(format_speedup)) + '; 最高加速比: ' + str(max(format_speedup)) )
print("TCU加速 有效矩阵个数: " + str(tcu_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(tcu_speedup)) + '; 最高加速比: ' + str(max(tcu_speedup)) )

# sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 
sorted_indices = sorted(range(len(tcu_ori_G)), key=lambda k: tcu_ori_G[k]) 
#按非零元进行排序
tcu_ori_G = [tcu_ori_G[i] for i in sorted_indices]
tcu_part_G = [tcu_part_G[i] for i in sorted_indices]
tcu_bianry_G = [tcu_bianry_G[i] for i in sorted_indices]
tcu_libra_G = [tcu_libra_G[i] for i in sorted_indices]
sputnik_G = [sputnik_G[i] for i in sorted_indices]
cusparse_G = [cusparse_G[i] for i in sorted_indices]
rode_G = [rode_G[i] for i in sorted_indices]

#间隔取平均值
interval = 5

num_edges = [num_edges[i] for i in sorted_indices]
num_edges = num_edges[::interval]
num_edges_str = []
for item in num_edges:
    num_edges_str.append(str(item))
# 计算平均值的数量
num_intervals = len(tcu_ori_G) // interval
# 计算最后剩余的不足 interval 个数的数量
remainder = len(tcu_ori_G) % interval

# 使用列表推导式对每隔 interval 个值求平均值
tcu_ori_G_avg = [round(sum(tcu_ori_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcu_part_G_avg = [round(sum(tcu_part_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcu_bianry_G_avg = [round(sum(tcu_bianry_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

tcu_libra_G_avg = [round(sum(tcu_libra_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

sputnik_G_avg = [round(sum(sputnik_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

cusparse_G_avg = [round(sum(cusparse_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]

rode_G_avg = [round(sum(rode_G[i:i+interval]) / interval, 4) for i in range(0, num_intervals * interval, interval)]
# 如果有剩余的数，计算剩余的数的平均值并添加到平均值列表中
if remainder > 0:
    last_avg = round(sum(tcu_ori_G_avg[num_intervals * interval:]) / remainder, 4)
    tcu_ori_G_avg.append(last_avg)
    
    last_avg = round(sum(tcu_part_G_avg[num_intervals * interval:]) / remainder, 4)
    tcu_part_G_avg.append(last_avg)

    last_avg = round(sum(tcu_bianry_G_avg[num_intervals * interval:]) / remainder, 4)
    tcu_bianry_G_avg.append(last_avg)
    
    last_avg = round(sum(tcu_libra_G_avg[num_intervals * interval:]) / remainder, 4)
    tcu_libra_G_avg.append(last_avg)
    
    last_avg = round(sum(sputnik_G_avg[num_intervals * interval:]) / remainder, 4)
    sputnik_G_avg.append(last_avg)
    
    last_avg = round(sum(cusparse_G_avg[num_intervals * interval:]) / remainder, 4)
    cusparse_G_avg.append(last_avg)

    last_avg = round(sum(rode_G_avg[num_intervals * interval:]) / remainder, 4)
    rode_G_avg.append(last_avg)
plt.figure(figsize=(10, 6))  # 设置宽度为 10，高度为 6


#3.all
sns.lineplot(x=num_edges_str, y=tcu_ori_G_avg, label='tcu_ori', linewidth=1.5)
# sns.lineplot(x=num_edges_str, y=tcu_part_G_avg, label='tcu_part', linewidth=0.9)
sns.lineplot(x=num_edges_str, y=tcu_bianry_G_avg, label='tcu_bianry', linewidth=1.5)
sns.lineplot(x=num_edges_str, y=tcu_libra_G_avg, label='tcu_libra', linewidth=1.5)
#3.all
# sns.scatterplot(x=num_edges_str, y=tcu_ori_G_avg, label='tcu_ori')
# # sns.lineplot(x=num_edges_str, y=tcu_part_G_avg, label='tcu_part', )
# sns.scatterplot(x=num_edges_str, y=tcu_bianry_G_avg, label='tcu_bianry')
# sns.scatterplot(x=num_edges_str, y=tcu_libra_G_avg, label='tcu_libra')

# 添加标题和标签
plt.title('fp16')
plt.xlabel('Matrices')
plt.ylabel('GFLOPS')
plt.xticks(rotation=45)
plt.xticks(fontsize=7)
plt.xticks(ticks=num_edges_str[::5])
# 显示图形
plt.savefig('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/fp16_all.png', dpi=800)
# 清空图形
plt.clf()

