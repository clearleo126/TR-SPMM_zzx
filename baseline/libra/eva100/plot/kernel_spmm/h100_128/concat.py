import torch
import pandas as pd
import csv

input_libra = 'h100/spmm/fp16/h100_spmm_fp16_result_128.csv'
input_redo = 'result__spmm_f32_n128.csv'
out_put = 'h100_spmm_fp16_result_128_concat_rode.csv'


df1 = pd.read_csv('/home/shijinliang/module/Libra/res/' + input_libra)
# 计算每行的最小值
df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu']].min(axis=1)
df1['libra_cuda'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
new_df1 = df1[['dataSet', 'num_nodes', 'num_edges', 'spmm_tcu', 'spmm_cuda', '1', '2', '3', '4', '5', '6', '7', '8', 'density', 'libra', 'libra_cuda']]
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/' + input_redo)
new_df2= df2[['dataSet', 'Sputnik_time', 'cuSPARSE_time', 'ours_time']]
merged_df = pd.merge(new_df1, new_df2, on='dataSet', how='inner')  # 使用内连接（inner join）

merged_df['sp1'] = merged_df['ours_time'] / merged_df['spmm_tcu']
merged_df['sp2'] = merged_df['ours_time'] / merged_df['libra']
merged_df['sp3'] = merged_df['ours_time'] / merged_df['libra_cuda']


df3 = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
# new_df3 = df3[['dataSet', '1']]
res = pd.merge(merged_df, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

res.to_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_fp16_128/' + out_put, index=False) 


#绘制Libra，cusaprse, sputnik, Rode的图
