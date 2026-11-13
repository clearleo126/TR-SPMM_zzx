#绘制Libra，cusaprse, sputnik, Rode的图
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

df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_128.csv')
# df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)

data = 'pkustk01'
filtered_rows = df1[df1['dataSet'] == data]
df2 = pd.read_csv('/home/shijinliang/module/Libra/data.csv')
df_res = pd.merge(filtered_rows, df2, on='dataSet', how='inner')  # 使用内连接（inner join）

df3 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n128.csv')
df3 = df3[['dataSet', 'cuSPARSE_time']]
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）


for index, row in df_res.iterrows():
    min_cur = row['cuSPARSE_time']
    speed = []
    # speed.append(round((min_cur / row['spmm_tcu']),2))
    speed.append(round((min_cur / row['1_x']),2))
    speed.append(round((min_cur / row['2_x']),2))
    speed.append(round((min_cur / row['3_x']),2))
    speed.append(round((min_cur / row['4_x']),2))
    speed.append(round((min_cur / row['5_x']),2))
    speed.append(round((min_cur / row['6_x']),2))
    speed.append(round((min_cur / row['7_x']),2))
    speed.append(round((min_cur / row['8_x']),2))
    speed.append(round((min_cur / row['spmm_cuda']),2))
    print(speed)
    percen = []
    percen.append(round(( row['1_y']),2))
    percen.append(round(( row['2_y']),2))
    percen.append(round(( row['3_y']),2))
    percen.append(round(( row['4_y']),2))
    percen.append(round(( row['5_y']),2))
    percen.append(round(( row['6_y']),2))
    percen.append(round(( row['7_y']),2))
    percen.append(round(( row['8_y']),2))
    print(percen)
    
    # sns.set_style("darkgrid")
    plt.figure(figsize=(5, 3))  # 设置宽度为 10，高度为 6
    #vs 
    x_axis= list(range(1, 10))
    #g = sns.lineplot(x=x_axis, y=speed, label='Libra-FP16', color='royalblue', linewidth=2, legend=False)
    g = sns.scatterplot(x=x_axis, y=speed, label='Libra-FP16', color='royalblue', s=80, legend=False)

    plt.xticks(ticks=x_axis)
    # 显示图形
    plt.savefig('/home/shijinliang/module/Libra/eva100/plot/ablation/hybird/typical_profile/' + data + '1.png', dpi=800)
    # 清空图形
    plt.clf()
