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

path = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/tf32'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/fp16'

df1 = pd.read_csv(path + '/filter_h100_spmm_tf32_result_new2_0221_128.csv')
df1_tf32 = df1[['spmm_tcu','dataSet']]
df2 = pd.read_csv('/home/shijinliang/module/Libra-sc25/eva100/plot/ablation/format_NG/result/tf32_sgt.csv')
df_res = pd.merge(df1_tf32, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
df5 = pd.read_csv(path + '/h100_spmm_tf32_result_flash_128.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

num_edges = []
sgt = []
metcf = []


for index, row in df_res.iterrows():
    if row['spmm_tcu_sgt'] != 1000000000:
        sgt.append(round((row['spmm_tcu_sgt']/row['spmm_tcu']),2))
    metcf.append(round((row['flash']/row['spmm_tcu']),2))
    num_edges.append(int(row['num_edges_x']))

#SGT:
a=0
b=0
eff=0
for item in sgt:
    if item > 1:
        eff+=1
    if item > 1 and item <1.2:
        a+=1
    if item >= 1.2:
        b+=1

# 输出结果
print("最大值:", max(sgt))
print("平均值:", round(stats.gmean(sgt),2))
print("Total:", len(sgt))
print("Eff:", eff)
print("1-1.2:", a/(a+b)*100)
print(">1.2:", b/(a+b)*100)
print()

#METCF:
a=0
b=0
eff=0
for item in metcf:
    if item > 1:
        eff+=1
    if item > 1 and item <1.2:
        a+=1
    if item >= 1.2:
        b+=1

# 输出结果
print("最大值:", max(metcf))
print("平均值:", round(stats.gmean(metcf),2))
print("Total:", len(metcf))
print("Eff:", eff)
print("1-1.2:", a/(a+b)*100)
print(">1.2:", b/(a+b)*100)