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

path = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/tf32'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/spmm/fp16'

df1 = pd.read_csv(path + '/new_filter_h100_sddmm_tf32_result_new0220_32.csv')
df1_tf32 = df1[['sddmm_tcu','dataSet']]
df5 = pd.read_csv(path + '/h100_sddmm_tf32_result_flash_32.csv')
df_res = pd.merge(df1_tf32, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

num_edges = []
sgt = []
metcf = []


for index, row in df_res.iterrows():
    metcf.append(round((row['flash']/row['sddmm_tcu']),2))
    num_edges.append(int(row['num_edges']))

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