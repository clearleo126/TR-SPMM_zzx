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

df1 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/tf32_test_all.csv')
df2 = pd.read_csv('/home/shijinliang/module/RoDe-main/res_spmm_h100/result__spmm_f32_n128.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
df3 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/result/tf32_sgt.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）
df4 = pd.read_csv('/home/shijinliang/module/Libra/data_filter.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）


num_edges = []
tcu_ori_G = []
tcu_part_G = []
tcu_bianry_G = []
tcu_dtc_G = []
tcu_libra_G = []
sputnik_G = []
cusparse_G = []
rode_G = []

total = 0
NG_effctive = 0
NG_speedup = []
format_effctive = 0
format_dtc_effctive = 0
format_speedup= []
format_dtc_speedup= []
tcu_effctive = 0
tcu_speedup= []

for index, row in df_res.iterrows():
    compute = row.iloc[2]*128*2
    # if row['spmm_tcu'] < row['spmm_tcu_binary'] and row['spmm_tcu_binary'] < row['spmm_tcu_sgt'] :
    #     tcu_ori_G.append(round((row['cuSPARSE_time']/row['spmm_tcu_sgt']),4))
    #     tcu_bianry_G.append(round((row['cuSPARSE_time']/row['spmm_tcu_binary']),4))
    #     tcu_libra_G.append(round((row['cuSPARSE_time']/row['spmm_tcu']),4))
    #     tcu_dtc_G.append(round((row['cuSPARSE_time']/row['spmm_tcu_dtc']),4))
        
    num_edges.append(int(row.iloc[2]))
    if row['spmm_tcu'] < row['spmm_tcu_binary']:
        NG_effctive+=1
        NG_speedup.append(round((row['spmm_tcu_binary']/row['spmm_tcu']),4))
    if row['spmm_tcu_binary'] < row['spmm_tcu_sgt'] :
        format_effctive+=1
        format_speedup.append(round((row['spmm_tcu_sgt']/row['spmm_tcu_binary']),4))
    if row['spmm_tcu_binary'] < row['spmm_tcu_dtc'] :
        format_dtc_effctive+=1
        format_dtc_speedup.append(round((row['spmm_tcu_dtc']/row['spmm_tcu_binary']),4))
    if  row['spmm_tcu'] < row['Sputnik_time'] :
        tcu_effctive+=1
        tcu_speedup.append(round(( row['Sputnik_time']/ row['spmm_tcu']),4))
    total+=1
print("NG 有效矩阵个数: " + str(NG_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(NG_speedup)) + '; 最高加速比: ' + str(max(NG_speedup)) )
print("Format vs sgt有效矩阵个数: " + str(format_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(format_speedup)) + '; 最高加速比: ' + str(max(format_speedup)) )
print("Format vs dtc 有效矩阵个数: " + str(format_dtc_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(format_dtc_speedup)) + '; 最高加速比: ' + str(max(format_dtc_speedup)) )
print("TCU加速 有效矩阵个数: " + str(tcu_effctive) + " / " + str(total) + '; 平均加速比: ' + str(mean(tcu_speedup)) + '; 最高加速比: ' + str(max(tcu_speedup)) )

# sorted_indices = sorted(range(len(num_edges)), key=lambda k: num_edges[k]) 




# 统计分布
# bitmap
a = 0
b = 0
c = 0

for item in format_speedup:
    if item < 1.5:
        a+=1
    else:
        c+=1
print("Bitmap distribution:")  
a_percen = round((a/(a+b+c)*100), 2)
print("<1.5 ",a_percen, '%')
print(">=2 ", (100-a_percen),  '%')

a = 0
b = 0
c = 0

for item in NG_speedup:
    if item < 1.5:
        a+=1
    else:
        c+=1
print("Balance distribution:")  
a_percen = round((a/(a+b+c)*100), 2)
print("<1.5 ",a_percen, '%')
print(">=2 ", (100-a_percen),  '%')

a = 0
b = 0
c = 0

for item in format_dtc_speedup:
    if item < 1.5:
        a+=1
    else:
        c+=1
print("DTC distribution:")  
a_percen = round((a/(a+b+c)*100), 2)
print("<1.5 ",a_percen, '%')
print(">=2 ", (100-a_percen),  '%')