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


rode = []
flash = []
total = 0


path = '/home/shijinliang/module/Libra-sc25/eva100/plot/res_4090'
'''
Flash
'''
df1 = pd.read_csv(path + '/new_filter_4090_sddmm_tf32_result_new0220_32.csv')
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64', 'sddmm_tcu', 'sddmm_cuda']].min(axis=1)
#RoDe
df2 = pd.read_csv(path + '/rode_result_sddmm_f32_n32_0220.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#tcgnn
df3 = pd.read_csv(path + '/tcgnn_sddmm_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

df4 = pd.read_csv(path + '/4090_sddmm_tf32_result_flash_32.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

#tir
df5 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/res_4090/result_tir_sddmm_32_nosearch.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

for index, row in df_res.iterrows():
    flash.append(round((row['flash']/row['libra']),4))
    total+=1

a = 0
b = 0
c = 0
d = 0
e = 0
for item in flash:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    elif item < 2:
        d+=1
    else:
        e+=1
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

# filtered_list = [x for x in flash if x > 1]
print("flash : ", stats.gmean(flash) )
print("max : ",max(flash) )
print()

'''
RoDe
'''
#(CUDA-v2 + TCU-bitmap)
df1 = pd.read_csv(path + '/new_filter_4090_sddmm_fp16_result_new0220_32.csv')
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64', 'sddmm_tcu', 'sddmm_cuda']].min(axis=1)
#RoDe
df2 = pd.read_csv(path + '/rode_result_sddmm_f32_n32_0220.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#tcgnn
df3 = pd.read_csv(path + '/tcgnn_sddmm_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

df4 = pd.read_csv(path + '/4090_sddmm_tf32_result_flash_32.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）

#tir
df5 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/res_4090/result_tir_sddmm_32_nosearch.csv')
df_res = pd.merge(df_res, df5, on='dataSet', how='inner')  # 使用内连接（inner join）

total = 0
for index, row in df_res.iterrows():
    rode.append(round((row['ours_time']/row['libra']),4))
    total+=1

# 4. Rode
a = 0
b = 0
c = 0
d = 0
e = 0
for item in rode:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    elif item < 2:
        d+=1
    else:
        e+=1
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("geo : ",stats.gmean(rode) )
print("max : ",max(rode) )
print()