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

tcgnn= []
sputnik = []
cusparse = []
rode = []
flash = []
total = 0
total_tcgnn = 0
libra_road = 0

path = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/fp16'
path1 = '/home/shijinliang/module/Libra-sc25/res/h100/sddmm/tf32'
'''
Flash
'''

#(CUDA-v2 + TCU-bitmap)
df1 = pd.read_csv(path1 + '/new_filter_h100_sddmm_tf32_result_new0220_32.csv')
# df1 = pd.read_csv(path1 + '/2new_filter_h100_sddmm_tf32_result_neww_128.csv')
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64', 'sddmm_tcu', 'sddmm_cuda']].min(axis=1)
# df1['flash'] = df1[['sddmm_tcu']].min(axis=1)
#RoDe
df2 = pd.read_csv(path + '/rode_sddmm_f32_n128.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#tcgnn
df3 = pd.read_csv(path + '/sddmm_tcgnn_libra_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

df4 = pd.read_csv(path1 + '/h100_sddmm_tf32_result_flash_32.csv')
df_res = pd.merge(df_res, df4, on='dataSet', how='inner')  # 使用内连接（inner join）
total_tcgnn = 0
for index, row in df_res.iterrows():
    flash.append(round((row['flash']/row['libra']),4))
    if row['tcgnn'] != 1000000000:
        tcgnn.append(round((row['tcgnn']/row['libra']),4))
        total_tcgnn +=1
    total+=1
    
# TC-GNN
print("TC-GNN")
a = 0
b = 0
c = 0
d = 0
e = 0
for item in tcgnn:
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
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((d/total_tcgnn*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("TC-GNN geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
print()

print("Flash")
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
print("flash geo : ", stats.gmean(flash) )
print("max : ",max(flash) )
print()

'''
RoDe
'''
print("RoDe")
#(CUDA-v2 + TCU-bitmap)
df1 = pd.read_csv(path + '/new_filter_h100_sddmm_fp16_result_new0221_32.csv')
# df1 = pd.read_csv(path + '/new_filter_h100_sddmm_fp16_result_32.csv')
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64', 'sddmm_tcu', 'sddmm_cuda']].min(axis=1)
df1['flash'] = df1[['sddmm_tcu']].min(axis=1)
#RoDe
df2 = pd.read_csv(path + '/rode_sddmm_f32_n32.csv')
df_res = pd.merge(df1, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
#tcgnn
df3 = pd.read_csv(path + '/sddmm_tcgnn_libra_h100_32.csv')
df_res = pd.merge(df_res, df3, on='dataSet', how='inner')  # 使用内连接（inner join）

total=0
for index, row in df_res.iterrows():
    rode.append(round((row['rode']/row['libra']),4))
    total+=1

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

print("RoDe geo : ",stats.gmean(rode) )
print("max : ",max(rode) )
print()