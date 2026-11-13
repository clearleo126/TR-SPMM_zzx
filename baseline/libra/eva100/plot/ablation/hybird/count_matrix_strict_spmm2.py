import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd
import seaborn as sns
from scipy import stats
from collections import Counter

bound = 1.01
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_128.csv')
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_new_0216_128.csv')
# df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_new_0218_128.csv')
# df2 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/data_square.csv')
# df = pd.merge(df, df2, on='dataSet', how='inner')  # 使用内连接（inner join）
cuda = 0
tcu = 0
libra = 0

cuda_sp = []
tcu_sp = []
libra_sp = []

density = []
df1['libra'] = df1[['1', '2', '3', '4', '5', '6', '7', '8']].min(axis=1)
for index, row in df1.iterrows():
    if row['spmm_tcu'] >= row['libra'] and row['spmm_cuda'] >= row['libra']:
        libra+=1
    else:
        if row['spmm_tcu'] < row['spmm_cuda']:
            tcu+=1
        else:
            cuda+=1
        continue
        
    # cur_min = min(row['libra'], row['spmm_cuda'])
    tcu_sp.append(round((row['spmm_tcu'] / row['libra']),2))

    # cur_min = min(row['libra'], row['spmm_tcu'])
    cuda_sp.append(round((row['spmm_cuda'] / row['libra']),2))
        
print("CUDA matrices: " + str(cuda))
print("TCU matrices: " + str(tcu))
print("Hybrid matrices: " + str(libra))

print("CUDA加速比分布:")
a = 0
b = 0
c = 0
d = 0

for item in cuda_sp:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    else:
        d+=1
a_percen = round((a/(a+b+c+d)*100), 2)
b_percen = round((b/(a+b+c+d)*100), 2)
c_percen = round((c/(a+b+c+d)*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print(">=1.5 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("Geo: ", round((stats.gmean(cuda_sp)),2))
print("Max: ", round((max(cuda_sp)),2))
print('CUDA' , ' & ', b_percen, '\% & ',  c_percen, '\% & ', round((100-a_percen-b_percen-c_percen),2), '\% & ',  round((stats.gmean(cuda_sp)),2), ' & ', round((max(cuda_sp)),2))


print("TCU加速比分布:")
a = 0
b = 0
c = 0
d = 0

for item in tcu_sp:
    if item < 1:
        a+=1
    elif item < 1.2:
        b+=1
    elif item < 1.5:
        c+=1
    else:
        d+=1
a_percen = round((a/(a+b+c+d)*100), 2)
b_percen = round((b/(a+b+c+d)*100), 2)
c_percen = round((c/(a+b+c+d)*100), 2)
print("<1 ",a_percen, '%')
print("<1.2 ", b_percen,  '%')
print("<1.5 ", c_percen,  '%')
print(">=1.5 ", round((100-a_percen-b_percen-c_percen),2),  '%')
print("Geo: ", round((stats.gmean(tcu_sp)),2))
print("Max: ", round((max(tcu_sp)),2))
print('TCU' , ' & ', b_percen, '\% & ',  c_percen, '\% & ', round((100-a_percen-b_percen-c_percen),2), '\% & ',  round((stats.gmean(tcu_sp)),2), ' & ', round((max(tcu_sp)),2))