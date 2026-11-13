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
df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/sddmm/tf32/2new_filter_h100_sddmm_tf32_result_neww_128.csv')
#df1 = pd.read_csv('/home/shijinliang/module/Libra-sc25/res/h100/sddmm/fp16/new_filter_h100_sddmm_fp16_result_new0220_32.csv')

cuda = 0
tcu = 0
libra = 0

cuda_sp = []
tcu_sp = []
libra_sp = []

density = []
df1['libra'] = df1[['8', '16', '24', '32', '40', '48', '56', '64']].min(axis=1)
for index, row in df1.iterrows():
    if row['sddmm_tcu'] >= row['libra'] and row['sddmm_cuda'] >= row['libra']:
        libra+=1
    else:
        if row['sddmm_tcu'] < row['sddmm_cuda']:
            tcu+=1
        else:
            cuda+=1
        continue    
    
    # tcu+=1
    # cur_min = min(row['libra'], row['spmm_cuda'])
    tcu_sp.append(round((row['sddmm_tcu'] / row['libra']),2))
    
    # cuda+=1
    # cur_min = min(row['libra'], row['spmm_tcu'])
    cuda_sp.append(round((row['sddmm_cuda'] / row['libra']),2))
        
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
print('CUDA' , ' & ',  b_percen, '\% & ',   c_percen, '\% & ', round((100-a_percen-b_percen-c_percen),2), '\% & ',  round((stats.gmean(cuda_sp)),2), ' & ', round((max(cuda_sp)),2))


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
print('TCU' , ' & ',  b_percen, '\% & ',   c_percen, '\% & ', round((100-a_percen-b_percen-c_percen),2), '\% & ',  round((stats.gmean(tcu_sp)),2), ' & ', round((max(tcu_sp)),2))