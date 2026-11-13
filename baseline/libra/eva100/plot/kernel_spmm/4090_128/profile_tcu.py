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

flash= []
tcgnn= []
sputnik = []
cusparse = []
rode = []
dtc = []
hcspmm = []
total = 0
total_tcgnn = 0
libra_road = 0
#统计128的数据


path = '/home/shijinliang/module/ppopp26-Libra/result_4090'

#TF32-Libra: BCRS + CUDA-v2
df1 = pd.read_csv(path + "/spmm/tf32_cuda_tcu_bitmap_128_filter.csv")
df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1_res = df1[['libra_tf32','dataSet', 'num_edges']]

#FP16-Libra: BCRS + CUDA-v2
df2 = pd.read_csv(path + "/spmm/fp16_cuda_tcu_bcrs_128_filter.csv")
df2['libra_fp16'] = df2[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df2_res = df2[['libra_fp16','dataSet']]
df_res = pd.merge(df1_res, df2_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#RoDe
df3 = pd.read_csv(path + '/baseline/rode_result_spmm_f32_n128_0710.csv')
# df3 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/kernel_spmm/h100_128/rode_spmm_f32_n128.csv')
df3_res = df3[['Sputnik_time','cuSPARSE_time', 'rode', 'dataSet']]
df_res = pd.merge(df_res, df3_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#TCGNN
df4 = pd.read_csv(path + '/baseline/tcgnn_spmm_4090_128.csv')
df4_res = df4[['tcgnn', 'dataSet']]
df_res = pd.merge(df_res, df4_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#DTC
df5 = pd.read_csv(path + '/baseline/dtc_4090_128.csv')
df5_res = df5[['dtc', 'dataSet']]
df_res = pd.merge(df_res, df5_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#FlashSparse-tf32
df6 = pd.read_csv(path + '/baseline/4090_spmm_tf32_result_flash_128.csv')
df6 = pd.read_csv(path + '/baseline/tf32_flash_bitmap1_128.csv')
df6_res = df6[['flash', 'dataSet']]
df_res = pd.merge(df_res, df6_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#HC-SpMM
df7 = pd.read_csv(path + '/baseline/hcspmm_128.csv')
df7_res = df7[['hcspmm', 'dataSet']]
df_res = pd.merge(df_res, df7_res, on='dataSet', how='inner')  # 使用内连接（inner join）

# #FlashSparse-fp16
# df = pd.read_csv(path + '/baseline/flashsparse_spmm_fp16_128.csv')
# df7_res = df7[['8_1_map_fp16', '8_1_balance_fp16','dataSet']]
# df_res = pd.merge(df_res, df7_res, on='dataSet', how='inner')  # 使用内连接（inner join）
# df_res['libra_fp16_new'] = df_res[['libra_fp16', '8_1_balance_fp16', '8_1_map_fp16']].min(axis=1)

for index, row in df_res.iterrows():
    sputnik.append(round((row['Sputnik_time']/row['libra_tf32']),4))
    cusparse.append(round((row['cuSPARSE_time']/row['libra_tf32']),4))
    rode.append(round((row['rode']/row['libra_tf32']),4))
    
    if row['tcgnn'] != 1000000000:
        tcgnn.append(round((row['tcgnn']/row['libra_tf32']),4))
        total_tcgnn +=1
    dtc.append(round((row['dtc']/row['libra_tf32']),4))
    flash.append(round((row['flash']/row['libra_tf32']),4))
    hcspmm.append(round((row['hcspmm']/row['libra_tf32']),4))
    total+=1


# 统计分布
# 1. TC-GNN
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
print("TC-GNN distribution:")  
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((d/total_tcgnn*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
print("geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
print()

# 5. DTC
a = 0
b = 0
c = 0
d = 0
e = 0
for item in dtc:
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
print("DTC distribution:")  
a_percen = round((a/total*100), 2)
b_percen = round((b/total*100), 2)
c_percen = round((c/total*100), 2)
d_percen = round((d/total*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')

print("dtc : ",stats.gmean(dtc) )
print("max : ",max(dtc) )
print()

# 6. Flash
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
print("Flash distribution:")  
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
print("flash : ",stats.gmean(flash) )
print("max : ",max(flash) )
print()


# 统计分布
# HC-SpMM
a = 0
b = 0
c = 0
d = 0
e = 0
for item in hcspmm:
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
print("HC-SpMM distribution:")  
a_percen = round((a/total_tcgnn*100), 2)
b_percen = round((b/total_tcgnn*100), 2)
c_percen = round((c/total_tcgnn*100), 2)
d_percen = round((d/total_tcgnn*100), 2)
print("<1 ",a_percen, '%')
# print("<1.2 ", b_percen,  '%')
print("<1.5 ", b_percen + c_percen,  '%')
print("<2 ", d_percen,  '%')
print(">=2 ", round((100-a_percen-b_percen-c_percen-d_percen),2),  '%')
print("geo : ",stats.gmean(tcgnn) )
print("max : ",max(tcgnn) )
print()

print(total)