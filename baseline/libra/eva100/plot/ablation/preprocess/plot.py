import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import coo_matrix
import scipy.sparse as sp
import torch
from collections import Counter
import pandas as pd


df1 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/preprocess/result/pre.csv')

speedup = []
total = 0
eff = 0
a =0 
b =0

for index, row in df1.iterrows():
    sp = round((row.iloc[3] / row.iloc[4]),2)
    speedup.append(sp)
    
    total +=1 
    if row.iloc[3] >= row.iloc[4]:
        eff+=1
        
    if sp > 1 and sp <1.2 :
        a+=1
    
    if sp > 1 and sp >=1.2 :
        b+=1
        
geometric_mean = np.exp(np.mean(np.log(speedup)))

# 输出结果
print("最大值:", max(speedup))
print("平均值:", geometric_mean)
print("Total:", total)
print("Eff:", eff)
print("1-1.2:", a/(a+b)*100)
print(">1.2:", b/(a+b)*100)