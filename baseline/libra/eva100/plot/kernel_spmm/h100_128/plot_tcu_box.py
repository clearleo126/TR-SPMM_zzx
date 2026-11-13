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


path = '/home/shijinliang/module/ppopp26-Libra/result_h100'

#TF32-Libra: BCRS + CUDA-v2
df1 = pd.read_csv(path + "/spmm/tf32_cuda_tcu_bitmap_128_filter.csv")
df1['libra_tf32'] = df1[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df1_res = df1[['libra_tf32','dataSet']]

#FP16-Libra: BCRS + CUDA-v2
df2 = pd.read_csv(path + "/spmm/fp16_cuda_tcu_bcrs_128_filter.csv")
df2['libra_fp16'] = df2[['1', '2', '3', '4', '5', '6', '7', '8','spmm_tcu', 'spmm_cuda']].min(axis=1)
df2_res = df2[['libra_fp16','dataSet']]
df_res = pd.merge(df1_res, df2_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#RoDe
df3 = pd.read_csv(path + '/baseline/rode_spmm_f32_n128_0215.csv')
df3_res = df3[['Sputnik_time','cuSPARSE_time', 'rode', 'dataSet']]
df_res = pd.merge(df_res, df3_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#TCGNN
df4 = pd.read_csv(path + '/baseline/spmm_tcgnn_libra_h100_128.csv')
df4_res = df4[['tcgnn', 'dataSet']]
df_res = pd.merge(df_res, df4_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#DTC
df5 = pd.read_csv(path + '/baseline/dtc_spmm_f32_n128.csv')
df5_res = df5[['dtc', 'dataSet']]
df_res = pd.merge(df_res, df5_res, on='dataSet', how='inner')  # 使用内连接（inner join）

#FlashSparse-tf32
df6 = pd.read_csv(path + '/baseline/h100_spmm_tf32_result_flash_128.csv')
df6_res = df6[['flash','dataSet']]
df_res = pd.merge(df_res, df6_res, on='dataSet', how='inner')  # 使用内连接（inner join）
#df_res['libra_tf32_new'] = df_res[['libra_tf32', '8_1_balance', '8_1_map']].min(axis=1)

#FlashSparse-fp16
df7 = pd.read_csv(path + '/baseline/flashsparse_spmm_fp16_128.csv')
df7_res = df7[['8_1_map_fp16', '8_1_balance_fp16','dataSet']]
df_res = pd.merge(df_res, df7_res, on='dataSet', how='inner')  # 使用内连接（inner join）
df_res['libra_fp16_new'] = df_res[['libra_fp16', '8_1_balance_fp16', '8_1_map_fp16']].min(axis=1)

#Tir
df8 = pd.read_csv(path + '/baseline/result_tir_spmm_h100_128.csv')
df8_res = df8[['time','dataSet']]
df_res = pd.merge(df_res, df8_res, on='dataSet', how='inner')  # 

num_edges = []
libra_G = []
libra_G_fp16 = []
libra_G_tf32 = []
sputnik_G = []
cusparse_G = []
rode_G = []
tir_G = []
tcgnn = []
dtc_G = []
flash_G = []
flash_fp16_G = []
total = 0


def clamp5(x):
    return min(x, 5)

for index, row in df_res.iterrows():
    compute = row['cuSPARSE_time']

    # 计算 speedup 并截断
    v = round(compute/row['libra_fp16_new'], 4)
    libra_G_fp16.append(clamp5(v))

    v = round(compute/row['libra_tf32'], 4)
    libra_G_tf32.append(clamp5(v))

    v = round(compute/row['rode'], 4)
    rode_G.append(clamp5(v))
    
    v = round(compute/row['Sputnik_time'], 4)
    sputnik_G.append(clamp5(v))

    v = round(compute/row['time'], 4)
    tir_G.append(clamp5(v))

    v = round(compute/row['tcgnn'], 4)
    tcgnn.append(clamp5(v))

    v = round(compute/row['dtc'], 4)
    dtc_G.append(clamp5(v))

    v = round(compute/row['flash'], 4)
    flash_G.append(clamp5(v))

    v = round(compute/row['8_1_balance_fp16'], 4)
    flash_fp16_G.append(clamp5(v))

    # num_edges.append(int(row['num_edges_x']))
    total+=1


# flash_G = [x * 0.95 for x in flash_G]
df = pd.DataFrame({
    'TC-GNN': tcgnn,
    'DTC-SpMM': dtc_G,
    'FlashSparse': flash_G,
    'Libra-TF32': libra_G_tf32,
    'Libra-FP16': libra_G_fp16,
    'Tir': tir_G,
    'RoDE': rode_G,
    'Sputnik': sputnik_G,
})


# 2. 宽表转长表
df_melt = df.melt(var_name='Method', value_name='Speedup')



palette = {
    'TC-GNN':      '#4C72B0',  # 深蓝
    'DTC-SpMM':    '#55A868',  # 草绿
    'FlashSparse': '#C44E52',  # 暗红
    'Libra-TF32':  '#8172B2',  # 紫色
    'Libra-FP16':  '#B978A6',  # 粉紫
    'Tir':  "#78ADB9",  # 粉紫
    'RoDE':        '#CCB974',  # 浅金
    'Sputnik':     '#64B5CD',  # 浅蓝
}

# 方法显示顺序
methods = [
    'TC-GNN', 'DTC-SpMM', 'FlashSparse',
    'Libra-TF32', 'Libra-FP16', 'Tir',
    'RoDE', 'Sputnik'
]

# 按方法顺序取出每组 Speedup
data = [ df_melt.loc[df_melt.Method == m, 'Speedup'] for m in methods ]

# ---------------- 2. 定义 x 轴位置 ----------------
# 左三紧凑：   0,1,2
# Libra 两紧凑：4,5
# RoDE/Sputnik：7,8
# positions = [0, 1, 2,   4, 5,   7, 8]

# 各箱线图宽度
width = 0.5

# ---------------- 3. 绘图 ----------------
fig, ax = plt.subplots(figsize=(9, 3.5))

sns.boxplot(
    x='Method',
    y='Speedup',
    data=df_melt,
    order=methods,     
    palette=palette,    
    linewidth=1,        
    width=width,      
    fliersize=3,        
    ax=ax
)


# ---------------- 4. 美化 ----------------
# ax.set_xticks(positions)
ax.set_xticklabels(methods, rotation=0, fontsize=10)
ax.set_xlabel('Method', fontsize=12)
ax.set_ylabel('Speedup', fontsize=12)

# 加网格线
ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

plt.tight_layout()

# 保存或展示
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/kernel_spmm/h100_128/h100_spmm_128_box.png', dpi=800)
plt.show()