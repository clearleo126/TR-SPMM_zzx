import numpy as np
import pandas as pd

df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/sddmm/tf32/2new_filter_h100_sddmm_tf32_result_neww_128.csv')
df_filtered_h100 = df1[df1['density'] == 24][['dataSet']]

df2 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/res_4090/filter_4090_sddmm_tf32_result_128_new.csv')
df_filtered_4090 = df2[df2['density'] == 24][['dataSet']]

intersection = pd.merge(df_filtered_h100, df_filtered_4090, on='dataSet')
df1_new = pd.merge(df1, intersection, on='dataSet')

# 将结果存储为 CSV 文件
intersection.to_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/intersection_sddmm.csv', index=False)
df1_new.to_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/sddmm_h100.csv', index=False)
# 打印保存成功的提示
print("交集结果已成功保存到 intersection.csv")