import numpy as np
import pandas as pd

df1 = pd.read_csv('/home/shijinliang/module/Libra/res/h100/spmm/fp16/filter_h100_spmm_fp16_result_new_128.csv')
df_filtered_h100 = df1[df1['density'] == 3][['dataSet']]

df2 = pd.read_csv('/home/shijinliang/module/Libra/eva100/plot/res_4090/filter_4090_spmm_fp16_result_128.csv')
df_filtered_4090 = df2[df2['density'] == 2][['dataSet']]

intersection = pd.merge(df_filtered_h100, df_filtered_4090, on='dataSet')
df1_new = pd.merge(df1, intersection, on='dataSet')

# 将结果存储为 CSV 文件
intersection.to_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/intersection.csv', index=False)
df1_new.to_csv('/home/shijinliang/module/Libra/eva100/plot/ablation/spmm_h100.csv', index=False)
# 打印保存成功的提示
print("交集结果已成功保存到 intersection.csv")