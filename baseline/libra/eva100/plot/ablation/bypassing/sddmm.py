# 读取数据
import pandas as pd
from scipy.stats import gmean

df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/bypassing/2new_filter_h100_sddmm_tf32_result_new0713_new_32.csv')

epsilon = 1e-5
df['improve_ratio'] = df['flash-shared'] / (df['flash'] + epsilon)
speedup_df = df[df['improve_ratio'] >= 1]


# 算术平均加速比（只对 speedup_df）
arith_avg_speedup = speedup_df['improve_ratio'].mean()

# （可选）几何平均加速比
geo_avg_speedup = gmean(speedup_df['improve_ratio'])


print(f"\n只对有加速部分的算术平均加速比: {arith_avg_speedup:.4f}")
print(f"只对有加速部分的几何平均加速比: {geo_avg_speedup:.4f}")

# 计算有加速比（>1）的总数及比例
total = len(df)
num_speedup = (df['improve_ratio'] >= 1).sum()
print(f"\n有加速比的比例 (>=1): {num_speedup}/{total} = {num_speedup/total:.4f}")

# 计算 1–1.2 小加速 和 ≥1.2 大加速 的数量及比例
num_small = ((df['improve_ratio'] >= 1) & (df['improve_ratio'] < 1.2)).sum()
num_large = (df['improve_ratio'] >= 1.2).sum()

print(f"1–1.2 倍小加速的比例: {num_small}/{total} = {num_small/total:.4f}")
print(f"≥1.2 倍加速的比例: {num_large}/{total} = {num_large/total:.4f}")