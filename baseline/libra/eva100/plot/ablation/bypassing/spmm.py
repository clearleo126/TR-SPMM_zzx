# 读取数据
import pandas as pd
from scipy.stats import gmean

df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/bypassing/tf32_flash_bitmap_128.csv')

epsilon = 1e-5
df['improve_ratio'] = df['flash-shared'] / (df['flash'] + epsilon)

# 最大／平均提升比
print(f"\n最大提升比: {df['improve_ratio'].max():.4f}")
print(f"算术平均提升比: {df['improve_ratio'].mean():.4f}")
print(f"几何平均提升比: {gmean(df.loc[df['improve_ratio'] > 0, 'improve_ratio'] + 1) - 1:.4f}")

# 计算有加速比（>1）的总数及比例
total = len(df)
num_speedup = (df['improve_ratio'] >= 1).sum()
print(f"\n有加速比的比例 (>=1): {num_speedup}/{total} = {num_speedup/total:.4f}")

# 计算 1–1.2 小加速 和 ≥1.2 大加速 的数量及比例
num_small = ((df['improve_ratio'] >= 1) & (df['improve_ratio'] < 1.2)).sum()
num_large = (df['improve_ratio'] >= 1.2).sum()

print(f"1–1.2 倍小加速的比例: {num_small}/{total} = {num_small/total:.4f}")
print(f"≥1.2 倍加速的比例: {num_large}/{total} = {num_large/total:.4f}")