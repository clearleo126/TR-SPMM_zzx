import pandas as pd
import numpy as np

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/hybird/fp16_cuda_tcu_bitmap_128_hybird_filter_new1.csv')

# 计算加速比
df['sp'] = df['rtgnn2'] / df['min_all']

# 仅保留 sp >= 1 的行
df_filtered = df[df['sp'] > 1]

# 统计数量
total = len(df_filtered)
print(f"满足 sp ≥ 1 的矩阵个数：{total}")

# 区间划分
range_counts = {
    '<1': len(df_filtered[df_filtered['sp'] < 1]),  # 实际为 0
    '1-1.2': len(df_filtered[(df_filtered['sp'] >= 1) & (df_filtered['sp'] < 1.2)]),
    '1.2-1.5': len(df_filtered[(df_filtered['sp'] >= 1.2) & (df_filtered['sp'] <= 1.5)]),
    '>1.5': len(df_filtered[df_filtered['sp'] > 1.5]),
}

# 计算百分比
percentages = {k: round(v / total * 100, 1) for k, v in range_counts.items()}

# 最大值
max_sp = df_filtered['sp'].max()

# 算术平均加速比
mean_sp = df_filtered['sp'].mean()

# 几何平均加速比（加一个小 epsilon 防止 log(0) 错误）
epsilon = 1e-8
geo_mean_sp = np.exp(np.mean(np.log(df_filtered['sp'] + epsilon)))

# 输出
print("\n范围统计（sp ≥ 1）:")
for k in ['<1', '1-1.2', '1.2-1.5', '>1.5']:
    print(f"{k}: {range_counts[k]} 个，占比 {percentages[k]}%")

print(f"\n最大加速比: {round(max_sp, 4)}")
print(f"算术平均加速比: {round(mean_sp, 4)}")
print(f"几何平均加速比: {round(geo_mean_sp, 4)}")