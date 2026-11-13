import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/hybird/fp16_cuda_tcu_bitmap_128_hybird_filter_new1.csv')

df['max_spmm'] = df[['spmm_cuda', 'spmm_tcu']].max(axis=1)
# 计算加速比
df['hc_speedup'] = df['hc-spmm2'] / df['rtgnn2']
df['libra_speedup'] = df['hc-spmm2'] / df['min_all']
df['pcgcn_speedup']   = df['hc-spmm2'] / df['pcgcn2']
# df['hc_speedup']    = df['hc_speedup'].clip(upper=6)  
# df['libra_speedup'] = df['libra_speedup'].clip(upper=6)  

# 筛除非正数（防止报错）
hc_valid = df['hc_speedup'][df['hc_speedup'] > 0]
libra_valid = df['libra_speedup'][df['libra_speedup'] > 0]
libra_valid = df['pcgcn_speedup'][df['pcgcn_speedup'] > 0]

# 计算几何平均
geo_avg_hc = gmean(hc_valid)
geo_avg_libra = gmean(libra_valid)

# 最大值
max_hc = hc_valid.max()
max_libra = libra_valid.max()

# 打印结果
print("HC-SpMM 加速比:")
print("  几何平均:", round(geo_avg_hc, 3))
print("  最大值:", round(max_hc, 3))

print("Libra 加速比:")
print("  几何平均:", round(geo_avg_libra, 3))
print("  最大值:", round(max_libra, 3))

# 按 num_edges 升序排序
df = df.sort_values(by='num_edges').reset_index(drop=True)

# 加一个小常数防止 log(0)
df['log_edges'] = np.log10(df['num_edges'] + 1)

# 每两个点合并：按2个为一组，取平均
def group_average(arr):
    return [np.mean(arr[i:i+2]) for i in range(0, len(arr), 2)]

hc_speedup_grouped = group_average(df['hc_speedup'].values)
libra_speedup_grouped = group_average(df['libra_speedup'].values)
pcgcn_speedup_grouped = group_average(df['pcgcn_speedup'].values)

x = group_average(df['log_edges'].values)  # log(num_edges)

# 绘图
plt.figure(figsize=(7, 2.5))
plt.axhline(y=1, color='red', linestyle='--', linewidth=2)
plt.scatter(x, pcgcn_speedup_grouped,  color='green', marker='o',s=15)
plt.scatter(x, hc_speedup_grouped,  color='darkorange', marker='o',s=15)
plt.scatter(x, libra_speedup_grouped,  color='blue', marker='^', s=20)


plt.ylabel('Speedup over PC-GCN', fontsize=12)
plt.xlabel('Matrix Group (sorted by num_edges)', fontsize=12)
# plt.xticks([])  # 不显示 x 轴标签
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.legend(loc='upper left', fontsize=10)

# 布局与保存
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/hybird/speedup_vs_pcgnc.png', dpi=800)
plt.show()