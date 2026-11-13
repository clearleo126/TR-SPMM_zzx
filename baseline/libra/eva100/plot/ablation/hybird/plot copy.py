import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean

# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/hybird/fp16_cuda_tcu_bitmap_128_hybird_filter_new1.csv')

# 1. 计算运算量
df['flops'] = df['num_edges'] * 128 * 2  # FLOP 数
time_to_sec = 1.0  # 时间单位换算（秒／秒）

# 2. 计算四种实现的 GFLOPS（注意这里除以1e6，单位是 kGFLOPS）
df['gflops_hc']      = df['flops'] / (df['hc-spmm2'] * time_to_sec) / 1e6
df['gflops_libra']   = df['flops'] / (df['min_all']   * time_to_sec) / 1e6
df['gflops_pcgcn']   = df['flops'] / (df['pcgcn2']    * time_to_sec) / 1e6
# 新增：RT-GNN 实现
df['gflops_rtgnn2']  = df['flops'] / (df['rtgnn2']    * time_to_sec) / 1e6

# 3. 按边数排序并准备横轴
df = df.sort_values(by='num_edges').reset_index(drop=True)
df['log_edges'] = np.log10(df['num_edges'] + 1)

# 4. 每 3 个点取平均的辅助函数
def group_average(arr, group_size=3):
    return [np.mean(arr[i:i+group_size]) 
            for i in range(0, len(arr), group_size)]

g =3
# 5. 分组平均
x_grouped         = group_average(df['log_edges'].values,    group_size=g)
hc_grouped        = group_average(df['gflops_hc'].values,    group_size=g)
libra_grouped     = group_average(df['gflops_libra'].values, group_size=g)
pcgcn_grouped     = group_average(df['gflops_pcgcn'].values, group_size=g)
rtgnn2_grouped    = group_average(df['gflops_rtgnn2'].values,group_size=g)
libra_grouped = [min(v, 8000) for v in libra_grouped]
# 6. 绘制分组后的散点图
fig, ax = plt.subplots(figsize=(9, 2.5))

# ax.scatter(num_edges_str, libra_G_tf32_avg, color='blue', s=12)
ax.scatter(x_grouped, pcgcn_grouped,
            label='PC-GCN (baseline)',
            s=12,  color='lightgreen')
ax.scatter(x_grouped, rtgnn2_grouped,
            label='RT-GNN',
             s=12, color='plum')
ax.scatter(x_grouped, hc_grouped,
            label='HC-SpMM',
            s=12, color='tomato')
ax.scatter(x_grouped, libra_grouped,
            label='Libra',
            s=12, color='blue')

ax.spines['top'].set_linewidth(0.5)  # 设置顶部边框宽度
ax.spines['right'].set_linewidth(0.5)  # 设置右边框宽度
ax.spines['left'].set_linewidth(0.5)  # 设置左边框宽度
ax.spines['bottom'].set_linewidth(0.5)  # 设置底部边框宽度

ax.grid(True, which='both', linestyle='--', linewidth=0.5)


plt.xlabel('log₁₀(Number of Edges + 1)', fontsize=12)
plt.ylabel('Performance (kGFLOPS)', fontsize=12)
plt.title('Binned GFLOPS vs. Matrix Size', fontsize=14)

# 保存和显示
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/hybird/gflops_scatter_binned.png', dpi=800)
plt.show()