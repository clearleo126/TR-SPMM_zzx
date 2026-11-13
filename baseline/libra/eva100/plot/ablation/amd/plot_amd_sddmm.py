import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gmean
# 读取数据
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/amd/sddmm_with_hybird_cuda_tcu.csv')

# 筛除非正数（防止报错）
libra_valid = df['speedup']


# 计算几何平均
geo_avg_libra = gmean(libra_valid)

# 最大值
max_libra = libra_valid.max()

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

libra_speedup_grouped = group_average(df['speedup'].values)

x = group_average(df['log_edges'].values)  # log(num_edges)

# 绘图
plt.figure(figsize=(7, 2))
plt.scatter(x, libra_speedup_grouped,  color='green', marker='o', s=20)
plt.axhline(y=1, color='red', linestyle='--', linewidth=2)

plt.grid(axis='y', linestyle='--', alpha=0.4)

# 布局与保存
plt.tight_layout()
plt.savefig('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/amd/sddmm_amd.png', dpi=800)
plt.show()