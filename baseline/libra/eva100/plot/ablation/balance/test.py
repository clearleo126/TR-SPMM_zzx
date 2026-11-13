import pandas as pd

# 读取CSV文件
df = pd.read_csv('/home/shijinliang/module/ppopp26-Libra/eva100/plot/ablation/balance/result.csv')

# spmm表示多流执行时间，spmm_seq表示串行执行时间
# 计算加速比：加速比 = 较慢时间 / 较快时间
# 这里以spmm_seq除以spmm为加速比，代表多流相对于串行的加速比
df['speedup_spmm'] = df['spmm_no'] / df['spmm']
df['speedup_sddmm'] = df['sddmm_no'] / df['sddmm']

# 找出执行更快的矩阵
spmm_faster = df[df['spmm'] <= df['spmm_no']]
sddmm_faster = df[df['sddmm'] < df['sddmm_no']]

def analyze_speedup(data, col_name):
    total = len(data)
    between_1_1_2 = data[(data[col_name] >= 1) & (data[col_name] < 1.2)].shape[0]
    above_1_2 = data[data[col_name] >= 1.2].shape[0]
    mean_speedup = data[col_name].mean()
    max_speedup = data[col_name].max()
    min_speedup = data[col_name].min()
    print(f"总矩阵数: {total}")
    print(f"1x ~ 1.2x 加速矩阵数: {between_1_1_2} ({between_1_1_2/total*100:.2f}%)")
    print(f">= 1.2x 加速矩阵数: {above_1_2} ({above_1_2/total*100:.2f}%)")
    print(f"平均加速比: {mean_speedup:.2f}x")
    print(f"最大加速比: {max_speedup:.2f}x")
    print(f"最小加速比: {min_speedup:.2f}x\n")

print("SpMM:")
analyze_speedup(spmm_faster, 'speedup_spmm')

print("SDDMM:")
analyze_speedup(sddmm_faster, 'speedup_sddmm')