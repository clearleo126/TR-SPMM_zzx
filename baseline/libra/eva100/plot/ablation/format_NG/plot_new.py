import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
types = ['SGT', 'METCF', 'SRBCRS', 'ours']
# time = [4702.0596, 19.306, 16.614, 17.7]  
time = [23.0596, 19.306, 16.614, 17.7]  
mem = [1.28, 0.85,  2.27, 0.80]           

# 设置图形大小
fig, ax = plt.subplots(figsize=(8, 6))

colors = ['cornflowerblue', 'lightcoral', 'lightgreen', 'gold']
patterns = ['/', 'x', '\\', '|-']
for i in range(len(types)):
    ax.bar(types[i], time[i], color=colors[i], hatch=patterns[i], edgecolor='black', linewidth=1, width=0.5)

# 添加图例
ax.legend()

# 添加标题和标签
ax.set_ylabel('Relative expression level')
ax.set_title('Gene Expression Comparison')

# 显示图表
plt.show()


plt.savefig('/home/shijinliang/module/Libra/eva100/plot/ablation/format_NG/plot_new.png', dpi=800)
# 清空图形
plt.clf()
