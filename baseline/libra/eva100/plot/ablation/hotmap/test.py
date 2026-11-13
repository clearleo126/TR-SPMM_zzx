import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
# # 创建示例数据
# data = {
#     'Category': ['A', 'B', 'C', 'D'],
#     'Group1': [10, 20, 15, 25],
#     'Group2': [15, 25, 20, 30]
# }

# # 将数据转换为DataFrame
# df = pd.DataFrame(data)
# num_rows = df.shape[0]
# # 使用Seaborn绘制叠状柱形图
# sns.barplot(x='Category', y='Group1', data=df, color='blue', label='Group 1')
# sns.barplot(x='Category', y='Group2', data=df, color='orange', label='Group 2', bottom=df['Group1'])

# # 添加图例
# plt.legend()

# # 显示图形
# plt.savefig('eva100/plot/ablation/tcu_cuda_density_spmm_fp16_128.png', dpi=800)
# # 清空图形
# plt.clf()
import seaborn as sns
import matplotlib.pyplot as plt

import seaborn as sns
import matplotlib.pyplot as plt

# 创建示例数据
x = [1.0, 2.0, 3.0, 4.0, 5.0]
y = [3, 5, 2, 7, 4]
x1 = [1.0, 2.0]
# 绘制折线图
sns.lineplot(x=x, y=y)

# 绘制折线

# 指定 x 区间进行填充
start_x = 2
end_x = 4
plt.fill_betweenx(x1, color='skyblue', alpha=0.3)


 




# 显示图形
plt.savefig('eva100/plot/ablation/tcu_cuda_density_spmm_fp16_128.png', dpi=800)
# 清空图形
plt.clf()

