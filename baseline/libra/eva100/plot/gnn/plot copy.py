import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# 示例数据
data = {
    'Category': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C'],
    'Type': ['Libra', 'Other1', 'Other2', 'Libra', 'Other1', 'Other2', 'Libra', 'Other1', 'Other2'],
    'Value': [1.2, 1.5, 1.8, 2.3, 2.6, 2.1, 3.4, 3.7, 3.2]
}

# 创建DataFrame
df = pd.DataFrame(data)

# 绘制多组柱状图
plt.figure(figsize=(10, 6))
sns.barplot(x='Category', y='Value', hue='Type', data=df)

# 添加标题和标签
plt.title('Complex Multiple Group Barplot')
plt.xlabel('Category')
plt.ylabel('Value')

plt.savefig('/home/shijinliang/module/Libra/eva100/plot/gnn/gcn.png', dpi=800)
