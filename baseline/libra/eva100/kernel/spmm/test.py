import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# 1. 读取CSV
df = pd.read_csv('/home/shijinliang/module/Libra-sc25/eva100/kernel/spmm/svm.csv')

X = df.drop(['dataSet', 'label'], axis=1).values
y = df['label'].values

# 3. 划分训练集和测试集（注意：不需要提前标准化，因为Pipeline里有Scaler）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. 建立Pipeline与参数网格
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC())
])

param_grid = {
    'svm__C': [0.1, 1, 10, 100],
    'svm__gamma': ['scale', 0.01, 0.1, 1, 10],
    'svm__kernel': ['rbf', 'linear'],
    'svm__class_weight': [None, 'balanced']
}

# 5. 自动调参（5折交叉验证）
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1)
grid.fit(X_train, y_train)
print("最优参数:", grid.best_params_)

# 6. 用最优模型做预测与评估
y_pred = grid.predict(X_test)
print("优化后Accuracy:", accuracy_score(y_test, y_pred))
print("混淆矩阵:\n", confusion_matrix(y_test, y_pred))
print("详细分类报告:\n", classification_report(y_test, y_pred))