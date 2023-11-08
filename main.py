import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score

# 1. 读取CSV文件，假设您有n个CSV文件
#csv_files = ["file1.csv", "file2.csv", "file3.csv", ...]  # 用实际文件名替换这些

# 2. 加载数据并进行预处理
#dataframes = [pd.read_csv(file) for file in csv_files]
test = pd.read_csv('C:\\Users\\51004\\Desktop\\MergeCSV\\Dennis\\gaze_merged.csv')

# 3. 合并数据
#data = pd.concat(dataframes)
print(test)

# 4. 根据需要合并多行 (x, y, z) 到一个特征向量
data["feature_vector"] = data.groupby("p")[["x", "y", "z"]].transform(lambda x: x.values.tolist())

# 5. 创建特征向量和目标
X = data["feature_vector"].tolist()
y = data["p"]

# 6. 初始化Random Forest分类器
rf_classifier = RandomForestClassifier()

# 7. 定义K-Fold交叉验证
kfold = KFold(n_splits=n)  # 将n替换为所需的折数

# 8. 进行K-Fold交叉验证
accuracies = []
for train_index, test_index in kfold.split(X):
    X_train, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # 训练模型
    rf_classifier.fit(X_train, y_train)

    # 进行预测
    y_pred = rf_classifier.predict(X_test)

    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    accuracies.append(accuracy)

# 9. 打印每次交叉验证的准确率
for i, accuracy in enumerate(accuracies):
    print(f"Fold {i+1} Accuracy: {accuracy}")

# 10. 打印平均准确率
average_accuracy = np.mean(accuracies)
print(f"Average Accuracy: {average_accuracy}")