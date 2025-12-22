from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn . preprocessing import StandardScaler , MinMaxScaler , OneHotEncoder
from sklearn . model_selection import train_test_split
# Step 1: Load data
iris = load_iris()
# Convert to DataFrame for practice
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['target'] = iris.target
print("Iris dataset shape:", data.shape)
print("\nTarget names:", iris.target_names)

# Step 2: Split data into features (X) and target (y)
# Use all feature columns for X
X = data[iris.feature_names]
# Use 'target' column for y
y = data['target']

# Step 3: Perform train-test split (use 70% training, 30% testing, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Import and instantiate a RandomForestClassifier
# Create the classifier with 100 trees
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Step 5: Fit the model on the training set
model.fit(X_train, y_train)

# Step 6: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 7: Evaluate the model's performance
# Calculate accuracy and print classification report
accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy:.2f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 导入必要的库
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 加载数据
data = pd.read_csv("data/household_power_consumption_1000.txt", sep=";")
X = data.iloc[:, 2:4] # 特征
Y = data.iloc[:, 5] # 目标变量

# 拆分训练集和测试集
trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.2)

# 创建并训练模型
model = LinearRegression()
model.fit(trainX, trainY)

# 预测并评估
y_pred = model.predict(testX)

# 可视化结果
plt.figure()
plt.plot(range(len(y_pred)), y_pred, label="预测值", color="red")
plt.plot(range(len(testY)), testY, label="真实值", color="green")
plt.legend()
plt.show()