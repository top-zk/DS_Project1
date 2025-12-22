import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')


class MLHelper:
    def __init__(self):
        self.numpy_functions = {
            'array_creation': self.numpy_array_creation,
            'array_operations': self.numpy_array_operations,
            'indexing_slicing': self.numpy_indexing_slicing,
            'linear_algebra': self.numpy_linear_algebra,
            'statistics': self.numpy_statistics
        }

        self.sklearn_functions = {
            'data_loading': self.sklearn_data_loading,
            'preprocessing': self.sklearn_preprocessing,
            'train_test_split': self.sklearn_train_test_split,
            'classification': self.sklearn_classification,
            'clustering': self.sklearn_clustering
        }

    def help(self, topic=None):
        """主帮助函数"""
        if topic is None:
            self.show_all_topics()
        elif topic.startswith('numpy'):
            self.handle_numpy_query(topic)
        elif topic.startswith('sklearn'):
            self.handle_sklearn_query(topic)
        else:
            print("请指定'numpy'或'sklearn'主题")

    def show_all_topics(self):
        """显示所有可用主题"""
        print("=" * 50)
        print("NumPy 主题:")
        print("=" * 50)
        for key in self.numpy_functions.keys():
            print(f"- numpy_{key}")

        print("\n" + "=" * 50)
        print("Scikit-learn 主题:")
        print("=" * 50)
        for key in self.sklearn_functions.keys():
            print(f"- sklearn_{key}")

        print("\n使用方式: helper.help('numpy_array_creation')")

    def handle_numpy_query(self, topic):
        """处理NumPy查询"""
        topic_key = topic.replace('numpy_', '')
        if topic_key in self.numpy_functions:
            self.numpy_functions[topic_key]()
        else:
            print(f"未找到主题: {topic}")
            print("可用的NumPy主题:")
            for key in self.numpy_functions.keys():
                print(f"- numpy_{key}")

    def handle_sklearn_query(self, topic):
        """处理Scikit-learn查询"""
        topic_key = topic.replace('sklearn_', '')
        if topic_key in self.sklearn_functions:
            self.sklearn_functions[topic_key]()
        else:
            print(f"未找到主题: {topic}")
            print("可用的Scikit-learn主题:")
            for key in self.sklearn_functions.keys():
                print(f"- sklearn_{key}")

    # ===== NumPy 函数 =====
    def numpy_array_creation(self):
        """NumPy数组创建"""
        print("=" * 60)
        print("NumPy 数组创建方法")
        print("=" * 60)

        # 从列表创建
        print("1. 从列表创建数组:")
        list_data = [1, 2, 3, 4, 5]
        arr_from_list = np.array(list_data)
        print(f"   np.array({list_data}) = {arr_from_list}")
        print(f"   形状: {arr_from_list.shape}, 数据类型: {arr_from_list.dtype}")

        # 创建特殊数组
        print("\n2. 创建特殊数组:")
        zeros_arr = np.zeros((2, 3))
        ones_arr = np.ones((2, 2))
        range_arr = np.arange(0, 10, 2)
        print(f"   np.zeros((2, 3)):\n{zeros_arr}")
        print(f"   np.ones((2, 2)):\n{ones_arr}")
        print(f"   np.arange(0, 10, 2): {range_arr}")

        # 随机数组
        print("\n3. 随机数组:")
        random_arr = np.random.rand(3, 2)
        print(f"   np.random.rand(3, 2):\n{random_arr}")

    def numpy_array_operations(self):
        """NumPy数组操作"""
        print("=" * 60)
        print("NumPy 数组操作")
        print("=" * 60)

        # 创建示例数组
        arr1 = np.array([[1, 2, 3], [4, 5, 6]])
        arr2 = np.array([[2, 2, 2], [1, 1, 1]])

        print("示例数组:")
        print(f"arr1:\n{arr1}")
        print(f"arr2:\n{arr2}")

        # 数学运算
        print("\n1. 数学运算:")
        print(f"加法: arr1 + arr2 =\n{arr1 + arr2}")
        print(f"乘法: arr1 * 2 =\n{arr1 * 2}")
        print(f"矩阵乘法 (dot): {np.dot([1, 2, 3], [4, 5, 6])}")

        # 数组方法
        print("\n2. 数组方法:")
        print(f"形状重塑: arr1.reshape(3, 2) =\n{arr1.reshape(3, 2)}")
        print(f"转置: arr1.T =\n{arr1.T}")
        print(f"展平: arr1.flatten() = {arr1.flatten()}")

    def numpy_indexing_slicing(self):
        """NumPy索引和切片"""
        print("=" * 60)
        print("NumPy 索引和切片")
        print("=" * 60)

        arr = np.array([[1, 2, 3, 4],
                        [5, 6, 7, 8],
                        [9, 10, 11, 12]])

        print(f"示例数组:\n{arr}")
        print(f"形状: {arr.shape}")

        print("\n1. 基本索引:")
        print(f"arr[0, 1] = {arr[0, 1]}")  # 第0行第1列
        print(f"arr[1] = {arr[1]}")  # 第1行

        print("\n2. 切片:")
        print(f"arr[0:2, 1:3] (前2行, 第1-2列):\n{arr[0:2, 1:3]}")
        print(f"arr[:, 2] (所有行的第2列): {arr[:, 2]}")

        print("\n3. 布尔索引:")
        bool_mask = arr > 5
        print(f"布尔掩码 (arr > 5):\n{bool_mask}")
        print(f"arr[arr > 5] = {arr[arr > 5]}")

    def numpy_linear_algebra(self):
        """NumPy线性代数"""
        print("=" * 60)
        print("NumPy 线性代数操作")
        print("=" * 60)

        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])

        print(f"矩阵 A:\n{A}")
        print(f"矩阵 B:\n{B}")

        print("\n1. 矩阵运算:")
        print(f"矩阵乘法 A @ B:\n{A @ B}")
        print(f"矩阵乘法 np.matmul(A, B):\n{np.matmul(A, B)}")

        print("\n2. 矩阵属性:")
        print(f"A 的迹: {np.trace(A)}")
        print(f"A 的行列式: {np.linalg.det(A):.2f}")
        print(f"A 的逆矩阵:\n{np.linalg.inv(A)}")

        print("\n3. 特征值和特征向量:")
        eigenvalues, eigenvectors = np.linalg.eig(A)
        print(f"特征值: {eigenvalues}")
        print(f"特征向量:\n{eigenvectors}")

    def numpy_statistics(self):
        """NumPy统计函数"""
        print("=" * 60)
        print("NumPy 统计函数")
        print("=" * 60)

        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        matrix = np.random.rand(3, 4) * 10

        print(f"示例数据: {data}")
        print(f"示例矩阵:\n{matrix}")

        print("\n1. 基本统计:")
        print(f"平均值: {np.mean(data):.2f}")
        print(f"中位数: {np.median(data)}")
        print(f"标准差: {np.std(data):.2f}")
        print(f"方差: {np.var(data):.2f}")

        print("\n2. 矩阵统计 (沿轴):")
        print(f"每列的平均值: {np.mean(matrix, axis=0)}")
        print(f"每行的最大值: {np.max(matrix, axis=1)}")
        print(f"矩阵总和: {np.sum(matrix):.2f}")

        print("\n3. 其他统计:")
        print(f"百分位数 (25%, 50%, 75%): {np.percentile(data, [25, 50, 75])}")
        print(f"相关性矩阵:\n{np.corrcoef(matrix)}")

    # ===== Scikit-learn 函数 =====
    def sklearn_data_loading(self):
        """Scikit-learn数据加载"""
        print("=" * 60)
        print("Scikit-learn 数据加载")
        print("=" * 60)

        print("1. 内置数据集:")

        # 鸢尾花数据集
        iris = datasets.load_iris()
        print(f"鸢尾花数据集:")
        print(f"  特征形状: {iris.data.shape}")
        print(f"  目标形状: {iris.target.shape}")
        print(f"  特征名称: {iris.feature_names}")
        print(f"  目标名称: {iris.target_names}")
        print(f"  类别数量: {len(np.unique(iris.target))}")

        # 手写数字数据集
        digits = datasets.load_digits()
        print(f"\n手写数字数据集:")
        print(f"  特征形状: {digits.data.shape}")
        print(f"  图像形状: {digits.images[0].shape}")
        print(f"  类别数量: {len(np.unique(digits.target))}")

        print("\n2. 生成数据集:")
        X, y = datasets.make_classification(n_samples=100, n_features=4,
                                            n_informative=2, n_redundant=0,
                                            random_state=42)
        print(f"生成分类数据集:")
        print(f"  特征形状: {X.shape}")
        print(f"  目标形状: {y.shape}")

    def sklearn_preprocessing(self):
        """Scikit-learn数据预处理"""
        print("=" * 60)
        print("Scikit-learn 数据预处理")
        print("=" * 60)

        # 创建示例数据
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]], dtype=float)
        print(f"原始数据:\n{X}")

        print("\n1. 标准化 (StandardScaler):")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        print(f"标准化后:\n{X_scaled}")
        print(f"均值: {scaler.mean_}")
        print(f"标准差: {scaler.scale_}")

        print("\n2. 数据分割示例:")
        X_train, X_test, y_train, y_test = train_test_split(
            X, [0, 1, 0, 1], test_size=0.25, random_state=42
        )
        print(f"训练集形状: {X_train.shape}")
        print(f"测试集形状: {X_test.shape}")

    def sklearn_train_test_split(self):
        """训练测试集分割"""
        print("=" * 60)
        print("Scikit-learn 训练测试集分割")
        print("=" * 60)

        # 加载数据
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        print(f"原始数据形状: {X.shape}")
        print(f"目标数据形状: {y.shape}")

        print("\n1. 基本分割:")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        print(f"训练集: {X_train.shape} ({(X_train.shape[0] / X.shape[0]) * 100:.1f}%)")
        print(f"测试集: {X_test.shape} ({(X_test.shape[0] / X.shape[0]) * 100:.1f}%)")

        print("\n2. 分层分割 (保持类别比例):")
        X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
            X, y, test_size=0.3, stratify=y, random_state=42
        )

        print("原始数据类别分布:")
        unique, counts = np.unique(y, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"  类别 {cls}: {count}样本")

        print("分层分割后训练集类别分布:")
        unique_train, counts_train = np.unique(y_train_strat, return_counts=True)
        for cls, count in zip(unique_train, counts_train):
            print(f"  类别 {cls}: {count}样本")

    def sklearn_classification(self):
        """Scikit-learn分类算法"""
        print("=" * 60)
        print("Scikit-learn 分类算法")
        print("=" * 60)

        # 加载数据
        iris = datasets.load_iris()
        X, y = iris.data, iris.target

        # 分割数据
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        # 标准化
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print("1. 逻辑回归分类器:")
        # 创建和训练模型
        log_reg = LogisticRegression(random_state=42)
        log_reg.fit(X_train_scaled, y_train)

        # 预测
        y_pred = log_reg.predict(X_test_scaled)
        y_pred_proba = log_reg.predict_proba(X_test_scaled)

        # 评估
        accuracy = accuracy_score(y_test, y_pred)
        print(f"   准确率: {accuracy:.3f}")
        print(f"   预测概率示例:\n{y_pred_proba[:3]}")

        print("\n2. 分类报告:")
        print(classification_report(y_test, y_pred, target_names=iris.target_names))

        print("3. 模型参数:")
        print(f"   系数形状: {log_reg.coef_.shape}")
        print(f"   截距: {log_reg.intercept_}")

    def sklearn_clustering(self):
        """Scikit-learn聚类算法"""
        print("=" * 60)
        print("Scikit-learn 聚类算法")
        print("=" * 60)

        # 生成示例数据
        X, y_true = datasets.make_blobs(n_samples=300, centers=3,
                                        cluster_std=0.60, random_state=42)

        print(f"生成数据形状: {X.shape}")
        print(f"真实类别数: {len(np.unique(y_true))}")

        print("\n1. K-means聚类:")
        # 应用K-means
        kmeans = KMeans(n_clusters=3, random_state=42)
        y_pred = kmeans.fit_predict(X)

        print(f"   聚类中心:\n{kmeans.cluster_centers_}")
        print(f"   惯性 (Within-cluster sum of squares): {kmeans.inertia_:.2f}")
        print(f"   迭代次数: {kmeans.n_iter_}")

        print("\n2. 聚类结果分析:")
        from sklearn.metrics import adjusted_rand_score
        ari = adjusted_rand_score(y_true, y_pred)
        print(f"   调整兰德指数: {ari:.3f}")

        # 显示每个聚类的样本数
        unique, counts = np.unique(y_pred, return_counts=True)
        for cluster, count in zip(unique, counts):
            print(f"   聚类 {cluster}: {count}个样本")


def main():
    """主函数"""
    helper = MLHelper()

    print("🤖 NumPy 和 Scikit-learn 学习助手")
    print("=" * 50)

    while True:
        print("\n请输入您想了解的主题:")
        print("1. 输入 'list' 查看所有主题")
        print("2. 输入主题名称 (如: numpy_array_creation)")
        print("3. 输入 'quit' 退出")

        user_input = input("\n请输入: ").strip().lower()

        if user_input == 'quit':
            print("再见！")
            break
        elif user_input == 'list':
            helper.show_all_topics()
        elif user_input:
            helper.help(user_input)
        else:
            print("请输入有效命令")


# 示例用法
if __name__ == "__main__":
    # 创建助手实例
    helper = MLHelper()

    # 示例：查看所有主题
    print("🔍 查看所有可用主题:")
    helper.show_all_topics()

    print("\n" + "=" * 70)
    print("📚 示例解析:")
    print("=" * 70)

    # 运行一些示例
    helper.numpy_array_creation()
    helper.sklearn_classification()

    # 交互式模式
    print("\n" + "=" * 70)
    print("💬 交互模式:")
    print("=" * 70)
    main()