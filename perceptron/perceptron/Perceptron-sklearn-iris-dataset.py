from sklearn import datasets
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.cross_validation import train_test_split

iris = datasets.load_iris() # 加载Iris数据集。
X = iris.data[:, [2, 3]]
y = iris.target # 标签已经转换成0，1，2了
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0) # 为了看模型在没有见过数据集上的表现，随机拿出数据集中30%的部分做测试

# 为了追求机器学习和最优化算法的最佳性能，我们将特征缩放
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(X_train) # 估算每个特征的平均值和标准差
sc.mean_ # 查看特征的平均值，由于Iris我们只用了两个特征，结果是array([ 3.82857143,  1.22666667])
sc.scale_ # 查看特征的标准差，结果是array([ 1.79595918,  0.77769705])
X_train_std = sc.transform(X_train)
# 注意：这里我们要用同样的参数来标准化测试集，使得测试集和训练集之间有可比性
X_test_std = sc.transform(X_test)

# 训练感知机模型
from sklearn.linear_model import Perceptron
# n_iter：可以理解成梯度下降中迭代的次数
# eta0：可以理解成梯度下降中的学习率
# random_state：设置随机种子的，为了每次迭代都有相同的训练集顺序
ppn = Perceptron(n_iter=40, eta0=0.2, random_state=0)
ppn.fit(X_train_std, y_train)

# 分类测试集，这将返回一个测试结果的数组
y_pred = ppn.predict(X_test_std)
# 计算模型在测试集上的准确性
print (y_pred)
print(ppn.coef_)
print (ppn.n_iter_)
print (ppn.intercept_)
print (accuracy_score(y_test, y_pred))
print (ppn.score(X_test_std,y_test))
