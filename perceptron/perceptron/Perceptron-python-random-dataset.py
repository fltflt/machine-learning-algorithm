# PLA算法

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:\\Users\\Administrator\\Desktop\\data1.csv', header=None)
# 样本输入，维度（100，2）
X = data.iloc[:,:2].values
# 样本输出，维度（100，）
y = data.iloc[:,2].values
print (X,y)

plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='Negative')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.title('原始数据')
plt.show()

# 均值
u = np.mean(X, axis=0)
# 方差
v = np.std(X, axis=0)
X = (X - u) / v

# 作图
plt.scatter(X[:50, 0], X[:50, 1], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 0], X[50:, 1], color='red', marker='x', label='Negative')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.title('Normalization data')
plt.show()

# 直线初始化
# X加上偏置项，
X = np.hstack((np.ones((X.shape[0],1)), X))
# 权重初始化
w = np.random.randn(3,1)

for i in range(100):
    s = np.dot(X, w)
    y_pred = np.ones_like(y)#返回一个用1填充的跟输入形状和类型一致的数组。
    loc_n = np.where(s < 0)[0]#两种用法1.只有条件 (condition)，没有x和y，则输出满足条件 (即非0) 元素的坐标 2.np.where(condition, x, y)满足条件(condition)，输出x，不满足输出y。
    y_pred[loc_n] = -1
    num_fault = len(np.where(y != y_pred)[0])
    print('第%2d次更新，分类错误的点个数：%2d' % (i, num_fault))
    if num_fault == 0:
        break
    else:
        t = np.where(y != y_pred)[0][0]
        w += y[t] * X[t, :].reshape((3,1))
print (w)
# 直线第一个坐标（x1，y1）
x1 = -2
y1 = -1 / w[2] * (w[0] * 1 + w[1] * x1)
# 直线第二个坐标（x2，y2）
x2 = 2
y2 = -1 / w[2] * (w[0] * 1 + w[1] * x2)
# 作图
plt.scatter(X[:50, 1], X[:50, 2], color='blue', marker='o', label='Positive')
plt.scatter(X[50:, 1], X[50:, 2], color='red', marker='x', label='Negative')
plt.plot([x1,x2], [y1,y2],'r')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(loc = 'upper left')
plt.show()

# 第 0次更新，分类错误的点个数：98
# 第 1次更新，分类错误的点个数：14
# 第 2次更新，分类错误的点个数：37
# 第 3次更新，分类错误的点个数： 1
# 第 4次更新，分类错误的点个数： 5
# 第 5次更新，分类错误的点个数： 0