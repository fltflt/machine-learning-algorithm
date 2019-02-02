import numpy as np
from sklearn.linear_model import LinearRegression#线性回归

import matplotlib.pyplot as plt#用于作图
x = [[1], [1], [2], [2]]
y = [1, 2, 2, 3]
# y = np.dot(X, np.array([1, 2])) + 3
reg = LinearRegression().fit(x, y)### y = 1 * x_0 + 2 * x_1 + 3
print (x,y)
y1 = 1 * x + 0.5
plt.scatter(x,y1)
plt.show()

print (reg.score(x, y))
print (reg.intercept_)##截距
print (reg.coef_)##系数