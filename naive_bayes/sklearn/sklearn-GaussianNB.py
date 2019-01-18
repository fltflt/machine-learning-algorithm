import numpy as np
from sklearn.naive_bayes import GaussianNB
X = np.array([[-1, -1], [-2, -2], [-3, -3],[-4,-4],[-5,-5], [1, 1], [2,2], [3, 3]])
y = np.array([1, 1, 1, 1, 1, 2, 2, 2])
clf = GaussianNB(priors=[0.625, 0.375])#默认priors=None
clf.fit(X, y, sample_weight=None)#训练样本，X表示特征向量，y类标记，sample_weight表各样本权重数组
print (clf.class_prior_)#priors属性：获取各个类标记对应的先验概率
print (clf.priors)#class_prior_属性：同priors一样，
print (clf.class_count_)#class_count_属性：获取各类标记对应的训练样本数
print (clf.theta_)#theta_属性：获取各个类标记在各个特征上的均值
print (clf.sigma_)#sigma_属性：获取各个类标记在各个特征上的方差
print (clf.get_params(deep=True))#get_params(deep=True)：返回priors与其参数值组成字典
clf.set_params(priors=[ 0.6,  0.4])#set_params(**params)：设置估计器priors参数
print (clf.get_params(deep=True))
print (clf.predict([[-6,-6],[4,5]]))#预测样本分类
print (clf.predict_proba([[-6,-6],[4,5]]))#predict_proba(X)：输出测试样本在各个类标记预测概率值
print (clf.predict_log_proba([[-6,-6],[4,5]]))#predict_log_proba(X)：输出测试样本在各个类标记上预测概率值对应对数值
print (clf.score([[-6,-6],[-4,-2],[-3,-4],[4,5]],[1,1,2,2])) #score(X, y, sample_weight=None)：返回测试样本映射到指定类标记上的平均得分(准确率)

# output:
# [0.625 0.375]
# [0.625, 0.375]
# [5. 3.]
# [[-3. -3.]
#  [ 2.  2.]]
# [[2.00000001 2.00000001]
#  [0.66666667 0.66666667]]
# {'priors': [0.625, 0.375]}
# {'priors': [0.6, 0.4]}
# [1 2]
# [[1.00000000e+00 3.29099984e-40]
#  [5.13191647e-09 9.99999995e-01]]
# [[ 0.00000000e+00 -9.09122123e+01]
#  [-1.90877867e+01 -5.13191623e-09]]
# 0.75