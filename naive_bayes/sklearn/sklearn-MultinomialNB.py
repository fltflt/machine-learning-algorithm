import numpy as np
from sklearn.naive_bayes import MultinomialNB
X = np.array([[1,2,3,4],[1,3,4,4],[2,4,5,5],[2,5,6,5],[3,4,5,6],[3,5,6,6]])
y = np.array([1,1,4,2,3,3])
clf = MultinomialNB(alpha=1, class_prior=None, fit_prior=False)
clf.fit(X, y, sample_weight=None)#训练样本，X表示特征向量，y类标记，sample_weight表各样本权重数组
print(clf.class_log_prior_)
#class_log_prior_：各类标记的平滑先验概率对数值，其取值会受fit_prior和class_prior参数的影响,三种情况
#若指定了class_prior参数，不管fit_prior为True或False，class_log_prior_取值是class_prior转换成log后的结果
#若fit_prior参数为False，class_prior=None，则各类标记的先验概率相同等于类标记总个数N分之一
#若fit_prior参数为True，class_prior=None，则各类标记的先验概率相同等于各类标记个数除以各类标记个数之和
print (clf.class_count_)#class_count_属性：获取各类标记对应的训练样本数
print (clf.feature_count_)#：各类别各个特征出现的次数，返回形状为(n_classes, n_features)数组)
print (clf.get_params(deep=True))#get_params(deep=True)：返回priors与其参数值组成字典
print (clf.predict_log_proba([[3,4,5,4],[1,3,5,6]]))#predict_log_proba(X)：输出测试样本在各个类标记上预测概率值对应对数值
print (clf.predict_proba([[3,4,5,4],[1,3,5,6]]))#predict_proba(X)：输出测试样本在各个类标记预测概率值
print (clf.score([[3,4,5,4],[1,3,5,6]],[1,1]))#score(X, y, sample_weight=None)：输出对测试样本的预测准确率的平均值
clf.set_params(alpha=2.0)#set_params(**params)：设置估计器参数
print (clf.get_params(deep=True))

# output:
# [-1.38629436 -1.38629436 -1.38629436 -1.38629436]
# [2. 1. 2. 1.]
# [[ 2.  5.  7.  8.]
#  [ 2.  5.  6.  5.]
#  [ 6.  9. 11. 12.]
#  [ 2.  4.  5.  5.]]
# {'fit_prior': False, 'class_prior': None, 'alpha': 1}
# [[-1.70084964 -1.31750168 -1.29059819 -1.29257843]
#  [-1.00382273 -1.59845908 -1.58396998 -1.48652445]]
# [[0.18252837 0.26780353 0.27510617 0.27456193]
#  [0.36647582 0.20220787 0.205159   0.22615731]]
# 0.5
# {'fit_prior': False, 'class_prior': None, 'alpha': 2.0}
