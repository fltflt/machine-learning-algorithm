# 逻辑斯谛回归

逻辑斯谛回归模型（Logistic regression）是对数线性模型，经典的分类方法。逻辑回归假设数据服从伯努利分布,通过极大化似然函数的方法，运用梯度下降来求解参数，来达到将数据二分类的目的。csdn博客总结https://blog.csdn.net/qq_39751437/article/details/86709231


## （1）算法思路

给定输入实例x，分别利用二项逻辑斯谛回归模型计算P(Y=1|x)与P(Y=0|x)，比较两个概率值的大小，将x分到概率较大的那一类。

## （2）算法特点

通过逻辑斯谛回归模型的定义式 **P(Y=1|x)** 可以将线性函数$\omega$*x转换为概率值，线性函数值$\omega$*x越接近正无穷，概率值越接近为1，线性函数越接近负无穷，概率值越接近为0。

## （3）代码简介(python3下)

Logistic_疝气病马预测_python预测.py 文件是用python写的疝气病马预测，训练集为horseColicTraining，测试集为horseColicTest

Logistic_疝气病马预测_sklearn预测.py 文件是调用sklearn逻辑回归API LogisticRegression 实现疝气病马预测，训练集为horseColicTraining，测试集为horseColicTest

logistic_sklearn.py 文件是LinearRegression线性回归与逻辑回归LogisticRegression sklearn
参数详解与实例，数据集为iris

两种梯度计算方法对比python.py 文件是两种梯度计算方法对比，一种为随机梯度上升算法（学习率变化，且一次使用一个样本更新参数），一种是梯度上升（学习率不变，且一次使用全部样本更新参数）

梯度上升决策边界（使用全部数据集）.png 为使用梯度上升画出来的决策边界

随机梯度上升决策边界（一次使用一个样本）.png 为使用随机梯度上升画出来的决策边界

梯度上升算法，回归系数与迭代次数关系.png  为梯度上升算法与随机梯度上升算法与回归系数与迭代次数关系 

