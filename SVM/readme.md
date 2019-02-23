# 支持向量机（SVM）

（csdn博客总结https://blog.csdn.net/qq_39751437/article/details/86521044）

支持向量机（Support Vector Machines）是一种二分类模型，基本思想是在特征空间上寻找可以正确的划分训练数据集且几何间隔最大的分离超平面。**几何间隔最大有利于区别于感知机。这时候分离超平面是唯一的** 

划分的原则是几何间隔最大化，最终转化为一个凸二次规划问题来求解，等价于正则化的合页损失函数最小化问题。包括以下三个模型：

 - 当训练样本**线性可分**时，通过**硬间隔最大化**，学习一个**线性可分支持向量机**； 
 - 当训练样本**近似线性可分**时，通过**软间隔最大化**，学习一个**线性支持向量机**；
 - 当训练样本**线性不可分**时，通过**核技巧和软间隔最大化**，学习一个**非线性支持向量机**；