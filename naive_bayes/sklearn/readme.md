## sklearn-GaussianNB.py 为sklearn中高斯分布模型，用于连续特征

## sklearn-BernoulliNB.py 为sklearn中伯努利型模型，用于离散特征

## sklearn-MultinomialNB.py 为sklearn中多项式模型，用于离散特征

## sklearn_iris_data_sklearn-GaussianNB 为在iris_data采用不同多项式模型，高斯分布模型代码

## sklearn_RSSdata.py 为在RSS data 上采用MultinomialNB模型与GaussianNB模型代码
### 结果对比
### MultinomialNB()

Building prefix dict from the default dictionary ...

Loading model from cache C:\Users\ADMINI~1\AppData\Local\Temp\jieba.cache

Loading model cost 1.681 seconds.

Prefix dict has been built succesfully.

当删掉前450个高频词分类精度为：0.51806

[Finished in 15.6s]

### GaussianNB()

Building prefix dict from the default dictionary ...

Loading model from cache C:\Users\ADMINI~1\AppData\Local\Temp\jieba.cache

Loading model cost 1.921 seconds.

Prefix dict has been built succesfully.

当删掉前450个高频词分类精度为：0.63055

[Finished in 9.0s]
