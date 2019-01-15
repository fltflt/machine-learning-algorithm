from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
x,y = make_classification(n_samples=1200, n_features=2,n_redundant=0,n_informative=1,n_clusters_per_class=1)
x_data_train = x[:800,:]
x_data_test = x[800:,:]
y_data_train = y[:800]
y_data_test = y[800:]

#定义感知机
clf = Perceptron(fit_intercept=False,shuffle=False,eta0=0.1,random_state=0)
#使用训练数据进行训练
clf.fit(x_data_train,y_data_train)
print(clf.coef_)
print (clf.n_iter_)
print (clf.intercept_)
# y_pred=clf.predict(x_data_test)
# acc = clf.score(x_data_test,y_data_test)
# print(acc)
# print (accuracy_score(y_data_test, y_pred))
# classify_report = classification_report(y_data_test, y_pred)
# print('classify_report : \n', classify_report)