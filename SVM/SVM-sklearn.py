from sklearn import tree
import numpy as np
import pandas as pd
import pydotplus 
from sklearn.metrics import accuracy_score
from sklearn import datasets,model_selection
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score,cross_val_predict,cross_validate
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_curve
from sklearn.metrics import auc   
from sklearn.metrics import roc_auc_score

iris=datasets.load_iris() # scikit-learn 自带的 iris 数据集
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=0)

# clf=svm.LinearSVC(penalty='l2', loss='squared_hinge', tol=0.0001, max_iter=1000).fit(X_train,y_train)
# clf=svm.NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None).fit(X_train,y_train)
clf=svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=False, tol=0.001, cache_size=200, class_weight=None, verbose=False, max_iter=-1, decision_function_shape='ovr', random_state=None).fit(X_train,y_train)

print (accuracy_score(y_test,y_pred))
print (recall_score(y_test, y_pred, average='macro'))
print (recall_score(y_test, y_pred, average='weighted'))
print (recall_score(y_test, y_pred, average=None))#none 返回每一类别的召回率
print (confusion_matrix(y_test, y_pred))#混淆矩阵
print (classification_report(y_test, y_pred))#F1分数，很好平衡了召回率和精准率，让二者同时达到最高，取一个平衡。
# print(clf.coef_)
print(clf.intercept_)
print(clf.support_vectors_)
print(clf.n_support_)
print (clf.get_params())#获取模型参数






