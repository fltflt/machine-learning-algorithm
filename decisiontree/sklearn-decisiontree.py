from sklearn import tree
import numpy as np
import pydotplus 
from sklearn import datasets,model_selection
from sklearn.metrics import accuracy_score
import os 
from IPython.display import Image       
os.environ['PATH'] += os.pathsep + 'D:\\Graphviz2.38\\bin\\'
import matplotlib.pyplot as plt
iris=datasets.load_iris() # scikit-learn 自带的 iris 数据集
X=iris.data
y=iris.target
X_train,X_test,y_train,y_test=model_selection.train_test_split(X,y,test_size=0.3,random_state=0)
clf = tree.DecisionTreeClassifier(criterion='gini')
clf.fit(X_train, y_train)

# 画出决策树的可视化结构，以pdf展现出来
with open("iris.dot", 'w') as f:
    f = tree.export_graphviz(clf, out_file=f)

dot_data = tree.export_graphviz(clf, out_file=None, 
                         feature_names=iris.feature_names,  
                         class_names=iris.target_names,  
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  

graph.write_pdf("C:\\Users\\Administrator\\Desktop\\iris.pdf")


print ('CART testing accuracy on iris datasets',clf.score(X_test,y_test))
print ('CART training accuracy on iris datasets',clf.score(X_train,y_train))



## 画出决策树随深度变化的training_score and test_score
maxdepth = 40

depths=np.arange(1,maxdepth)
training_scores=[]
testing_scores=[]
for depth in depths:
    clf = tree.DecisionTreeClassifier(max_depth=depth)
    clf.fit(X_train, y_train)
    training_scores.append(clf.score(X_train,y_train))
    testing_scores.append(clf.score(X_test,y_test))

## 绘图
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(depths,training_scores,label="traing score",marker='o')
ax.plot(depths,testing_scores,label="testing score",marker='*')
ax.set_xlabel("maxdepth")
ax.set_ylabel("score")
ax.set_title("Decision Tree Classification")
ax.legend(framealpha=0.5,loc='best')
plt.show()
