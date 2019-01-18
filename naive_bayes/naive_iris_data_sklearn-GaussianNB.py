from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn import datasets
iris = datasets.load_iris()
gnb = GaussianNB()
scores=cross_val_score(gnb, iris.data, iris.target, cv=10)
print (scores)
print("Accuracy:%.3f"%scores.mean())