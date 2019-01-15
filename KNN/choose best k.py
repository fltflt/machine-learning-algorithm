#coding=utf-8
def main():
    from sklearn import datasets
    digits=datasets.load_digits()
    x=digits.data
    y=digits.target
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=666)
    from sklearn.neighbors import KNeighborsClassifier
    # 寻找最好的k
    best_k=-1
    best_score=0
    for i in range(1,11):
        knn_clf=KNeighborsClassifier(n_neighbors=i)
        knn_clf.fit(x_train,y_train)
        scores=knn_clf.score(x_test,y_test)
        if scores>best_score:
            best_score=scores
            best_k=i
    print('最好的k为:%d,最好的得分为:%.4f'%(best_k,best_score))
if __name__ == '__main__':
    main()
