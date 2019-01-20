##决策树源代码 python3 
from math import log
import operator

#创建数据集
def createDataSet():
    dataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    #change to discrete values
    return dataSet, labels



#计算给定数据集香农熵，总体思路，统计不同分类占总数的百分比，然后利用信息熵公式来计算
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)#计算数据集大小
    #print (numEntries)
    labelCounts = {}
    #{'yes': 1}{'yes': 2}{'no': 1, 'yes': 2}{'no': 2, 'yes': 2}{'no': 3, 'yes': 2}
    for featVec in dataSet: #遍历数据集
        currentLabel = featVec[-1]#最后一列
        #print (currentLabel)
        if currentLabel not in labelCounts.keys(): 
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key])/numEntries#每一个类别所占百分比
        shannonEnt -= prob * log(prob,2)#香农公式
    #log base 2
    return shannonEnt




#按照数据集划分特征，dataSet, axis, value 分别代表待划分的数据集，划分的特征，需要返回特征的值
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet
dataSet, labels=createDataSet()
shannonEnt=calcShannonEnt(dataSet)




##选择最好的数据集划分方式，返回此特征，根据信息增益最大原则，计算每一个特征的信息增益，选择信息增益最大的特征进行划分
def chooseBestFeatureToSplit(dataSet):
    numFeatures = len(dataSet[0]) - 1      #the last column is used for the labels
    baseEntropy = calcShannonEnt(dataSet) 
    bestInfoGain = 0.0; bestFeature = -1
    for i in range(numFeatures):        #iterate over all the features
        featList = [example[i] for example in dataSet]#create a list of all the examples of this feature
        uniqueVals = set(featList)       #将所有特征筛选出来
        newEntropy = 0.0
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet)/float(len(dataSet))
            newEntropy += prob * calcShannonEnt(subDataSet)   #条件熵，每一类别的熵乘以所占的比重
        infoGain = baseEntropy - newEntropy     ##信息增益等于经验熵-经验条件熵
        if (infoGain > bestInfoGain):       #比较哪一个信息增益最大
            bestInfoGain = infoGain         #if better than current best, set to best
            bestFeature = i
    return bestFeature                      #returns an integer

bestFeature=chooseBestFeatureToSplit(dataSet)
print(bestFeature)#最好的特征是第一个特征

def majorityCnt(classList):
    classCount={}
    for vote in classList:
        if vote not in classCount.keys(): classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

## 创建树的代码，返回树
def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    if classList.count(classList[0]) == len(classList): 
        return classList[0] #类别相同则停止划分
    if len(dataSet[0]) == 1: #stop splitting when there are no more features in dataSet
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        subLabels = labels[:]       #copy all of labels, so trees don't mess up existing labels
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet, bestFeat, value),subLabels)
    return myTree                            
myTree=createTree(dataSet,labels)
print (myTree) 

##决策树的预测
def classify(inputTree,featLabels,testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict): 
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else: classLabel = valueOfFeat
    return classLabel

dataSet, labels=createDataSet()
classLabel=classify(myTree,labels,[1,0])
print (classLabel)


## 决策树的储存
def storeTree(inputTree,filename):
    import pickle
    fw = open(filename,'wb')
    pickle.dump(inputTree,fw)
    fw.close()
myTree=createTree(dataSet,labels)
storeTree(myTree,'myTree_save.txt')  

def grabTree(filename):
    import pickle
    fr = open(filename,'rb')
    return pickle.load(fr)
print (grabTree('myTree_save.txt'))

