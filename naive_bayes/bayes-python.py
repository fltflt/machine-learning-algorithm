#机器学习实战源代码python3 朴素贝叶斯垃圾邮件分类、文本分类
import numpy as np
import re
import random


def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0,1,0,1,0,1]    #1 is abusive, 0 not
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet|set(document)# 符号| 用于取两个集合的并集
    return list(vocabSet)



# def createVocabList(dataSet):
#     vocabSet = set([])  #create empty set
#     for document in dataSet:
#         vocabSet = vocabSet | set(document) #union of the two sets
#     return list(vocabSet)


def setOfWords2Vec(vocabList, inputSet):
    # 创建一个其中所含元素都为0的向量
    returnVec = [0] * len(vocabList)
    # 遍历每个词条
    for word in inputSet:
        if word in vocabList:
            # 如果词条存在于词汇表中，则置1
            # index返回word出现在vocabList中的索引
            # 若这里改为+=则就是基于词袋的模型，遇到一个单词会增加单词向量中德对应值
            returnVec[vocabList.index(word)] = 1
        else:
            print("the word: %s is not in my Vocabulary" % word)
    # 返回文档向量
    return returnVec


# postingList,classVec=loadDataSet()
# vocabSet=createVocabList(postingList)
# print (vocabSet)
# returnVec2=setOfWords2Vec(vocabSet,['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'])
# print ('returnVec2 is',returnVec2)

# trainMate=[]
# for i in postingList:
#     trainMate.append(setOfWords2Vec(vocabSet,i))
# print (trainMate)
# numTrainDocs = len(trainMate)
# numWords = len(trainMate[0])
# print (numWords,numTrainDocs)

def trainNB0(trainMatrix, trainCategory):
    # 计算训练文档数目,共6个
    numTrainDocs = len(trainMatrix)
    # 计算每篇文档的词条数目，共32个单词，词汇表大小
    numWords = len(trainMatrix[0])
    # 文档属于侮辱类的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    # 创建numpy.zeros数组，词条出现数初始化为0
    # p0Num = np.zeros(numWords)
    # p1Num = np.zeros(numWords)
    # 创建numpy.ones数组，词条出现数初始化为1,拉普拉斯平滑
    p0Num = np.ones(numWords)
    p1Num = np.ones(numWords)
    # 分母初始化为0
    # p0Denom = 0.0
    # p1Denom = 0.0
    # 分母初始化为2，拉普拉斯平滑，二分类问题，分母初始化为类别数
    p0Denom = 2.0
    p1Denom = 2.0
    for i in range(numTrainDocs):
        # 统计属于侮辱类的条件概率所需的数据，即P(w0|1),P(w1|1),P(w2|1)...
        if trainCategory[i] == 1:
            # 统计所有侮辱类文档中每个单词出现的个数，1为侮辱性文字
            p1Num += trainMatrix[i]
            # 统计一共出现的侮辱单词的个数
            p1Denom += sum(trainMatrix[i])
        # 统计属于非侮辱类的条件概率所需的数据，即P(w0|0),P(w1|0),P(w2|0)...
        else:
            # 统计所有非侮辱类文档中每个单词出现的个数
            p0Num += trainMatrix[i]
            # 统计一共出现的非侮辱单词的个数
            p0Denom += sum(trainMatrix[i])
    # 每个侮辱类单词分别出现的概率
    # p1Vect = p1Num / p1Denom
    # 取对数，防止下溢出
    p1Vect = np.log(p1Num / p1Denom)
    # 每个非侮辱类单词分别出现的概率
    # p0Vect = p0Num / p0Denom
    # 取对数，防止下溢出
    p0Vect = np.log(p0Num / p0Denom)
    # 返回属于侮辱类的条件概率数组、属于非侮辱类的条件概率数组、文档属于侮辱类的概率
    return p0Vect, p1Vect, pAbusive
    
# p0Vect, p1Vect, pAbusive=trainNB0(trainMate, classVec)
# print (p0Vect, p1Vect, pAbusive)

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    # 对应元素相乘
    # p1 = reduce(lambda x,y:x*y, vec2Classify * p1Vec) * pClass1
    # p0 = reduce(lambda x,y:x*y, vec2Classify * p0Vec) * (1.0 - pClass1)
    # 对应元素相乘，logA*B = logA + logB所以这里是累加
    # vec2Classify为要分类的向量，p0Vec, p1Vec, pClass1分别对应p0Vect, p1Vect, pAbusive
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    # print('p0:', p0)
    # print('p1:', p1)
    if p1 > p0:
        return 1
    else:
        return 0
    
def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec


# returnVec1=bagOfWords2VecMN(vocabSet,['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'])
# print (print ('returnVec1 is',returnVec2))

def testingNB():
    # 创建实验样本
    listOPosts, listclasses = loadDataSet()
    # 创建词汇表,将输入文本中的不重复的单词进行提取组成单词向量
    myVocabList = createVocabList(listOPosts)
    trainMat = []
    for postinDoc in listOPosts:
        # 将实验样本向量化若postinDoc中的单词在myVocabList出现则将returnVec该位置的索引置1
        # 将6组数据list存储在trainMat中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    # 训练朴素贝叶斯分类器
    p0V, p1V, pAb = trainNB0(np.array(trainMat), np.array(listclasses))
    # 测试样本1
    testEntry = ['love', 'my', 'dalmation']
    # 测试样本向量化返回这三个单词出现位置的索引
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        # 执行分类并打印结果
        print(testEntry, '属于侮辱类')
    else:
        # 执行分类并打印结果
        print(testEntry, '属于非侮辱类')
    # 测试样本2
    testEntry = ['stupid', 'garbage']
    # 将实验样本向量化
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    if classifyNB(thisDoc, p0V, p1V, pAb):
        # 执行分类并打印结果
        print(testEntry, '属于侮辱类')
    else:
        # 执行分类并打印结果
        print(testEntry, '属于非侮辱类')

# testingNB()    
#接受一个大字符串并将其解析为字符串列表
def textParse(bigString):
    # 用特殊符号作为切分标志进行字符串切分，即非字母、非数字
    # \W* 0个或多个非字母数字或下划线字符（等价于[^a-zA-Z0-9_]）
    listOfTockens = re.split(r'\W*', bigString)
    # 除了单个字母，例如大写I，其他单词变成小写，去掉少于两个字符的字符串,low()变小写，upper()变大写
    return [tok.lower() for tok in listOfTockens if len(tok) > 2]
# print (textParse('my dog has flea problems help please 2 33 take dog ed'))
    
def spamTest():
    docList = []
    classList = []
    fullText = []
    # 遍历25个txt文件
    for i in range(1, 26):
        # 读取每个垃圾邮件,并以字符串转换成字符串列表
        wordList = textParse(open('D:\\Sublime Text 3\\python\\email\\spam\\%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记垃圾邮件，1表示垃圾文件
        classList.append(1)
        # 读取每个非垃圾邮件，并以字符串转换成字符串列表
        wordList = textParse(open('D:\\Sublime Text 3\\python\\email\\ham\\%d.txt' % i, 'r').read())
        docList.append(wordList)
        fullText.append(wordList)
        # 标记非垃圾邮件，0表示非垃圾文件
        classList.append(0)
    # 创建词汇表，不重复
    vocabList = createVocabList(docList)
    # 创建存储训练集的索引值的列表和测试集的索引值的列表
    trainingSet = list(range(50))
    testSet = []
    # 从50个邮件中，随机挑选出40个作为训练集，10个作为测试集,留存交叉验证
    for i in range(10):
        # 随机选取索引值,随机生成一个实数
        randIndex = int(random.uniform(0, len(trainingSet)))
        # 添加测试集的索引值
        testSet.append(trainingSet[randIndex])
        # 在训练集列表中删除添加到测试集的索引值
        del(trainingSet[randIndex])
    # 创建训练集矩阵和训练集类别标签向量
    trainMat = []
    trainClasses = []
    # 遍历训练集
    for docIndex in trainingSet:
        # 将生成的词集模型添加到训练集矩阵中
        trainMat.append(setOfWords2Vec(vocabList, docList[docIndex]))
        # 将类别添加到训练集类别标签向量中
        trainClasses.append(classList[docIndex])
    # 训练朴素贝叶斯模型
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # 错误分类计数
    errorCount = 0
    # 遍历测试集
    for docIndex in testSet:
        # 测试集的词集模型
        wordVector = setOfWords2Vec(vocabList, docList[docIndex])
        # 如果分类错误
        if classifyNB(np.array(wordVector), p0V, p1V, pSpam) != classList[docIndex]:
            # 错误计数器加1
            errorCount += 1
            print("分类错误的测试集：", docList[docIndex])
    print("错误率：%.2f%%" % (float(errorCount) / len(testSet) * 100))
        

if __name__ == '__main__':
    spamTest()
