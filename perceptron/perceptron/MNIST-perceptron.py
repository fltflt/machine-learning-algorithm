#-*- coding:utf-8 –*-
'''MNIST数据集的多层感知机模型
'''

from __future__ import print_function
import keras
from keras.datasets import mnist#！！！数据集重导入时需要修改
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import RMSprop

#定义抓包大小，类别总数，训练轮数
batch_size = 128
num_classes = 10
epochs = 20

#数据集被分为训练集与测试集
(x_train, y_train), (x_test, y_test) = mnist.load_data()#接口由导入函数实现

x_train = x_train.reshape(60000, 784)#reshape函数来自numpy，将图像的二维存储序列化为一维信息
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')#astype函数来自numpy，将元素数据类型修改为float32
x_test = x_test.astype('float32')
x_train /= 255#训练样本归一化处理（色彩区间为0-255）
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 生成测试数据集与标签之间的二元关系
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

###网络模型搭建
#这里采用线性网络，用sequential函数初始化。有三层网络。网络的叠加通过add函数实现。第一个数字512规定了隐层结点个数，
#relu规定了神经元激活函数。从全连接的神经网络结构看，只需要定义输入结点的接收信息维数即可，最后一行结点的输出向量
#即为输出的特征判别结果的一维二分类表示
#dropout函数原型为keras.layers.core.Dropout(rate, noise_shape=None, seed=None)，用于去掉一定比例的结点防止过学习
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

model.summary()#该函数给出了网络结构信息表

###损失函数定义
#categorical_crossentropy特定用于类别向量中只有一项为1，其它全为0的情况
#RMSprop参数优化方法可以采用默认参数
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

###参数优化过程
#verbose=1即输出训练过程进度条，0即不输出进度
#validation_data即与测试集标签进行比对，得到分类准确率
#score的0，1项分别为最后计算的损失值与正确率
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test))

###性能评估
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])