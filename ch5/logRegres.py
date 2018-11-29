
# coding: utf-8

# In[16]:


# matplotlib inline


# In[37]:


import time
import numpy as np
import matplotlib.pyplot as plt

# Logistic回归梯度上升哟花算法
def loadDataSet():
    dataMat = []; labelMat = []
#     打开文件，每行前两个值分别是X1 和X2,第三个值是数据对应的标签
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
#         print("lineArr:", lineArr)
#         将X0的值设置为1.0,报错：could not convert string to float: '-'
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
#         print("dataMat:", dataMat)
        labelMat.append(int(lineArr[2]))
#         print("labelMat:", labelMat)
    return dataMat,labelMat

# Sigmoid函数
# 参数 inX
def sigmoid(inX):
    return 1.0/(1 + np.exp(-inX))

# 梯度上升算法核心过程
# 参数 dataMatIn 2维Numpy数组 每列分别代表每个不同的特征，每行代表每个训练样本
# classLabels 类别标签
def gradAscent(dataMatIn, classLabels):
#     获得输入数据并转换为NumPy矩阵
    dataMatrix = np.mat(dataMatIn)
#     print("dataMatrix:",dataMatrix)
#     print(type(dataMatrix))
#     print(type(dataMatIn))
#     将原向量转置并赋值给labelMat
    labelMat = np.mat(classLabels).transpose()
#     print("labelMat",labelMat)
#     得到矩阵大小
    m, n = np.shape(dataMatrix)
#     print("m",m)
#     print("n",n)
#     向目标移动的步长
    alpha = 0.001
#     迭代次数
    maxCycles = 500
#     weights 回归系数
    weights = np.ones((n, 1))
#     print("weights:", weights)
    for k in range(maxCycles):
#         h 为列向量,列向量的元素个数等于样本个数，这里是100
#         dataMatrix * weights 代表不止一次乘积运算，包含了300次乘积
        h = sigmoid(dataMatrix * weights)
#         print("h :", h)
#         计算真实类别与预测类别的差值
        error = (labelMat - h)
#         print("error:", error)
#         矩阵运算，按照差值的方向调整回归系数
        weights = weights + alpha * dataMatrix.transpose() * error
#         print("weights:", weights)
    return weights

# 画出数据集和Logistic回归最佳拟合直线的函数
# 参数 weights 回归系数
def plotBestFit(weights):
    dataMat, labelMat = loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArr[i, 1]); ycord1.append(dataArr[i, 2])
        else:
            xcord2.append(dataArr[i, 1]); ycord2.append(dataArr[i, 2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
#     最佳拟合直线
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()
    
# 随机梯度上升算法
def stocGradAscent0(dataMatrix, classLabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i]*weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

# 改进的随机梯度上升算法
def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m,n = np.shape(dataMatrix)
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
#             alpha 每次迭代时需要调整
            alpha = 4/(1.0 + j + i) +0.01
#             随机选取更新
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

# Logistic 回归分类函数
def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0
    
def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []; trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet), trainingLabels, 700)
    errorCount = 0; numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),trainWeights))!=int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print("the error rate of this test is: %f" % errorRate)
    return errorRate

# 调用函数colicTest()10次并求结果的平均值
def multiTest():
    numTests = 10; errorSum =0.0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f" %(numTests, errorSum/float(numTests)))

if __name__ == '__main__':
    start = time.clock()
#     dataArr, labelMat = loadDataSet()
#     weights = gradAscent(dataArr, labelMat)
#     plotBestFit(weights.getA())
#     weights = stocGradAscent0(np.array(dataArr), labelMat)
#     plotBestFit(weights)
#     weights = stocGradAscent1(np.array(dataArr), labelMat)aaw
#     plotBestFit(weights)
    multiTest()
    end = time.clock()
    print("run time:",end - start)

