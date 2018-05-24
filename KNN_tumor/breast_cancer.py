import numpy as np
import pandas as pd
import operator

"""
函数说明:对数据进行归一化

Parameters:
    dataSet - 特征矩阵
Returns:
    normDataSet - 归一化后的特征矩阵
    ranges - 数据范围
    minVals - 数据最小值

"""
def autoNorm(dataSet):
    #获得数据的最小值
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    #最大值和最小值的范围
    ranges = maxVals - minVals
    #shape(dataSet)返回dataSet的矩阵行列数
    normDataSet = np.zeros(np.shape(dataSet))
    #返回dataSet的行数
    m = dataSet.shape[0]
    #原始值减去最小值
    normDataSet = dataSet - np.tile(minVals, (m, 1))
    #除以最大和最小值的差,得到归一化数据
    normDataSet = normDataSet / np.tile(ranges, (m, 1))
    #返回归一化数据结果
    return normDataSet


"""
函数说明:打开并解析文件

Parameters:
    无
Returns:
    train_data - 训练集
    labels - 标签
    test_data - 测试集

"""
def file2matrix():
    X = pd.read_excel("data.xlsx")
    #提取有效数据
    data = X.iloc[93:41361,2:17]
    #去除所有值为nan的行（axis=0）
    data = data.dropna(axis = 0,how = 'all')

    #提取训练数据
    train_data = data.iloc[:,3:17]
    #提取测试数据
    test_data = data.iloc[:,0:3]

    #训练集 
    train_data = train_data.as_matrix()
    train_data = np.transpose(train_data)
    #归一化处理
    train_data = autoNorm(train_data)
    
    #测试集
    test_data = test_data.as_matrix()
    test_data = np.transpose(test_data)
    #归一化处理
    test_data = autoNorm(test_data)

    labels = ['primary tumor','primary tumor','primary tumor',
    'lung-aggressive','lung-aggressive','lung-aggressive',
    'bone-aggressive','bone-aggressive','bone-aggressive',
    'liver-aggressive','liver-aggressive','liver-aggressive']

    return train_data,labels,test_data


"""
函数说明:kNN算法

Parameters:
    inX - 测试集
    dataSet - 训练集
    labels - 分类标签
    k - 选择距离最小的k个点
Returns:
    sortedClassCount[0][0] - 分类结果

"""
def classify0(inX, dataSet, labels, k):
    #dataSet的行数
    dataSetSize = dataSet.shape[0] 
    #在列向量方向上重复inX共1次(横向)
    #行向量方向上重复inX共dataSetSize次(纵向)
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    #二维特征相减后平方
    sqDiffMat = diffMat**2
    #sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = np.sum(sqDiffMat,axis = 1)

    #开方，计算出距离,distances中存储测试集到训练集点的全部距离
    distances = sqDistances**0.5
    #返回distances中元素从小到大排序后的索引值
    sortedDistIndices = distances.argsort()
    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        voteIlabel = labels[sortedDistIndices[i]]
        #计算类别次数
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #key=operator.itemgetter(1)根据字典的值进行排序
    #reverse降序排序字典
    sortedClassCount = sorted(classCount.items(),
        key=operator.itemgetter(1),reverse=True)
    print('距离最小的前 %d 个数据：' % k,sortedClassCount)
    #返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]


"""
函数说明:main函数

Parameters:
    无
Returns:
    无
    
"""
if __name__ == '__main__':
    train_data,labels,test_data = file2matrix()
    for i in range(len(test_data)):
        test_class = classify0(test_data[i,:], train_data, labels, 3)
        print('第 %d 个测试体预测结果：' % (i+1),test_class,'\n')



    