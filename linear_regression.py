# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:09:43 2018
其中：
x∈R^(m×n),y∈R^(1×n),w∈R^((m+1)×1) 包含偏置b


@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def gradient_descent(x,y,epsilon,learningrate=0.01):
    m = np.shape(x)[0] # 特征的个数
    n = np.shape(x)[1] # 样本点的个数
    #将w,b都初始化为零向量和0
    w = np.zeros((m,1)) # 包含偏置b
    #初始的costfunction的值
    cost = costfunction(w,x,y)
    #cost function对于参数w和b的偏导数
    #dw = 2(X*X.T*w + x*y.T - x*y.T)/n
    dw = 2*(np.dot(np.dot(x,x.T),w)-np.dot(x,y.T))/n # m by 1的向量
    k = 0
    while(k < epsilon):
        w = w - learningrate*dw 
        cost = costfunction(w,x,y)
        print(cost)
        k = k + 1
        dw = 2*(np.dot(np.dot(x,x.T),w)-np.dot(x,y.T))/n
    return w


def costfunction(w,x,y):
    y_predict = np.dot(w.T,x)
    j = np.dot((y_predict - y),(y_predict - y).T)
    return j
  
    
def feature_scaling(x,y):
    m = np.shape(x)[0]
    x_max = x.max(axis=1).reshape((m,1)) # 其中x.max(axis=1)返回是一个（m，）的array，
    x_min = x.min(axis=1).reshape((m,1))
    y_max = y.max(axis=1).reshape((1,1))
    y_min = y.min(axis=1).reshape((1,1))
    
    y_ = (y - y_min)/(y_max - y_min)
    x_ = (x - x_min)/(x_max - x_min)
    
    
    return x_,y_


def loadDataSet(fileName):      
    #numFeat = len(open(fileName).readline().split('\t')) - 1 #特征数
    numFeat = len(open(fileName).readline().split(',')) - 1
    x = []; y = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArrX =[]
        lineArrY =[]
        curLine = line.strip().split(',')
        for i in range(numFeat):
            lineArrX.append(float(curLine[i]))
        lineArrY.append(float(curLine[-1]))
        x.append(lineArrX)
        y.append(lineArrY)
    return x,y
    

def train(fileurl):
    file = fileurl
    x_,y_ = loadDataSet(file)
    x = np.array(x_).T
    y = np.array(y_).T
    #进行特征缩放(feature scaling)
    x,y = feature_scaling(x,y)
    
    bias = np.ones((1,x.shape[1])) # 处理偏置项
    x = np.concatenate((x,bias))

    w = gradient_descent(x,y,100) #梯度下降中假设输入都是列向量
    return w
    
   
#def predict():
    

x,y = loadDataSet('ex1data2.txt')
x1 = np.array(x)[:,0]
x2 = np.array(x)[:,1]
y1 = np.array(y)
ax = plt.subplot(111, projection='3d')
ax.scatter(x1, x2, y1, c='y')
ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()

ans = train('ex1data2.txt')
b = ans[-1]
w = ans[:-1]
print('w和b的值为：')
print(w)
print(b)


