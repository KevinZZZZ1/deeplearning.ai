# -*- coding: utf-8 -*-
"""
Created on Thu May 10 11:09:43 2018

@author: kevin
"""
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(x,y,epsilon,learningrate=0.01):
    m = np.shape(x)[0] # 特征的个数
    n = np.shape(x)[1] # 样本点的个数
    #将w,b都初始化为零向量和0
    w = np.zeros((m,1)) # 加1代表偏置b
    #初始的costfunction的值
    cost = costfunction(w,x,y)
    #cost function对于参数w和b的偏导数
    #dw = 2(X*X.T*w + x*y.T - x*y.T)/n
    dw = 2*(np.dot(np.dot(x,x.T),w)-np.dot(x,y.T))/n # m+1 by 1的向量
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
    
def loadDataSet(fileName):      
    #numFeat = len(open(fileName).readline().split('\t')) - 1 #特征数
    numFeat = 1
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
    bias = np.ones((1,x.shape[1])) # 处理偏置项
    x = np.concatenate((x,bias))
    y = np.array(y_).T
    w = gradient_descent(x,y,1000) #梯度下降中假设输入都是列向量
    return w
    
   
#def predict():
    
x,y = loadDataSet('ex1data1.txt')
plt.plot(x,y,'ro')
plt.show()

w,b = train('F:/kevin/ex1data1.txt')
plt.plot(x,y,'ro')
x1 = np.linspace(0,1,23)

def y1(x1):    
    y1 = w*x+b   
    return y1 

plt.plot(x, y1(x1),'g--')
plt.show()

