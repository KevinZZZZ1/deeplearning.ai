# -*- coding: utf-8 -*-
"""
Created on Tue May 15 16:17:31 2018

斜率没有什么问题，但是偏置b始终存在问题，而且没找到 = =
补充：b不对的问题好像找到了，问题似乎是出在前期数据处理时进行的特征缩放，把特征缩放去掉之后，经过100000次的迭代得到了正确的解，至于原因还没弄清楚
@author: keivn
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import scipy.optimize as op


def sigmoid(x):
    # 定义的sigmoid函数，非线性
    # 而且sigmoid函数的导数为，sigmoid（x）*(1-sigmoid(x))
    y = 1/(1 + np.exp(-x))
    return y


def sigmoidGradient(w,x,y):
    n = np.shape(x)[1] # 样本点的个数
    dw = (np.dot(x,(sigmoid(np.dot(w.T,x))-y).T))/n
    return dw

def feature_scaling(x,y):
    #进行特征缩放
    m = np.shape(x)[0]
    x_max = x.max(axis=1).reshape((m,1)) # 其中x.max(axis=1)返回是一个（m，）的array，
    x_min = x.min(axis=1).reshape((m,1))
    y_max = y.max(axis=1).reshape((1,1))
    y_min = y.min(axis=1).reshape((1,1))
    
    y_ = (y - y_min)/(y_max - y_min)
    x_ = (x - x_min)/(x_max - x_min)
    
    
    return x_,y_




def gradient_descent(x,y,epsilon,learningrate=0.001):
    m = np.shape(x)[0] # 特征的个数(已经包括偏置b)
    n = np.shape(x)[1] # 样本点的个数
    #将w,b都初始化为零向量和0
    w = np.zeros((m,1)) 
    #初始的costfunction的值
    cost = costfunction(w,x,y)
    print(cost)
    dw = np.zeros((m,1))
    #cost function对于参数w和b的偏导数
    #dw = (np.dot(x,(sigmoid(np.dot(w.T,x))-y).T))/n
    #for i in range(n):
    #   dw = (sigmoid(np.dot(w.T,x[:,i].reshape(m,1)))-y[:,i])*x[:,i].reshape(m,1)
    #dw = dw / n
    #print("1  dw:")
    #print(dw)
    dw = (np.dot(x,(sigmoid(np.dot(w.T,x))-y).T))/n
    k = 0
    while(k < epsilon):
        w = w - learningrate*dw 
        cost = costfunction(w,x,y)
        print('w:')
        print(w)
        print('dw:')
        print(dw)
        print('cost:')
        print(cost)
        k = k + 1
        dw = (np.dot(x,(sigmoid(np.dot(w.T,x))-y).T))/n
        #dw = (np.dot(x,(y*(1-sigmoid(np.dot(w.T,x)))).T) + np.dot(x,((1-y)*sigmoid(np.dot(w.T,x))).T))/n
    return w



def costfunction(w,x,y):
    # *表示对应位置每个元素间的乘积，np.dot()表示两个矩阵的乘积,axis = 0(默认)表示横向的，axis = 1表示竖向的
    n = np.shape(x)[1]
    j = np.sum(-y*np.log(sigmoid(np.dot(w.T,x)))-(1-y)*np.log(1-sigmoid(np.dot(w.T,x))))
    return j/n

def loadDataSet(fileName):      
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
    bias = np.ones((1,x.shape[1])) # 处理偏置项
    x = np.concatenate((x,bias))
    
    w = gradient_descent(x,y,100000)
    return w

    
x,y = loadDataSet('ex2data1.txt')
m = len(x)
for i in range(m):
    if(y[i][0]==0.0): plt.plot(x[i][0],x[i][1],'ro')
    if(y[i][0]==1.0): plt.plot(x[i][0],x[i][1],'go')
plt.show()

w1,w2,b1 = train('ex2data1.txt')
w = -w1/w2
b = -b1/w2
x1 = np.linspace(30,100,90)  
for i in range(m):
    if(y[i][0]==0.0): plt.plot(x[i][0],x[i][1],'ro')
    if(y[i][0]==1.0): plt.plot(x[i][0],x[i][1],'go')
def y1(x1):    
    y1 = w*x1+b
    return y1 
plt.plot(x1, y1(x1), 'r-',linewidth=1,label='f(x)') 
plt.show()  

X = np.array(x).T
Y = np.array(y).T
bias = np.ones((1,X.shape[1])) # 处理偏置项
X = np.concatenate((bias,X))
test_theta = np.array([[-24], [0.2], [0.2]])
cost = costfunction(test_theta, X, Y)
grad = sigmoidGradient(test_theta, X, Y)
print('Cost at test theta: {}'.format(cost))
print('Expected cost (approx): 0.218')
print('Gradient at test theta: {}'.format(grad))
print('Expected gradients (approx): 0.043 2.566 2.647')
options = {'full_output': True, 'maxiter': 400}
initial_theta = np.zeros((X.shape[0],1))
theta, cost, _, _, _ = op.fmin(lambda t: costfunction(t, X, Y), initial_theta, **options)

