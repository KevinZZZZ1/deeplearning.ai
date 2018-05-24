# -*- coding: utf-8 -*-
"""
Created on Wed May 16 19:48:16 2018
使用逻辑回归 (logistic regression) 来解决多类别分类问题，具体来说，我想通过一个叫做"一对多" (one-vs-all) 的分类算法
多类别分类器的基本思想：将多类别分类问题转化为多个二值分类问题，然后可以求解出多个预测函数hi(x)，
当有新的x到来时，就可以将其带入所有的预测函数中，计算中max时的i值，即可得到其所在的分类

假设要分类的类的数目为num_label,其实就是设计了num_label个分类器对数据进行二分类，最后在预测时是比较每个分类器给出的是否属于该类的概率进行判断，预测结果为概率大的那个


@author: kevin
"""

import scipy.io as sio  
import matplotlib.pyplot as plt  
import numpy as np

def loaddataset(fileurl):
    data = sio.loadmat(fileurl) # 打开之后是一个字典，即data是一个字典
    x = data['X'].T # data['X'] 是一个 n by m的矩阵，其中n是样本点数，m是每个样本的特征数，data['X'].T是一个m by n的矩阵
    y = data['y']  # data['y'] 是一个 n by 1 的矩阵
    return x,y

def sigmoid(x):
    # 定义的sigmoid函数，非线性
    # 而且sigmoid函数的导数为，sigmoid（x）*(1-sigmoid(x))
    y = 1/(1 + np.exp(-x))
    return y

def costfunction(w,x,y):
    # *表示对应位置每个元素间的乘积，np.dot()表示两个矩阵的乘积,axis = 0(默认)表示横向的，axis = 1表示竖向的
    n = np.shape(x)[1]
    j = np.sum(-y*np.log(sigmoid(np.dot(w.T,x)))-(1-y)*np.log(1-sigmoid(np.dot(w.T,x))))
    return j/n

def sigmoidGradient(w,x,y):
    n = np.shape(x)[1] # 样本点的个数
    dw = (np.dot(x,(sigmoid(np.dot(w.T,x))-y).T))/n
    return dw

def gradient_descent(x,y,epsilon,learningrate=0.001,num_label=10):
    m = np.shape(x)[0] # 特征的个数(已经包括偏置b)
    n = np.shape(x)[1] # 样本点的个数
    #将w,b都初始化为零向量和0
    W = np.zeros((m,num_label)) 
    #初始的costfunction的值

    dw = np.zeros((m,1))

    k=0
    while(k<epsilon):
        for i in range(num_label):
            w = W
            print("第%d次迭代中，第%d个分类器的cost:%f",(k,i+1,))
        
    
def train(fileurl):
    x,y = loaddataset(fileurl)
    
    bias = np.ones((1,x.shape[1])) # 处理偏置项
    x = np.concatenate((x,bias))
     
