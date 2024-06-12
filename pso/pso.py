# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import math

def getweight():
    # 惯性权重
    weight = 1
    return weight

def getlearningrate():
    # 分别是粒子的个体和社会的学习因子，也称为加速常数
    lr = (0.49445,1.49445)
    return lr

def getmaxgen():
    # 最大迭代次数
    maxgen = 300
    return maxgen

def getsizepop(size):
    # 种群规模
    sizepop = size
    return sizepop

def getrangepop():
    # 粒子的位置的范围限制,x、y方向的限制相同
    rangepop = (0 , 20)
    #rangepop = (-2,2)
    return rangepop

def getrangespeed():
    # 粒子的速度范围限制
    rangespeed = (-0.5,0.5)
    return rangespeed

def func(x):
    # x输入粒子位置
    # y 粒子适应度值
    if (x[0]==0)&(x[1]==0):
        y = np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    else:
        y = np.sin(np.sqrt(x[0]**2+x[1]**2))/np.sqrt(x[0]**2+x[1]**2)+np.exp((np.cos(2*np.pi*x[0])+np.cos(2*np.pi*x[1]))/2)-2.71289
    return y

def initpopvfit(sizepop,rangepop):
    pop = np.zeros((sizepop,1))
    v = np.zeros((sizepop,1))
    fitness = np.zeros(sizepop)

    for i in range(sizepop):
        # pop[i] = [(np.random.rand()-0.5)*rangepop[0]*2,(np.random.rand()-0.5)*rangepop[1]*2]
        # v[i] = [(np.random.rand()-0.5)*rangepop[0]*2,(np.random.rand()-0.5)*rangepop[1]*2]
        pop[i] = [(np.random.rand())*rangepop[1]]
        v[i] = [(np.random.rand())*rangepop[1]]
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()

    return gbestpop,gbestfitness,pbestpop,pbestfitness
