# 文件功能：实现 GMM 算法

import numpy as np
from numpy import *
import pylab
import random,math

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.stats import multivariate_normal
plt.style.use('seaborn')


def points_show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)

class GMM(object):
    def __init__(self, n_clusters, max_iter=50):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        
        self.mu = None
        self.sigma = None
        self.pi = None
        self.gamma = None
        self.nk = None

        
    # 屏蔽开始
    
    def e_step(self, data):
        # 更新W
        for n in range(len(data)):
            for k in range(self.n_clusters):
                self.gamma[n][k] = self.pi[k] * multivariate_normal.pdf(x = data[n], mean = self.mu[k], cov = self.sigma[k])
        for n in range(len(data)):
            w = 0.0
            for k in range(self.n_clusters):
                w += self.gamma[n][k]
            for k in range(self.n_clusters):
                self.gamma[n][k] /= w
        return 

    def m_step(self, data):
        self.nk = self.gamma.sum(axis = 0)
        # 更新pi
        self.pi = self.nk / len(data)
        
        # 更新Mu
        self.mu.fill(0)
        for k in range(self.n_clusters):   
            for n in range(len(data)):
                self.mu[k] += self.gamma[n][k] * data[n]
            self.mu[k] /= self.nk[k]
            
        # 更新Var
        self.sigma.fill(0)
        for k in range(self.n_clusters):
            for n in range(len(data)):
                var = data[n] - self.mu[k]
                var = var.reshape(1, 2)
                self.sigma[k] += self.gamma[n][k] * np.matmul(var.T, var)
            self.sigma[k] /= self.nk[k]
        return

    # 屏蔽结束
    
    def fit(self, data):
        # 作业3
        # 屏蔽开始
    
        cnt = 0
        self.mu = np.asarray(data[:self.n_clusters])
        self.sigma = np.asarray([eye(2,2)] * self.n_clusters)
        self.pi = np.asarray([1 / self.n_clusters] * self.n_clusters)
        self.gamma = np.zeros((len(data), self.n_clusters), dtype = float)
        past_gamma = np.zeros((len(data), self.n_clusters), dtype = float)
        for i in range(self.max_iter):
            past_gamma = self.gamma
            self.e_step(data)
            self.m_step(data)
            if self.gamma.all() == past_gamma.all():
                if cnt >= 3:
                    break
                else:
                    cnt += 1
            else:
                cnt = 0
                
        return
        # 屏蔽结束
    
    def predict(self, data):
        # 屏蔽开始
        result = []
        result = np.argmax(self.gamma, axis = 1 )    #比较每个点的后验概率
        return result
        # 屏蔽结束
        
    # def plot(self, data):
    #     # visualize:
    #     color = ['red', 'blue', 'green', 'cyan', 'magenta']
    #     labels = [f'Cluster{k:02d}' for k in range(3)]

    #     cluster = []  # 用于分类所有数据点
    #     color = []
    #     for i in range(len(data)):
    #         cluster.append(X[i])
    #         color.append([self.gamma[i][0]*255, self.gamma[i][1]*255, self.gamma[i][2]*255])
            
    
    #     for i in range(len(data)):
    #         points_show(cluster[i], color=color[i])
    #     plt.show()
    #     return

# 生成仿真数据
def generate_X(true_Mu, true_Var):
    # 第一簇的数据
    num1, mu1, var1 = 400, true_Mu[0], true_Var[0]
    X1 = np.random.multivariate_normal(mu1, np.diag(var1), num1)
    # 第二簇的数据
    num2, mu2, var2 = 600, true_Mu[1], true_Var[1]
    X2 = np.random.multivariate_normal(mu2, np.diag(var2), num2)
    # 第三簇的数据
    num3, mu3, var3 = 1000, true_Mu[2], true_Var[2]
    X3 = np.random.multivariate_normal(mu3, np.diag(var3), num3)
    # 合并在一起
    X = np.vstack((X1, X2, X3))
    # 显示数据
    plt.figure(figsize=(10, 8))
    plt.axis([-10, 15, -5, 15])
    plt.scatter(X1[:, 0], X1[:, 1], s=5)
    plt.scatter(X2[:, 0], X2[:, 1], s=5)
    plt.scatter(X3[:, 0], X3[:, 1], s=5)
    plt.show()
    return X

if __name__ == '__main__':
    # 生成数据
    # true_Mu = [[0.5, 0.5], [5.5, 2.5], [1, 7]]
    # true_Var = [[1, 3], [2, 2], [6, 2]]
    
    true_Mu = [[7, 1], [0, 0], [1, 7]]
    true_Var = [[0.5, 3], [2, 2], [3, 0.5]]
    
    X = generate_X(true_Mu, true_Var)

    gmm = GMM(n_clusters=3)
    gmm.fit(X)
    cat = gmm.predict(X)
    print(cat)
    # 初始化
    # gmm.plot(X)

    K = 3
    # visualize:
    color = ['red', 'blue', 'green', 'cyan', 'magenta']
    labels = [f'Cluster{k:02d}' for k in range(K)]

    cluster = [[] for i in range(K)]  # 用于分类所有数据点
    for i in range(len(X)):
        if cat[i] == 0:
            cluster[0].append(X[i])
        elif cat[i] == 1:
            cluster[1].append(X[i])
        elif cat[i] == 2:
            cluster[2].append(X[i])

    points_show(cluster[0], color="red")
    points_show(cluster[1], color="green")
    points_show(cluster[2], color="blue")
    plt.show()