# 文件功能： 实现 K-Means 算法

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph

from KMeans import K_Means

def Point_Show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)
    
class SpectralClustering(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters, max_iter=100, n_neighbors = 20):
        self.k_ = n_clusters
        self.n_neighbors = n_neighbors
        self.max_iter_ = max_iter
        self.KMeans = K_Means(n_clusters , max_iter)
        
    def fit(self, data):
        # 作业1
        # 屏蔽开始
        
        #建立相似矩阵
        weight = kneighbors_graph(data, n_neighbors = self.n_neighbors, mode='connectivity', include_self=False)
        weight = 0.5 * (weight + weight.T)
        W = weight.toarray()
        #计算拉普拉斯算子
        D = np.diag(np.sum(W, axis=1))
        L = D - W
        
        #计算L的特征向量
        w , v = np.linalg.eig(L)
        
        #计算前最小的n个特征向量
        n_idx = np.argsort(w)[:self.k_]
        self.vec = v[:,n_idx]
        
        #对特征向量进行kmeans聚类
        self.KMeans.fit(self.vec)
        self.label = self.KMeans.predict(self.vec)
        # 屏蔽结束
        
    def predict(self, p_datas):
        return self.label
        # 屏蔽结束

if __name__ == '__main__':
    # x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # a = np.array(np.random.rand(100, 2) * 5)
    # b = np.array(np.random.rand(100, 2) * 5 + 5)
    # x = np.append(a, b, axis = 0)
    
    true_Mu = [[7, 1], [0, 0], [1, 7]]
    true_Var = [[0.5, 3], [2, 2], [3, 0.5]]
    
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
    x = np.vstack((X1, X2, X3))
    
    spectral_clustering = SpectralClustering(n_clusters=3)
    r = spectral_clustering.fit(x)
    
    cluster = [[] for i in range(3)]
    for i in range(len(x)):
        for j in range(3):
            if r[i][j] == 1:
                cluster[j].append(x[i])
    Point_Show(cluster[0],"red")
    Point_Show(cluster[1], "green")
    Point_Show(cluster[2], "blue")
    plt.show()
    # print(r)
    # cat = spectral_clustering.predict(x)
    # print(cat)

