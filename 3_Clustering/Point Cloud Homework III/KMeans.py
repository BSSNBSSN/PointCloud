# 文件功能： 实现 K-Means 算法

import numpy as np
import matplotlib.pyplot as plt


def Point_Show(point,color):
    x = []
    y = []
    point = np.asarray(point)
    for i in range(len(point)):
        x.append(point[i][0])
        y.append(point[i][1])
    plt.scatter(x, y,color=color)


    
class K_Means(object):
    # k是分组数；tolerance‘中心点误差’；max_iter是迭代次数
    def __init__(self, n_clusters=2, tolerance=0.0001, max_iter=300):
        self.k_ = n_clusters
        self.tolerance_ = tolerance
        self.max_iter_ = max_iter
        
    def e_step(self, miu, data):
        r = np.zeros([len(data), self.k_], dtype = int)
        for i in range(len(data)):
            min_dis = 99999.999
            miu_idx = int()
            for j in range(self.k_):
                dis = np.linalg.norm(data[i] - miu[j])
                if dis < min_dis:
                    min_dis = dis
                    miu_idx = j
            r[i][miu_idx] = 1
        return r

    def m_step(self, r, data):
        miu = np.empty([self.k_, 2], dtype = float)
        for i in range(self.k_):
            point_num = 0
            point_place = np.zeros([1, 2], dtype = float)
            for j in range(len(data)):
                point_num += r[j][i]
                point_place += r[j][i] * data[j]
            miu[i] = point_place / point_num
        return miu

    def fit(self, data):
        # 作业1
        # 屏蔽开始
        
        miu = data[:self.k_]
        r = np.zeros([len(data), self.k_], dtype = int)
        past_r = np.zeros([len(data), self.k_], dtype = int)
        
        cnt = 0
        for i in range(self.max_iter_):
            past_r = r
            r = self.e_step(miu, data)
            miu = self.m_step(r, data)
            
            if r.all() == past_r.all():
                if cnt >= 3:
                    break
                else:
                    cnt += 1
            else:
                cnt = 0
                    
        return r
        # 屏蔽结束
        
    def predict(self, p_datas):
        result = []
        # 作业2
        # 屏蔽开始

        # 屏蔽结束
        return result

if __name__ == '__main__':
    x = np.array([[1, 2], [1.5, 1.8], [5, 8], [8, 8], [1, 0.6], [9, 11]])
    # a = np.array(np.random.rand(100, 2) * 5)
    # b = np.array(np.random.rand(10, 2) * 5 + 5)
    # x = np.append(a, b, axis = 0)
    k_means = K_Means(n_clusters=2)
    r = k_means.fit(x)
    
    cluster = [[] for i in range(2)]
    for i in range(len(x)):
        for j in range(2):
            if r[i][j] == 1:
                cluster[j].append(x[i])
    Point_Show(cluster[0],"red")
    Point_Show(cluster[1], "blue")
    plt.show()
    print(r)
    # cat = k_means.predict(x)
    # print(cat)

