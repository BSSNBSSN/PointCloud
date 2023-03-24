# 实现PCA分析和法向量计算，并加载数据集中的文件进行验证

import open3d as o3d 
import os
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame
from pyntcloud import PyntCloud

# matplotlib显示点云函数
def Point_Cloud_Show(points):
    fig = plt.figure(dpi=150)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], cmap='spectral', s=2, linewidths=0, alpha=1, marker=".")
    plt.title('Point Cloud')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()


# 二维点云显示函数
def Point_Show(pca_point_cloud):
    x = []
    y = []
    pca_point_cloud = np.asarray(pca_point_cloud)
    for i in range(10000):
        x.append(pca_point_cloud[i][0])
        y.append(pca_point_cloud[i][1])
    plt.scatter(x, y)
    plt.show()


# 功能：计算PCA的函数
# 输入：
#     data：点云，NX3的矩阵
#     correlation：区分np的cov和corrcoef，不输入时默认为False
#     sort: 特征值排序，排序是为了其他功能方便使用，不输入时默认为True
# 输出：
#     eigenvalues：特征值
#     eigenvectors：特征向量
def PCA(data, correlation=False, sort=True):
    # 作业1
    # 屏蔽开始
    average_data = np.mean(data,axis=0)                         #求 NX3 向量的均值
    decentration_matrix = data - average_data                   #去中心化
    H = np.dot(decentration_matrix.T,decentration_matrix)       #求解协方差矩阵 H
    eigenvectors,eigenvalues,eigenvectors_T = np.linalg.svd(H)  # SVD求解特征值、特征向量
    # 屏蔽结束

    if sort:
        sort = eigenvalues.argsort()[::-1]      #降序排列
        eigenvalues = eigenvalues[sort]         #索引
        eigenvectors = eigenvectors[:, sort]

    return eigenvalues, eigenvectors


def main():
    # 指定点云路径
    # cat_index = 10 # 物体编号，范围是0-39，即对应数据集中40个物体
    # root_dir = '/Users/renqian/cloud_lesson/ModelNet40/ply_data_points' # 数据集路径
    # cat = os.listdir(root_dir)
    # filename = os.path.join(root_dir, cat[cat_index],'train', cat[cat_index]+'_0001.ply') # 默认使用第一个点云

    # 加载原始点云，txt处理
    point_cloud_raw = np.genfromtxt(r"C:\\Users\\15803\\Desktop\\PointCloudCourse\\1_PCA_SurfaceNomal_Filter\\dataset\\piano_0001.txt", delimiter=",")  #为 xyz的 N*3矩阵
    point_cloud_raw = DataFrame(point_cloud_raw[:, 0:3])  # 选取每一列 的 第0个元素到第二个元素   [0,3)
    point_cloud_raw.columns = ['x', 'y', 'z']  # 给选取到的数据 附上标题
    point_cloud_pynt = PyntCloud(point_cloud_raw)  # 将points的数据 存到结构体中

    point_cloud_o3d = point_cloud_pynt.to_instance("open3d", mesh=False)  # 实例化

    o3d.visualization.draw_geometries([point_cloud_o3d]) # 显示原始点云

    # 从点云中获取点，只对点进行处理
    print(point_cloud_o3d)              #打印点数

    # 用PCA分析点云主方向
    w, v = PCA(point_cloud_raw)        # w为特征值 v为主方向
    point_cloud_vector1 = v[:, 0]   #点云主方向对应的向量，第一主成分
    point_cloud_vector2 = v[:, 1]  # 点云主方向对应的向量，第二主成分
    point_cloud_vector = v[:,0:2]  # 点云主方向与次方向
    print('the main orientation of this pointcloud is: ', point_cloud_vector1)
    print('the main orientation of this pointcloud is: ', point_cloud_vector2)

    #在原点云中画图
    point = [[0,0,0],point_cloud_vector1,point_cloud_vector2]  #画点：原点、第一主成分、第二主成分
    lines = [[0,1],[0,2]]      #画出三点之间两两连线
    colors = [[1,0,0],[0,0,0]]
    #构造open3d中的LineSet对象，用于主成分显示
    line_set = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(point),lines=o3d.utility.Vector2iVector(lines))
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([point_cloud_o3d,line_set]) # 显示原始点云和PCA后的连线

    #将原数据进行降维度处理
    point_cloud_encode = (np.dot(point_cloud_vector.T,point_cloud_raw.T)).T   #主成分的转置 dot 原数据
    Point_Show(point_cloud_encode)
    #使用主方向进行升维
    point_cloud_decode = (np.dot(point_cloud_vector,point_cloud_encode.T)).T
    Point_Cloud_Show(point_cloud_decode)

    # 循环计算每个点的法向量
    pcd_tree = o3d.geometry.KDTreeFlann(point_cloud_o3d)           #将原始点云数据输入到KD,进行近邻取点
    normals = []    #储存曲面的法向量
    # 作业2
    # 屏蔽开始
    print(point_cloud_raw.shape[0])      #打印当前点数 20000个点
    for i in range(point_cloud_raw.shape[0]):
        # search_knn_vector_3d函数 ， 输入值[每一点，x]      返回值 [int, open3d.utility.IntVector, open3d.utility.DoubleVector]
        [_,idx,_] = pcd_tree.search_knn_vector_3d(point_cloud_o3d.points[i],10) #取10个临近点进行曲线拟合
        # asarray和array 一样 但是array会copy出一个副本，asarray不会，节省内存
        k_nearest_point = np.asarray(point_cloud_o3d.points)[idx, :] #找出每一点的10个临近点，类似于拟合成曲面，然后进行PCA找到特征向量最小的值，作为法向量
        w, v = PCA(k_nearest_point)
        normals.append(v[:, 2])

    # 屏蔽结束
    normals = np.array(normals, dtype=np.float64)
    # TODO: 此处把法向量存放在了normals中
    point_cloud_o3d.normals = o3d.utility.Vector3dVector(normals)
    o3d.visualization.draw_geometries([point_cloud_o3d])


if __name__ == '__main__':
    main()

