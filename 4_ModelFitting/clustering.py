# 文件功能：
#     1. 从数据集中加载点云数据
#     2. 从点云数据中滤除地面点云
#     3. 从剩余的点云中提取聚类

import random
import numpy as np
import os
import struct
from sklearn import cluster, datasets, mixture
from itertools import cycle, islice
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import KDTree
import open3d as o3d
import time

# 功能：从kitti的.bin格式点云文件中读取点云
# 输入：
#     path: 文件路径
# 输出：
#     点云数组
def read_velodyne_bin(path):
    '''
    :param path:
    :return: homography matrix of the point cloud, N*3
    '''
    pc_list = []
    with open(path, 'rb') as f:
        content = f.read()
        pc_iter = struct.iter_unpack('ffff', content)
        for idx, point in enumerate(pc_iter):
            pc_list.append([point[0], point[1], point[2]])
    return np.asarray(pc_list, dtype=np.float32)

# 功能：从点云文件中滤除地面点
# 输入：
#     data: 一帧完整点云
# 输出：
#     segmengted_cloud: 删除地面点之后的点云
def ground_segmentation(data):
    # 作业1
    # 屏蔽开始
    
    n = len(data)
    p = 0.99
    iter_num = 100
    sigma = 0.3
    outlier_ratio = 0.5
    best_idx = []
    best_inlier = (1 - outlier_ratio) * n
    best_A, best_B, best_C, best_D = 0, 0, 0, 0
    
    for i in range(iter_num):
        random_idx = random.sample(range(n), 3)
        point0 = data[random_idx[0]]
        point1 = data[random_idx[1]]
        point2 = data[random_idx[2]]
        
        vector01 = point0 - point1
        vector02 = point0 - point2
        normal_vec = np.cross(vector01, vector02)
        A, B, C = normal_vec[0], normal_vec[1], normal_vec[2]
        D = -np.dot(normal_vec, point0)
        
        inliers = 0
        distance = abs(np.dot(data, normal_vec) + D) / np.linalg.norm(normal_vec)
        
        idx = distance < sigma
        inliers = idx.sum()
        
        if inliers > best_inlier:
            best_idx = idx
            best_inlier = inliers
            best_A, best_B, best_C, best_D = A, B, C, D
        
        # if inliers > (1 - outlier_ratio) * n:
        #     print("Break in advance")
        #     break
        
    segmented_cloud_idx = np.logical_not(best_idx)
    # print(best_idx.sum())
    # print(segmented_cloud_idx.sum())

    # 屏蔽结束

    print('origin data points num:', data.shape[0])
    # print('ground data points num:', best_idx.sum())
    # print('segmented data points num:', segmented_cloud_idx.sum())
    return data[best_idx], data[segmented_cloud_idx]

# 功能：从点云中提取聚类
# 输入：
#     data: 点云（滤除地面后的点云）
# 输出：
#     clusters_index： 一维数组，存储的是点云中每个点所属的聚类编号（参考上一章内容容易理解）
def clustering(data):
    # 作业2
    # 屏蔽开始
    
    dis = 0.3
    min_sample = 5
    n = len(data)
    
    leaf_size = 8
    kdtree = KDTree(data, leaf_size)

    core_set = set()
    unvisit_set = set(range(n))
    k = 0
    cluster_idx = np.zeros(n, dtype = int)
    
    nearest_idx = kdtree.query_radius(data, dis)
    for i in range(n):
        if len(nearest_idx[i]) > min_sample:
            core_set.add(i)

    while(len(core_set)):
        unvisit_set_old = unvisit_set
        core = list(core_set)[np.random.randint(0, len(core_set))]
        unvisit_set = unvisit_set - set([core])
        visited = []
        visited.append(core)
        
        while(len(visited)):
            new_core = visited[0]
            if new_core in core_set:
                S = set(unvisit_set) & set(nearest_idx[new_core])
                visited.extend(list(S))
                unvisit_set = unvisit_set - S
            visited.remove(new_core)
        cluster = unvisit_set_old - unvisit_set
        core_set = core_set - cluster
        cluster_idx[list(cluster)] = k
        k = k + 1
        print("core_set:", len(core_set), "unvisit_set:", len(unvisit_set))
    
    noise_cluster = unvisit_set
    cluster_idx[list(noise_cluster)] = -1

    # 屏蔽结束

    return cluster_idx

# 功能：显示聚类点云，每个聚类一种颜色
# 输入：
#      data：点云数据（滤除地面后的点云）
#      cluster_index：一维数组，存储的是点云中每个点所属的聚类编号（与上同）
def plot_clusters(data, cluster_index):
    ax = plt.figure().add_subplot(111, projection = '3d')
    colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(cluster_index) + 1))))
    colors = np.append(colors, ["#000000"])
    ax.scatter(data[:, 0], data[:, 1], data[:, 2], s=2, color=colors[cluster_index])
    plt.show()
    
def plot_clusters_o3d(segmented_ground, segmented_cloud, cluster_index):
    """
    Visualize segmentation results using Open3D

    Parameters
    ----------
    segmented_cloud: numpy.ndarray
        Segmented surrounding objects as N-by-3 numpy.ndarray
    segmented_ground: numpy.ndarray
        Segmented ground as N-by-3 numpy.ndarray
    cluster_index: list of int
        Cluster ID for each point

    """
    def colormap(c, color_list):
        """
        Colormap for segmentation result

        Parameters
        ----------
        c: int
            Cluster ID
        C

        """
        # outlier:
        if c == -1:
            color = [1]*3
        # surrouding object:
        else:
            color = color_list[c]

        return color

    # ground element:
    pcd_ground = o3d.geometry.PointCloud()
    pcd_ground.points = o3d.utility.Vector3dVector(segmented_ground)
    pcd_ground.colors = o3d.utility.Vector3dVector(
        [
            [0, 0, 0] for i in range(segmented_ground.shape[0])
        ]
    )

    # surrounding object elements:
    pcd_objects = o3d.geometry.PointCloud()
    pcd_objects.points = o3d.utility.Vector3dVector(segmented_cloud)
    num_clusters = max(cluster_index) + 1
    color_list = []
    for _ in range(num_clusters):
        color_list.append([random.random(), random.random(), random.random()])
    pcd_objects.colors = o3d.utility.Vector3dVector(
        [
            colormap(c, color_list) for c in cluster_index
        ]
    )

    # visualize:
    o3d.visualization.draw_geometries([pcd_ground, pcd_objects])


def main():
    # root_dir = './' # 数据集路径
    # cat = os.listdir(root_dir)
    # cat = cat[1:]
    # iteration_num = len(cat)

    # for i in range(iteration_num):
    #     filename = os.path.join(root_dir, cat[i])
    #     print('clustering pointcloud file:', filename)

    #     origin_points = read_velodyne_bin(filename)
    #     segmented_points = ground_segmentation(data=origin_points)
    #     cluster_index = clustering(segmented_points)

    #     plot_clusters(segmented_points, cluster_index)
    

    filename = '000000.bin'
    print('clustering pointcloud file:', filename)

    origin_points = read_velodyne_bin(filename)
    ground_points, segmented_points = ground_segmentation(data=origin_points)
    begin_t =time.time()
    cluster_index = clustering(segmented_points)
    dbscan_time = time.time() - begin_t
    print("dbscan time:%f"%dbscan_time)

    plot_clusters_o3d(ground_points, segmented_points, cluster_index)
    

if __name__ == '__main__':
    main()
