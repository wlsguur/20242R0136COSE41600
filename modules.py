import open3d as o3d
import numpy as np
from utils import DBSCAN

def get_moving_pcd(pcd_prev, pcd_cur, threshold=0.2):
    current_points = np.asarray(pcd_cur.points)
    previous_points = np.asarray(pcd_prev.points)
    moving_points = []

    tree = o3d.geometry.KDTreeFlann(pcd_prev)
    moving_points = []
    
    for point in current_points:
        _, idx, _ = tree.search_knn_vector_3d(point, 1)
        nearest_point = previous_points[idx[0]]

        if np.linalg.norm(point - nearest_point) > threshold:
            moving_points.append(point)
    
    if moving_points:
        moving_pcd = o3d.geometry.PointCloud()
        moving_pcd.points = o3d.utility.Vector3dVector(np.array(moving_points))
    else:
        moving_pcd = None

    return moving_pcd
    
def get_moved_pedestrian(center_prev, pcd_cur):
    tree = o3d.geometry.KDTreeFlann(pcd_cur)
    _, idxs, _ = tree.search_knn_vector_3d(center_prev, 50)
    nearest_points = np.array(pcd_cur.points)[idxs]
    nearest_pcd = o3d.geometry.PointCloud()
    nearest_pcd.points = o3d.utility.Vector3dVector(nearest_points)
    pcd, labels = DBSCAN(nearest_pcd)
    num_clusters = labels.max() + 1

    if num_clusters <= 1:
        return get_pedestrians(pcd, labels)
    
    else:
        cluster_distances = []
        cluster_pcds = []

        for i in range(num_clusters):
            cluster_idx = np.where(labels == i)[0]
            cluster_pcd = pcd.select_by_index(cluster_idx)
            cluster_center = np.array(cluster_pcd.points).mean(axis=0)
            distance = np.linalg.norm(cluster_center - center_prev)
            cluster_distances.append(distance)
            cluster_pcds.append(cluster_pcd)

        closest_cluster_idx = np.argmin(cluster_distances)
        selected_pcd = cluster_pcds[closest_cluster_idx]
        selected_labels = np.full(len(cluster_pcd.points), 0)
        return get_pedestrians(selected_pcd, selected_labels)
    
def get_pedestrians(pcd, labels):
    num_clusters = labels.max() + 1

    min_points_in_cluster = 10
    max_points_in_cluster = 60

    min_z_value = -1.0
    max_z_value = 5.0

    min_height = 0.2
    max_height = 2.0

    min_width_x = 0.2
    max_width_x = 0.8

    min_width_y = 0.2
    max_width_y = 0.8

    bboxes = []
    centers = []

    pedestrians = o3d.geometry.PointCloud()
    for i in range(num_clusters):
        cluster_idxs = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_idxs) <= max_points_in_cluster:
            cluster_pcd = pcd.select_by_index(cluster_idxs)
            points = np.asarray(cluster_pcd.points)
            x_min, x_max = points[:, 0].min(), points[:, 0].max()
            y_min, y_max = points[:, 1].min(), points[:, 1].max()
            z_min, z_max = points[:, 2].min(), points[:, 2].max()

            if min_z_value <= z_min and z_max <= max_z_value:
                x_diff = x_max - x_min
                y_diff = y_max - y_min
                z_diff = z_max - z_min
                if ((min_height <= z_diff <= max_height)
                    and (min_width_x <= x_diff <= max_width_x)
                    and (min_width_y <= y_diff <= max_width_y)):
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0) 
                    bboxes.append(bbox)
                    centers.append(points.mean(axis=0))
                    pedestrians += cluster_pcd

    return pedestrians, bboxes, centers