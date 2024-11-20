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
    pedestrian, bbox, center = get_pedestrians(pcd, labels)

    #assert len(bbox) > 0, "there is no moved pedestrian"
    return pedestrian, bbox, center

def is_directional(points_prev, tree, cluster_points, similarity_threshold=0.9):
    direction_vectors = []

    for point in cluster_points:
        _, idx, _ = tree.search_knn_vector_3d(point, 1)
        nearest_point = points_prev[idx[0]]
        direction = point - nearest_point
        direction_vectors.append(direction)

    direction_vectors = np.array(direction_vectors)

    norms = np.linalg.norm(direction_vectors, axis=1, keepdims=True)
    unit_vectors = direction_vectors / (norms + 1e-8)

    mean_vector = np.mean(unit_vectors, axis=0)
    mean_vector /= np.linalg.norm(mean_vector)

    cosine_similarities = np.dot(unit_vectors, mean_vector)

    directional_count = np.sum(cosine_similarities > similarity_threshold)
    is_directional = directional_count / len(cosine_similarities) > 0.8

    return is_directional


def get_directional_moving_objects(pcd_prev, pcd_cur, labels):
    
    num_clusters = labels.max() + 1
    
    points_prev = np.asarray(pcd_prev.points)
    tree = o3d.geometry.KDTreeFlann(pcd_prev)

    moving_objects = o3d.geometry.PointCloud()
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_pcd = pcd_cur.select_by_index(cluster_indices)
        cluster_points = np.asarray(cluster_pcd.points)
        if is_directional(points_prev, tree, cluster_points):
            moving_objects += cluster_pcd

    return moving_objects
    
def get_pedestrians(pcd, labels):
    num_clusters = labels.max() + 1

    min_points_in_cluster = 0
    max_points_in_cluster = 100

    min_z_value = -1.0
    max_z_value = 5.0

    min_height = 0.5
    max_height = 2.5

    bboxes = []
    centers = []

    pedestrians = o3d.geometry.PointCloud()
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        if min_points_in_cluster <= len(cluster_indices) <= max_points_in_cluster:
            cluster_pcd = pcd.select_by_index(cluster_indices)
            points = np.asarray(cluster_pcd.points)
            z_values = points[:, 2]
            z_min = z_values.min()
            z_max = z_values.max()

            #variance_xyz = np.var(points, axis=0)

            if min_z_value <= z_min and z_max <= max_z_value:
                height_diff = z_max - z_min
                if min_height <= height_diff <= max_height:
                    bbox = cluster_pcd.get_axis_aligned_bounding_box()
                    bbox.color = (1, 0, 0) 
                    bboxes.append(bbox)
                    centers.append(points.mean(axis=0))
                    pedestrians += cluster_pcd

    return pedestrians, bboxes, centers