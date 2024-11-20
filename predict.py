import open3d as o3d
import numpy as np
import os
from tqdm import tqdm

from utils import preprocess_pcd, DBSCAN, visualize_pcd_sequence
from modules import get_moving_pcd, get_pedestrians, get_moved_pedestrian

scinario_type = "01_straight_walk"
data_dir = f"dataset/{scinario_type}/pcd/"

pedestrian_list = []
bbox_list = []
pcd_list = []

previous_pcd = None
previous_centers = None
period = 10
i = -1

for data_name in tqdm(sorted(os.listdir(data_dir))):
    i += 1
    data_path = os.path.join(data_dir, data_name)
    pcd = o3d.io.read_point_cloud(data_path)
    pcd = preprocess_pcd(pcd)
    pcd_list.append(pcd)

    if i % period != 0 and previous_centers is not None:
        for center in previous_centers:
            pedestrians, bboxes, centers = get_moved_pedestrian(center, pcd)
            pedestrian_list.append(pedestrians)
            bbox_list.append(bboxes)
            previous_centers = centers

    elif previous_pcd is not None:
        moving_pcd = get_moving_pcd(previous_pcd, pcd)
        if moving_pcd:
            moving_pcd, labels = DBSCAN(moving_pcd)
            moving_pcd, bboxes, centers = get_pedestrians(moving_pcd, labels)
            pedestrian_list.append(moving_pcd)
            bbox_list.append(bboxes)
            previous_centers = centers
            previous_pcd = pcd

    elif previous_pcd is None:
        previous_pcd = pcd

if pedestrian_list:
    #o3d.visualization.draw_geometries(pedestrian_list, window_name="Moving Objects", width=1600, height=1200)
    visualize_pcd_sequence(pcd_list, bbox_list)    
else:
    print("No moving objects detected.")
