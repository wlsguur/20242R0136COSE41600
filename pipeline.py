import open3d as o3d
import os
import numpy as np
from tqdm import tqdm
from utils import set_color, preprocess_pcd, DBSCAN, visualize_pcd_trajectory, save_pcd_trajectory, visualize_pcd_sequence
from modules import get_moving_pcd, get_pedestrians, get_moved_pedestrian

class ScanUpdatePipeline():

    def __init__(self):
        self.pcd_dataset = []
        self.pedestrian_list = []
        self.moving_pcd_list = []
        self.bbox_list = []
        self.pcd_cur = None
        self.pcd_prev = None
        self.centers_prev = None
    
    def reset(self):
        self.__init__(self)
     
    def get_dataset(self, data_dir):
        for data_name in tqdm(sorted(os.listdir(data_dir)), desc="Load PCD dataset"):
            if not data_name.endswith(".pcd"):
                continue
            pcd_path = os.path.join(data_dir, data_name)
            pcd = o3d.io.read_point_cloud(pcd_path)
            pcd = preprocess_pcd(pcd)
            self.pcd_dataset.append(pcd)

    def scan(self, threshold=0.5):
        moving_pcd = get_moving_pcd(self.pcd_prev, self.pcd_cur, threshold=threshold)
        if moving_pcd is not None:
            moving_pcd, labels = DBSCAN(moving_pcd)
            pedestrians, idxs, bboxes, centers = get_pedestrians(moving_pcd, labels)
            moving_pcd = set_color(moving_pcd.select_by_index(idxs, invert=True), color=(0,0,0))
            self.moving_pcd_list.append(moving_pcd)
            pedestrians = set_color(pedestrians, color=(1,0,0))
            self.pedestrian_list.append(pedestrians)
            self.bbox_list.append(bboxes)
            self.centers_prev = centers
            self.pcd_prev = self.pcd_cur
        else:
            self.bbox_list.append(None)

    def update(self):
        temp_bboxes = []
        temp_centers = []
        for center in self.centers_prev:
            if center is not None:
                pedestrians, _, bboxes, centers = get_moved_pedestrian(center, self.pcd_cur)
                pedestrians = set_color(pedestrians, color=(0,0,1))
                self.pedestrian_list.append(pedestrians)
                temp_bboxes.extend(bboxes)
                temp_centers.extend(centers)
        self.bbox_list.append(temp_bboxes) if temp_bboxes else self.bbox_list.append(None)
        self.centers_prev = temp_centers if temp_centers else None

    def predict_loop(self, period):
        i = -1
        for pcd in tqdm(self.pcd_dataset, desc="Inference"):
            i += 1
            self.pcd_cur = pcd

            if i < period:
                self.bbox_list.append(None)
            elif i % period == 0 and self.pcd_prev is not None:
                self.scan(threshold=0.2)
            elif i % period != 0 and self.centers_prev is not None:
                self.update()
            else:
                self.bbox_list.append(None)
            if self.pcd_prev is None:
                self.pcd_prev = self.pcd_cur

    def run(self,
            period=10,
            show_trajectory=True,
            show_video=False,
            save_trajectory=False,
            save_video=False,
            save_dir="result/"):
        
        self.predict_loop(period)

        if (save_trajectory or save_video) and not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        if show_trajectory:
            visualize_pcd_trajectory(self.pedestrian_list + self.moving_pcd_list)

        if save_trajectory:
            save_pcd_trajectory(self.pedestrian_list + self.moving_pcd_list,
                                     save_dir=save_dir)
        if show_video or save_video:
            visualize_pcd_sequence(self.pcd_dataset,
                                   self.bbox_list,
                                   save_video=save_video,
                                   save_dir=save_dir)