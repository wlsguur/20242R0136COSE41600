import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time

def preprocess_pcd(pcd):
    downsample_pcd = pcd.voxel_down_sample(voxel_size=0.2)

    _, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    _, inliers = ror_pcd.segment_plane(distance_threshold=0.05,
                                                 ransac_n=3,
                                                 num_iterations=3000)

    preproessed_pcd = ror_pcd.select_by_index(inliers, invert=True)

    return preproessed_pcd

def DBSCAN(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False))

    num_clusters = labels.max()
    colors = plt.get_cmap("tab20")(labels / (num_clusters + 1 if num_clusters > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels

def visualize_pcd(pcd, bounding_boxes=None, window_name="Filtered Clusters and Bounding Boxes", point_size=1.0):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.add_geometry(pcd)
    for bbox in bounding_boxes:
        vis.add_geometry(bbox)
    vis.get_render_option().point_size = point_size
    vis.run()
    vis.destroy_window()

def visualize_pcd_sequence(pcd_sequence, bbox_sequence, window_name="PCD Sequence Visualization", point_size=1.0, fps=5):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name)
    vis.get_render_option().point_size = point_size

    for i in range(len(pcd_sequence)):
        vis.clear_geometries()
        vis.add_geometry(pcd_sequence[i])
        if bbox_sequence[i] is not None:
            for bbox in bbox_sequence[i]:
                vis.add_geometry(bbox)
        vis.poll_events()
        vis.update_renderer()

        time.sleep(1.0 / fps)

    vis.destroy_window()