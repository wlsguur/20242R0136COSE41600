import open3d as o3d
import matplotlib.pyplot as plt
import numpy as np
import time
import cv2
import os

def preprocess_pcd(pcd):
    downsample_pcd = pcd.voxel_down_sample(voxel_size=0.2)

    _, ind = downsample_pcd.remove_radius_outlier(nb_points=6, radius=1.2)
    ror_pcd = downsample_pcd.select_by_index(ind)

    _, inliers = ror_pcd.segment_plane(distance_threshold=0.05,
                                                 ransac_n=3,
                                                 num_iterations=3000)

    preproessed_pcd = ror_pcd.select_by_index(inliers, invert=True)

    return preproessed_pcd

def set_color(pcd, color):  # color = (R, G, B)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (len(pcd.points), 1)))
    return pcd

def DBSCAN(pcd):
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=0.3, min_points=10, print_progress=False))

    num_clusters = labels.max()
    colors = plt.get_cmap("tab20")(labels / (num_clusters + 1 if num_clusters > 0 else 1))
    colors[labels < 0] = 0
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    return pcd, labels

def visualize_pcd_trajectory(pcd_list):
    o3d.visualization.draw_geometries(pcd_list, window_name="Pedestrian trajectory", width=1920, height=1080)

def save_pcd_trajectory(pcd_list,
                        save_dir="result/", 
                        window_name="Pedestrian trajectory",
                        point_size=1.0):
                                
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=1080)
    vis.get_render_option().point_size = point_size

    for geom in pcd_list:
        vis.add_geometry(geom)
    
    vis.poll_events()
    vis.update_renderer()

    save_path = os.path.join(save_dir, "trajectory.png")
    vis.capture_screen_image(save_path, do_render=True)
    vis.destroy_window()
    print(f"Image saved to {save_path}")


def visualize_pcd_sequence(pcd_sequence,
                           bbox_sequence,
                           save_video=False,
                           save_dir="result/",
                           window_name="Pedestrian with bbox video",
                           point_size=1.0,
                           fps=10):
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=window_name, width=1920, height=1080)
    vis.get_render_option().point_size = point_size

    if save_video:
        frame_width = 1920
        frame_height = 1080
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        save_path = os.path.join(save_dir, "video.mp4")
        video_writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_width, frame_height))

    for i in range(len(pcd_sequence)):
        vis.clear_geometries()
        vis.add_geometry(set_color(pcd_sequence[i], color=(0,0,0)))
        if bbox_sequence[i] is not None:
            for bbox in bbox_sequence[i]:
                vis.add_geometry(bbox)

        vis.poll_events()
        vis.update_renderer()

        if save_video:
            img = vis.capture_screen_float_buffer(do_render=True)
            img = (255 * np.asarray(img)).astype(np.uint8)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video_writer.write(img)

        time.sleep(1.0 / fps)

    vis.destroy_window()

    if save_video:
        video_writer.release()
        print(f"Video saved to {save_path}")

def analyze_pedestrians(pedestrian_list):
    x_ranges = []
    y_ranges = []
    z_max_values = []
    z_min_values = []
    z_ranges = []

    for pedestrian in pedestrian_list:
        if not pedestrian.points:
            continue

        points = np.asarray(pedestrian.points)
        
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        z_min, z_max = points[:, 2].min(), points[:, 2].max()

        x_ranges.append(x_max - x_min)
        y_ranges.append(y_max - y_min)
        z_max_values.append(z_max)
        z_min_values.append(z_min)
        z_ranges.append(z_max - z_min)

    plt.figure(figsize=(16, 12))

    # Z-axis: Max, Min, and Range
    plt.subplot(2, 3, 1)
    plt.hist(z_max_values, bins=10, color='red', alpha=0.7, label='Z Max Values')
    plt.title("Histogram of Z Max Values")
    plt.xlabel("Z Max")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 3, 2)
    plt.hist(z_min_values, bins=10, color='pink', alpha=0.7, label='Z Min Values')
    plt.title("Histogram of Z Min Values")
    plt.xlabel("Z Min")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(2, 3, 3)
    plt.hist(z_ranges, bins=10, color='red', alpha=0.7, label='Z Ranges')
    plt.title("Histogram of Z Ranges")
    plt.xlabel("Z Range (Max - Min)")
    plt.ylabel("Frequency")
    plt.legend()

    # X-axis: Range
    plt.subplot(2, 3, 4)
    plt.hist(x_ranges, bins=10, color='blue', alpha=0.7, label='X Ranges')
    plt.title("Histogram of X Ranges")
    plt.xlabel("X Range (Max - Min)")
    plt.ylabel("Frequency")
    plt.legend()

    # Y-axis: Range
    plt.subplot(2, 3, 5)
    plt.hist(y_ranges, bins=10, color='green', alpha=0.7, label='Y Ranges')
    plt.title("Histogram of Y Ranges")
    plt.xlabel("Y Range (Max - Min)")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    plt.boxplot([z_max_values, z_min_values, z_ranges, x_ranges, y_ranges],
                labels=['Z Max', 'Z Min', 'Z Range', 'X Range', 'Y Range'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red'))

    plt.title("Boxplot of Z, X, and Y Values and Ranges")
    plt.ylabel("Value")
    plt.grid(True)
    plt.tight_layout()
    plt.show()