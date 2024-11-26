# COSE416 Self-driving Cars : Homework 1

**Jinhyeok Choi**  
Computer Science and Engineering  
2022320006  

---
## Quick start
1. Locate pcd sequence dataset at `dataset/{data name}/pcd/`
2. type following:
```
python run.py
```

## Methods

### Overview
- Special 3D pedestrian detection constrained with:
  - No ground truth
  - Limited resources
  - Real-time prediction

- Algorithm-based method (non-deep learning) with two main components:
  - **Scan** compares previous and current point clouds to identify moving points and detect pedestrians.
  - **Update** predicts future positions of pedestrians based on current points.

### Scan
1. Compare previous and current point clouds entirely to get moved points, then find pedestrian points.
2. Challenges: Scanning every frame may fail due to insufficient pedestrian movement and LiDAR sensor noise.
3. Definitions:
   - <code>p</code>: period
   - <code>&tau;</code>: threshold
   - <code>D<sub>t</sub></code>: point cloud at time <code>t</code>
   - <code>d<sub>t</sub><sup>i</sup> &isin; D<sub>t</sub></code>: <code>i</code>-th point in <code>D<sub>t</sub></code>

4. Steps:
   - For each period, perform KNN on <code>d<sub>t</sub><sup>i</sup></code> to find the nearest point <code>d<sub>t-p</sub><sup>j</sup> &isin; D<sub>t-p</sub></code> and compute the distance:
     <pre>
     l = |d<sub>t</sub><sup>i</sup> - d<sub>t-p</sub><sup>j</sup>|
     where j = argmin<sub>k</sub> |d<sub>t</sub><sup>i</sup> - d<sub>t-p</sub><sup>k</sup>|
     </pre>
   - If <code>l &gt; &tau;</code>, add <code>d<sub>t</sub><sup>i</sup></code> to moving points <code>V<sub>t</sub></code>.
   - Perform DBSCAN on <code>V<sub>t</sub></code> to get clusters <code>C<sub>t</sub><sup>1</sup>, ..., C<sub>t</sub><sup>N</sup></code>.
   - Run `get_pedestrian()` on each cluster <code>C<sub>t</sub><sup>1</sup>, ..., C<sub>t</sub><sup>N</sup></code> to get pedestrian clusters <code>P<sub>t</sub><sup>1</sup>, ..., P<sub>t</sub><sup>M</sup></code>.

### Update
1. Based on current pedestrian points, find the nearest future points to update moving pedestrian points.
2. Assumes the previous prediction is correct.
3. Definitions:
   - <code>D<sub>t</sub></code>: point cloud at time <code>t</code>
   - <code>P<sub>t</sub><sup>1</sup>, ..., P<sub>t</sub><sup>M</sup></code>: pedestrian clusters at time <code>t</code>

4. Steps:
   - Find the center <code>c<sub>t-1</sub><sup>i</sup> &isin; P<sub>t-1</sub><sup>i</sup></code> for all <code>i = 1, ..., M</code>.
   - Perform KNN on <code>c<sub>t-1</sub><sup>i</sup></code> to find the top 50 nearest points <code>S<sub>t</sub><sup>i</sup> = {s<sub>t</sub><sup>1</sup>, ..., s<sub>t</sub><sup>50</sup>} &sub; D<sub>t</sub></code>.
   - Perform DBSCAN on <code>S<sub>t</sub><sup>i</sup></code> for all <code>i = 1, ..., M</code> to get clusters <code>C<sub>t</sub><sup>i,1</sup>, ..., C<sub>t</sub><sup>i,N</sup></code>.
   - Find the closest cluster <code>C<sub>t</sub><sup>i,k</sup></code> to <code>c<sub>t-1</sub><sup>i</sup></code>.
   - Run `get_pedestrian()` on <code>C<sub>t</sub><sup>i,k</sup></code> to get pedestrian clusters <code>P<sub>t</sub><sup>i</sup></code>.

---

## Experiment and Results

### Setup
- For all experiments:
  - <code>p = 10</code>
  - <code>&tau; = 0.2</code>
- Hyperparameter tuning results for the `get_pedestrian()` function:

| Dataset           | Min z  | Max z  | Min height | Max height | Min width \( x \) | Max width \( x \) | Min width \( y \) | Max width \( y \) |
|-------------------|--------|--------|------------|------------|-------------------|-------------------|-------------------|-------------------|
| 01_straight_walk  | -1     | 5      | 0.5        | 1.2        | 0.4               | 0.8               | 0.2               | 0.6               |
| 02_straight_duck  | -1     | 5      | 0.2        | 1.2        | 0.3               | 0.8               | 0.2               | 0.8               |
| 03_straight_crawl | -1     | 3      | 0.2        | 1.0        | 0.2               | 0.8               | 0.2               | 0.8               |
| 04_zigzag_walk    | -1     | 3      | 0.2        | 2.0        | 0.2               | 0.8               | 0.2               | 0.8               |
| Final parameters  | -1     | 5      | 0.2        | 2.0        | 0.2               | 0.8               | 0.2               | 0.8               |

### Results
- Inference time: 40fps
- Detailed result for `01_straight_walk` dataset:
  - **Red points** represent results of the **Scan** process.
  - **Blue points** represent results of the **Update** process.
  ![trajectory_detailed](https://github.com/user-attachments/assets/65f3ca8c-06a6-4fdc-b086-463202a4f715)

  - Prediction trajectory and bounding box visualizations provided in the dataset.
  ![bbox](https://github.com/user-attachments/assets/0dcbc6a6-b472-4774-906a-fe51776961ad)

## References

1. Y. Zhang, "A novel fast pedestrian recognition algorithm based on point cloud compression and boundary extraction," *PeerJ Computer Science*, vol. 9, p. e1426, 2023. doi: [10.7717/peerj-cs.1426](https://doi.org/10.7717/peerj-cs.1426).

2. K. Liu, W. Wang, and J. Wang, "Pedestrian detection with LiDAR point clouds based on single template matching," *Electronics*, vol. 8, no. 7, p. 780, 2019. doi: [10.3390/electronics8070780](https://doi.org/10.3390/electronics8070780).

3. K.-I. Na and B. Park, "Real-time 3D multi-pedestrian detection and tracking using 3D LiDAR point cloud for mobile robot," *ETRI Journal*, vol. 45, pp. 836–846, 2023. doi: [10.4218/etrij.2023-0116](https://doi.org/10.4218/etrij.2023-0116).