# COSE416 Self-driving Cars : Homework 1

**Jinhyeok Choi**  
Computer Science and Engineering  
2022320006  

---

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
   - \( p \): period
   - \( \tau \): threshold
   - \( D_t \): point cloud at time \( t \)
   - \( d_t^i \in D_t \): i-th point in \( D_t \)

4. Steps:
   - For each period, perform KNN on \( d_t^i \) to find the nearest point \( d_{t-p}^j \in D_{t-p} \) and compute the distance:
     \[
     l = \lvert d_t^i - d_{t-p}^j \rvert \quad \text{where} \quad j = \text{argmin}_k \lvert d_t^i - d_{t-p}^k \rvert
     \]
   - If \( l > \tau \), add \( d_t^i \) to moving points \( V_t \).
   - Perform DBSCAN on \( V_t \) to get clusters \( C_t^1, \ldots, C_t^N \).
   - Run `get_pedestrian()` on each cluster \( C_t^1, \ldots, C_t^N \) to get pedestrian clusters \( P_t^1, \ldots, P_t^M \).

### Update
1. Based on current pedestrian points, find the nearest future points to update moving pedestrian points.
2. Assumes the previous prediction is correct.
3. Definitions:
   - \( D_t \): point cloud at time \( t \)
   - \( P_t^1, \ldots, P_t^M \): pedestrian clusters at time \( t \)

4. Steps:
   - Find the center \( c_{t-1}^i \in P_{t-1}^i \) for all \( i = 1, \ldots, M \).
   - Perform KNN on \( c_{t-1}^i \) to find the top 50 nearest points \( S_t^i = \{ s_t^1, \ldots, s_t^{50} \} \subset D_t \).
   - Perform DBSCAN on \( S_t^i \) for all \( i = 1, \ldots, M \) to get clusters \( C_t^{i,1}, \ldots, C_t^{i,N} \).
   - Find the closest cluster \( C_t^{i,k} \) to \( c_{t-1}^i \).
   - Run `get_pedestrian()` on \( C_t^{i,k} \) to get pedestrian clusters \( P_t^i \).

---

## Experiment and Results

### Setup
- For all experiments:
  - \( p = 10 \)
  - \( \tau = 0.2 \)
- Hyperparameter tuning results for the `get_pedestrian()` function:

| Dataset           | Min z  | Max z  | Min height | Max height | Min width \( x \) | Max width \( x \) | Min width \( y \) | Max width \( y \) |
|-------------------|--------|--------|------------|------------|-------------------|-------------------|-------------------|-------------------|
| 01_straight_walk  | -1     | 5      | 0.5        | 1.2        | 0.4               | 0.8               | 0.2               | 0.6               |
| 02_straight_duck  | -1     | 5      | 0.2        | 1.2        | 0.3               | 0.8               | 0.2               | 0.8               |
| 03_straight_crawl | -1     | 3      | 0.2        | 1.0        | 0.2               | 0.8               | 0.2               | 0.8               |
| 04_zigzag_walk    | -1     | 3      | 0.2        | 2.0        | 0.2               | 0.8               | 0.2               | 0.8               |
| Final parameters  | -1     | 5      | 0.2        | 2.0        | 0.2               | 0.8               | 0.2               | 0.8               |

### Results
- Detailed result for `01_straight_walk` dataset:
  - **Red points** represent results of the **Scan** process.
  - **Blue points** represent results of the **Update** process.
- Prediction trajectory and bounding box visualizations provided in the dataset.
