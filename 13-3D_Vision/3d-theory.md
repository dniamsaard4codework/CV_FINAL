# Unit 13: 3D Vision (Exam-Ready Guide)

## Table of Contents
1. [Introduction](#1-introduction)
2. [3D Data Representations](#2-3d-data-representations)
3. [Depth Estimation](#3-depth-estimation)
4. [3D Reconstruction](#4-3d-reconstruction)
5. [Point Clouds](#5-point-clouds)
6. [PointNet and PointNet++](#6-pointnet-and-pointnet)
7. [Meshes](#7-meshes)
8. [Rendering](#8-rendering)
9. [NeRF](#9-nerf)
10. [From Lecture to Lab](#10-from-lecture-to-lab)
11. [Key Formulas](#11-key-formulas)
12. [Common Exam Questions](#12-common-exam-questions)

---

## 1. Introduction

3D vision aims to understand and process 3D information from sensors and images.

Applications: autonomous driving (distance to obstacles), robotics (manipulation and navigation), AR/VR (realistic rendering), medical imaging (3D organs), manufacturing (inspection, metrology).

Key tasks: depth estimation, 3D reconstruction, 3D detection, point cloud processing, mesh editing, and differentiable rendering.

---

## 2. 3D Data Representations

- **Depth maps:** per-pixel depth; often obtained from stereo, LiDAR projected to image plane, or RGB-D cameras.
- **Point clouds:** unordered sets of points $\{(x, y, z, [\text{features}])\}$; variable size and density; sparse in 3D space.
- **Meshes:** vertices / edges / faces (often triangles); can include materials, textures, UV coordinates; common formats: OBJ, PLY, STL, OFF.
- **Voxels:** regular 3D grid of occupancy or scalar values; simple to use with 3D CNNs but memory-heavy.
- **Implicit representations:** signed distance functions (SDF), occupancy networks, neural fields; represent surfaces as level sets of continuous functions.

Representation comparison (exam-style):
- Depth: compact and image-like, but only visible surface and view-dependent.
- Point cloud: flexible, no explicit topology, good for sparse sensors like LiDAR.
- Mesh: explicit surface + topology, great for rendering, but harder to learn directly.
- Voxel: regular grid easy for CNNs, but memory grows cubic with resolution.
- Implicit: continuous and resolution-free, can represent complex topology, need sampling to render.

---

## 3. Depth Estimation

**Stereo vision (detailed derivation):**

**Setup:**
- Two cameras (left and right) with parallel optical axes
- Baseline: $B$ (distance between camera centers)
- Focal length: $f$ (same for both cameras, assuming identical cameras)
- Image planes: at distance $f$ from camera centers

**Geometry:**
- 3D point $P = (X, Y, Z)$ projects to:
  - Left image: $x_L = f \frac{X}{Z}$
  - Right image: $x_R = f \frac{X - B}{Z}$ (right camera is shifted by $B$)

**Disparity:**
```math
d = x_L - x_R = f \frac{X}{Z} - f \frac{X - B}{Z} = f \frac{B}{Z}
```

**Depth formula:**
```math
Z = \frac{f \cdot B}{d}
```

**Key observations:**
- **Larger disparity** (bigger shift) → **closer object** (smaller $Z$)
- **Small disparity** → **far object** (larger $Z$)
- **Zero disparity** → object at infinity
- **Disparity uncertainty:** $\Delta Z = -\frac{fB}{d^2} \Delta d$, so depth error increases quadratically with distance

**Correspondence problem:**
- Finding matching pixels between left and right images
- **Epipolar constraint:** Corresponding point lies on epipolar line (reduces search to 1D)
- **Methods:** Block matching, SAD (Sum of Absolute Differences), SSD, normalized cross-correlation
- **Challenges:** Occlusions, textureless regions, repetitive patterns

**Rectification:**
- Process of aligning image planes to be parallel (simplifies correspondence search)
- Transforms images so epipolar lines are horizontal

**Monocular depth:**
- Use CNNs (encoder–decoder) to regress depth from a single image.
- Often trained with supervised ground-truth or self-supervision via view synthesis.
- Has scale ambiguity: absolute scale difficult without additional information.

**Active sensors:**
- Structured light: project known pattern, observe deformation; good indoors, short-range.
- Time-of-Flight (ToF): emit light pulse, measure return time, directly get depth.
- LiDAR: laser scanning, long-range, sparse, common in autonomous driving.

---

## 4. 3D Reconstruction

**Structure from Motion (SfM):**
- Feature detection and matching across images.
- Estimate relative pose using epipolar geometry.
- Triangulate 3D points from correspondences.
- Bundle adjustment: jointly refine camera poses and 3D points by minimizing reprojection error.

**Multi-View Stereo (MVS):**
- After SfM gives camera poses, perform dense matching to obtain detailed depth maps / dense point clouds.
- Surface reconstruction: convert dense point cloud to mesh (e.g., Poisson reconstruction).

**SLAM (Simultaneous Localization and Mapping):**
- Estimate camera trajectory while building a map.
- Variants: visual SLAM, RGB-D SLAM, LiDAR SLAM.
- Typical components: front-end (feature tracking / odometry), back-end (optimization / loop closure), mapping.

**Epipolar geometry:**
- Fundamental matrix $F$ encodes relation between corresponding points in two uncalibrated images:

  ```math
  x'^\top F x = 0
  ```

- Essential matrix $E = K'^\top F K$ for calibrated cameras with intrinsics $K, K'$.
- Epipolar constraint: match for a point in one image must lie on corresponding epipolar line in the other image.

---

## 5. Point Clouds

Properties:
- Unordered, irregular, and with variable density.
- Permutation invariance: ordering of points should not affect model output.

Traditional operations:
- Downsampling (e.g., voxel grid, random or Farthest Point Sampling).
- Normal estimation, filtering, segmentation.
- Registration: ICP (Iterative Closest Point) to align two point clouds.
  - **ICP algorithm:**
    - **Goal:** Find rigid transformation (rotation $R$, translation $t$) aligning source point cloud to target
    - **Steps (iterative):**
      1. For each point in source, find closest point in target
      2. Compute optimal $R, t$ minimizing: $\sum_i \|R \mathbf{p}_i + \mathbf{t} - \mathbf{q}_i\|^2$ where $\mathbf{q}_i$ is closest point
      3. Apply transformation to source
      4. Repeat until convergence (change in transformation < threshold)
    - **Solution for step 2:**
      - Center both point clouds: $\bar{\mathbf{p}} = \frac{1}{n}\sum \mathbf{p}_i$, $\bar{\mathbf{q}} = \frac{1}{n}\sum \mathbf{q}_i$
      - Compute cross-covariance: $H = \sum_i (\mathbf{p}_i - \bar{\mathbf{p}})(\mathbf{q}_i - \bar{\mathbf{q}})^T$
      - SVD: $H = U \Sigma V^T$, then $R = V U^T$ (ensuring $\det(R) = 1$)
      - Translation: $\mathbf{t} = \bar{\mathbf{q}} - R \bar{\mathbf{p}}$
    - **Limitations:** Requires good initialization, can get stuck in local minima, sensitive to outliers
    - **Variants:** Point-to-plane ICP (uses normals), robust ICP (handles outliers)

Challenges for CNNs:
- No regular grid and varying point count, so standard 2D/3D convolutions are not directly applicable.
- Require permutation-invariant operations and sometimes explicit neighborhood definitions.

---

## 6. PointNet and PointNet++

**PointNet (key idea):** achieve permutation invariance via shared MLP per point + symmetric aggregation (max pooling).

Architecture sketch:
- Input transform (T-Net) predicts a $3 \times 3$ matrix to align points.
- Shared MLP across points (e.g., 64, 64).
- Feature transform (another T-Net on features, e.g., $64 \times 64$).
- Shared MLP up to a 1024-d global feature per point.
- Symmetric max-pooling across points yields a global feature vector.
- Heads:
  - Classification: MLP from global feature to class logits.
  - Segmentation: concatenate global feature back to per-point features, then per-point MLP.

Implementation trick: shared MLP implemented as `Conv1d` with `kernel_size = 1` over points.

**PointNet++:**
- Adds hierarchical structure: set abstraction layers.
- Each layer:
  - Sample a subset of points (FPS).
  - Group local neighborhoods (ball query).
  - Apply a small PointNet to each group.
- Captures local structure at multiple scales, improving performance on complex shapes.

---

## 7. Meshes

Mesh basics:
- Vertices $V$ ($N \times 3$), faces $F$ ($M \times 3$ indices), edges $E$.
- Optional attributes: vertex normals, UV coordinates, materials / textures.
- Common formats:
  - OBJ: vertices, faces, normals, texture coordinates, material references.
  - PLY: points / meshes with flexible attributes.
  - STL: triangular faces only.

Normals:
- Face normal:

  ```math
  n = \frac{(v_1 - v_0) \times (v_2 - v_0)}{\lVert (v_1 - v_0) \times (v_2 - v_0) \rVert}
  ```

- Vertex normals often average incident face normals (possibly weighted).

Mesh operations:
- Subdivision (increase resolution), decimation (simplify), smoothing (e.g., Laplacian), remeshing.
- Boolean operations: union, intersection, difference.

---

## 8. Rendering

Rendering pipeline:
- 3D mesh / point cloud → model and world transform → camera / view transform → projection (perspective or orthographic) → rasterization → shading → 2D image.

**Pinhole camera model (detailed):**

**Coordinate systems:**
1. **World coordinates:** $(X_w, Y_w, Z_w)$ - 3D scene points
2. **Camera coordinates:** $(X_c, Y_c, Z_c)$ - after world-to-camera transform
3. **Image coordinates:** $(x, y)$ - 2D projection

**Perspective projection:**
- Camera center at origin, image plane at $Z = f$
- Similar triangles: $\frac{x}{f} = \frac{X_c}{Z_c}$, $\frac{y}{f} = \frac{Y_c}{Z_c}$

```math
x = f \frac{X_c}{Z_c}, \quad y = f \frac{Y_c}{Z_c}
```

**Homogeneous coordinates:**
```math
\begin{bmatrix}
x \\ y \\ 1
\end{bmatrix}
\propto
\begin{bmatrix}
f & 0 & 0 & 0 \\
0 & f & 0 & 0 \\
0 & 0 & 1 & 0
\end{bmatrix}
\begin{bmatrix}
X_c \\ Y_c \\ Z_c \\ 1
\end{bmatrix}
```

**Camera intrinsics matrix $K$:**
Accounts for:
- Focal length: $f_x, f_y$ (may differ if pixels are not square)
- Principal point: $(c_x, c_y)$ (image center, accounting for optical axis offset)
- Skew: $s$ (usually 0 for modern cameras)

```math
K =
\begin{bmatrix}
f_x & s   & c_x \\
0   & f_y & c_y \\
0   & 0   & 1
\end{bmatrix}
```

**Full projection:**
```math
\begin{bmatrix}
u \\ v \\ 1
\end{bmatrix}
= K
\begin{bmatrix}
X_c \\ Y_c \\ Z_c
\end{bmatrix}
= K [R | \mathbf{t}]
\begin{bmatrix}
X_w \\ Y_w \\ Z_w \\ 1
\end{bmatrix}
```

where $R, \mathbf{t}$ are rotation and translation (extrinsics).

**Distortion:**
- **Radial distortion:** $r' = r(1 + k_1 r^2 + k_2 r^4 + k_3 r^6)$
- **Tangential distortion:** $(x', y') = (x, y) + [2p_1 xy + p_2(r^2 + 2x^2), p_1(r^2 + 2y^2) + 2p_2 xy]$
- Usually corrected via calibration before projection

Shading (Phong model):

```math
I = I_a + I_d (N \cdot L) + I_s (R \cdot V)^n
```

- $I_a$: ambient, $I_d$: diffuse, $I_s$: specular.
- $N$: normal, $L$: light direction, $R$: reflection, $V$: view direction, $n$: shininess exponent.

Differentiable rendering:
- Compute approximate gradients through rasterization / shading.
- Enables learning 3D shape / texture from 2D images by backpropagating image losses into 3D parameters.

Practical issues:
- Face winding (CW vs CCW) affects which faces are front-facing.
- Z-fighting when surfaces are nearly coplanar.
- Near / far plane choices affect depth precision.

---

## 9. NeRF

**Neural Radiance Field (NeRF):**
- Represent a scene as a continuous function implemented by an MLP:
  - Input: 3D position $x$ and view direction $d$.
  - Output: color $c$ and volume density $\sigma$.
- Use positional encoding for $x$ and $d$ to capture high-frequency details.

**Volume rendering (detailed derivation):**

**Continuous formulation:**
For a ray $\mathbf{r}(t) = \mathbf{o} + t \mathbf{d}$ (origin $\mathbf{o}$, direction $\mathbf{d}$), the expected color is:

```math
C(\mathbf{r}) = \int_{t_n}^{t_f} T(t) \sigma(\mathbf{r}(t)) \mathbf{c}(\mathbf{r}(t), \mathbf{d})\, dt
```

where:
- $T(t) = \exp\left(-\int_{t_n}^{t} \sigma(\mathbf{r}(s))\, ds\right)$ is **transmittance** (probability ray travels from $t_n$ to $t$ without hitting anything)
- $\sigma(\mathbf{r}(t))$ is **volume density** (differential probability of termination per unit distance)
- $\mathbf{c}(\mathbf{r}(t), \mathbf{d})$ is **emitted color** (view-dependent)

**Discrete approximation:**
Partition ray into $N$ segments $[t_i, t_{i+1}]$ with $\delta_i = t_{i+1} - t_i$:

```math
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \bigl(1 - e^{-\sigma_i \delta_i}\bigr) \mathbf{c}_i
```

where:
```math
T_i = \exp\left(-\sum_{j=1}^{i-1} \sigma_j \delta_j\right)
```

**Interpretation:**
- $1 - e^{-\sigma_i \delta_i}$: probability of termination in segment $i$ (opacity)
- $T_i$: probability ray reaches segment $i$ (transmittance)
- Product: probability ray terminates in segment $i$ and emits color $\mathbf{c}_i$
- Sum: expected color (alpha compositing)

**Alpha compositing:**
```math
\alpha_i = 1 - e^{-\sigma_i \delta_i}
```

```math
\hat{C}(\mathbf{r}) = \sum_{i=1}^{N} T_i \alpha_i \mathbf{c}_i
```

where $T_i = \prod_{j=1}^{i-1} (1 - \alpha_j)$ (standard alpha blending).

**Hierarchical sampling:**
- **Coarse network:** Samples uniformly, predicts density
- **Fine network:** Samples more densely in regions with high density (importance sampling)
- Reduces number of samples needed while maintaining quality

Training:
- Camera poses are known; for each ray, predict $\hat C(r)$ and minimize difference to ground-truth pixel color.
- Hierarchical sampling: coarse network predicts where interesting regions are, fine network samples more densely there.

Variants:
- Instant-NGP (hash encoding) for speed.
- Mip-NeRF for anti-aliasing and multi-scale.
- NeRF-W for scenes with changing appearance.
- D-NeRF for dynamic scenes.

---

## 10. From Lecture to Lab

### Lab 12.1 (PyTorch3D Rendering Tutorial)

- Build or load a mesh, set up cameras (e.g., `look_at_view_transform`), lights, rasterization settings, and a SoftPhong shader.
- Render images; experiment with rotating the camera, changing lighting, and visualizing point clouds.
- Important notes:
  - Ensure correct face winding so backface culling behaves as expected.
  - Adjust near / far planes and blur radius to reduce z-fighting.
  - Keep track of coordinate conventions (world, view, NDC).

### Lab 12.2 (Open3D Mesh Work)

- Load a mesh, visualize it, estimate normals, crop parts, run Laplacian smoothing.
- Paint mesh with colors, sample points on surface, filter outlier points.
- Work with file formats like OBJ and PLY; observe how normals and visualization update.

### Lab 12.3 (PointNet Classification)

- Preprocess: sample a fixed number of points (FPS vs random), normalize (zero-mean, unit sphere), augment (jitter, rotate, scale).
- Model: PointNet with shared Conv1d layers, T-Nets, max-pool to 1024-d global feature, followed by MLP for classification.
- Training / testing: standard CE loss on classes; evaluate accuracy on a held-out test split.

---

## 11. Key Formulas

- Depth from disparity:

  ```math
  Z = \frac{f \cdot B}{d}
  ```

- Perspective projection:

  ```math
  x = \frac{f X}{Z}, \quad y = \frac{f Y}{Z}
  ```

- Phong shading:

  ```math
  I = I_a + I_d (N \cdot L) + I_s (R \cdot V)^n
  ```

- Triangle normal:

  ```math
  n = \frac{(v_1 - v_0) \times (v_2 - v_0)}{\lVert (v_1 - v_0) \times (v_2 - v_0) \rVert}
  ```

- Fundamental matrix constraint:

  ```math
  x'^\top F x = 0
  ```

- NeRF rendering:

  ```math
  \hat C(r) = \sum_i T_i \bigl(1 - e^{-\sigma_i \delta_i}\bigr) c_i
  ```

---

## 12. Common Exam Questions

1) **Main 3D representations?** Point clouds, meshes, voxels, depth maps, implicit neural fields.  
2) **Why do standard CNNs fail on raw point clouds?** No regular grid or consistent ordering; need permutation-invariant operations and variable-size handling.  
3) **How does PointNet get permutation invariance?** Applies a shared MLP to each point, then aggregates with a symmetric function (max-pooling).  
4) **Purpose of T-Net in PointNet?** Learns spatial (and feature) transformations to align inputs, handling rotations / translations.  
5) **Explain stereo depth estimation.** Find correspondences, compute disparity $d$, then compute depth via $Z = fB / d$.  
6) **What is NeRF?** An MLP that maps 3D position and view direction to color and density, using volume rendering along rays; trained using known camera poses and 2D images.

---

## 13. Exam-Style Questions

1. **Stereo geometry basics**  
   (a) Draw the stereo setup and define baseline $B$, focal length $f$, and disparity $d$.  
   (b) Starting from similar triangles, derive $Z = fB / d$.  
   (c) Explain what happens to depth uncertainty when disparity measurement noise increases.

2. **Point cloud processing**  
   (a) Explain why naively feeding a point cloud into a standard 3D CNN after voxelization can be inefficient.  
   (b) Describe how PointNet handles unordered input and why max-pooling is key.  
   (c) Explain at a high level how PointNet++ adds local structure on top of PointNet.

3. **Mesh-based rendering pipeline**  
   For a triangle mesh:  
   (a) Outline the steps from object space to image space (model → world → view → projection → rasterization → shading).  
   (b) Write the perspective projection of a 3D point in camera coordinates and explain the role of the camera intrinsics matrix $K$.  
   (c) List two common numerical issues in rasterization and how you mitigate them.

4. **Differentiable rendering vs NeRF**  
   Compare classical differentiable rasterization (e.g., PyTorch3D) and NeRF-based volume rendering in terms of:  
   (a) Required representation (mesh vs implicit field).  
   (b) Type of gradients they provide (surface-based vs volumetric).  
   (c) A situation where NeRF is preferable and one where mesh-based methods are preferable.

5. **SLAM and SfM relationship**  
   (a) Explain how SfM and SLAM are similar at a high level (what they both estimate).  
   (b) Explain two key differences in assumptions or outputs (e.g., offline vs online, loop closure, map structure).  
   (c) Describe why loop closure is critical in SLAM but not necessarily emphasized in pure SfM.

