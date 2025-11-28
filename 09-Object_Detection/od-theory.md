# Unit 9: Object Detection (Exam-Ready Guide)

Read this as a checklist-first cheat sheet, then use the equations and pipeline notes to rehearse derivations and quick calculations. All important formulas use LaTeX-style math.

## Table of Contents
1. [Exam Checklist](#1-exam-checklist)
2. [Problem Definition and Outputs](#2-problem-definition-and-outputs)
3. [Metrics and How They’re Computed](#3-metrics-and-how-theyre-computed)
4. [Classical Baselines](#4-classical-baselines)
5. [Two-Stage Detectors](#5-two-stage-detectors)
6. [One-Stage Detectors](#6-one-stage-detectors)
7. [YOLO Lineage](#7-yolo-lineage)
8. [Anchors and Box Regression](#8-anchors-and-box-regression)
9. [Non-Maximum Suppression](#9-non-maximum-suppression)
10. [Modern Architectures](#10-modern-architectures)
11. [From Lecture to Lab](#11-from-lecture-to-lab)
12. [Key Formulas](#12-key-formulas)
13. [Rapid Q&A](#13-rapid-qa)

---

## 1. Exam Checklist

- Define detection vs classification / localization / instance segmentation.
- Compute **IoU**, precision / recall, AP / mAP (VOC vs COCO averaging).
- Explain **two-stage vs one-stage** trade-offs (speed, accuracy, class imbalance).
- Walk through **R-CNN → Fast / Faster R-CNN → FPN** improvements.
- Describe **SSD / RetinaNet** heads, focal loss purpose, and anchor design.
- Derive **YOLOv1 loss terms** and output tensor shapes $(S, B, C)$.
- Explain **anchor matching** and **box delta parameterization**.
- Run through **NMS / Soft-NMS / DIoU-NMS** logic.
- Summarize **anchor-free and DETR** ideas.

---

## 2. Problem Definition and Outputs

- Detection = **localization + classification** for multiple objects.
- Outputs: boxes in $(x_\text{min}, y_\text{min}, x_\text{max}, y_\text{max})$ or $(x_\text{center}, y_\text{center}, w, h)$ (YOLO style) plus class scores.
- Tasks table:
  - Classification → 1 label.
  - Localization → 1 label + 1 box.
  - Detection → many boxes + labels + confidences.
  - Instance segmentation → mask per object; panoptic = semantic masks for “stuff” + instance masks for “things”.
- Challenges: variable object count, mixed regression + classification outputs, strong speed constraints.

---

## 3. Metrics and How They’re Computed

- **IoU:** intersection-over-union

```math
\text{IoU} = \frac{\text{area(intersection)}}{\text{area(union)}}
```

  - VOC: IoU threshold $0.5$.
  - COCO: AP averaged over IoU thresholds $0.5:0.95$ (step $0.05$).

- **Precision / Recall:**

```math
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}
```

- **AP (Average Precision):** area under the precision–recall curve for a single class.
  - **VOC11 method (11-point interpolation):**
    - Sample precision at 11 recall levels: $0.0, 0.1, 0.2, \ldots, 1.0$
    - For each recall level $r$, take maximum precision at any recall $\geq r$
    - AP = average of these 11 precision values
    - Formula: $\text{AP} = \frac{1}{11} \sum_{r \in \{0, 0.1, \ldots, 1.0\}} \max_{\tilde{r} \geq r} P(\tilde{r})$
  - **COCO method (all-point interpolation):**
    - More accurate: integrate over all points in PR curve
    - For each unique recall value, compute maximum precision at that recall or higher
    - AP = area under this monotonically decreasing curve
    - Formula: $\text{AP} = \int_0^1 P(r) dr$ where $P(r) = \max_{\tilde{r} \geq r} P(\tilde{r})$
  - **Practical computation:**
    1. Sort predictions by confidence (descending)
    2. For each prediction, compute cumulative TP/FP
    3. Compute precision/recall at each point
    4. Apply interpolation and integrate

- **mAP (mean Average Precision):**
  - **VOC-style:** $\text{mAP} = \frac{1}{N} \sum_{c=1}^{N} \text{AP}_c$ where $N$ is number of classes, computed at single IoU threshold (typically 0.5)
  - **COCO-style:** $\text{mAP} = \frac{1}{N} \sum_{c=1}^{N} \frac{1}{M} \sum_{t=1}^{M} \text{AP}_{c,t}$ where $M$ is number of IoU thresholds (0.5:0.95 step 0.05)
  - COCO also reports $\text{AP}_{50}$ (IoU=0.5) and $\text{AP}_{75}$ (IoU=0.75) separately

- **Speed:** FPS / latency / FLOPs / parameters; always mention the speed–accuracy trade-off.

---

## 4. Classical Baselines

- **Sliding window:** multi-scale crops + classifier; huge redundancy and slow.
- **HOG + SVM, DPM:** hand-crafted features with part-based models; better than raw sliding but still heavy.
- **Region proposals (Selective Search, EdgeBoxes, BING):** generate ~2k candidate boxes + classify; set the stage for R-CNN.

---

## 5. Two-Stage Detectors

- **R-CNN (2014):** proposals via Selective Search → warp to fixed size → CNN features → SVM classify + box regressor; multi-stage training, slow inference.
  - **Pipeline:**
    1. Generate ~2000 region proposals per image (Selective Search - bottom-up segmentation)
    2. Warp each proposal to fixed size (e.g., $227 \times 227$)
    3. Extract CNN features (AlexNet pre-trained on ImageNet)
    4. Train SVM classifier on features (one SVM per class)
    5. Train linear regressor for bounding box refinement (one per class)
  - **Problems:**
    - Slow: forward pass through CNN for each proposal (~2000 per image)
    - Multi-stage: not end-to-end trainable
    - Memory: stores all features for SVM training
    - Fixed-size warping distorts aspect ratios
- **Fast R-CNN (2015):** shared backbone feature map; **RoI Pool** to e.g., $7 \times 7$; single network with multi-task loss (class + box); still depends on external proposals.
  - **RoI Pooling:** 
    - Input: RoI coordinates (from proposals) on feature map
    - Process: Divide RoI into fixed grid (e.g., $7 \times 7$), max-pool each bin
    - **Problem:** Quantization error - RoI coordinates are quantized to feature map grid, causing misalignment
    - **Example:** RoI at $(10.7, 20.3, 50.2, 60.8)$ → quantized to $(10, 20, 50, 60)$ → loses sub-pixel precision
  
- **Faster R-CNN (2016):** **RPN (Region Proposal Network)** generates proposals on the shared feature map using anchors.
  - **RoI Align (improvement over RoI Pool):**
    - **Key difference:** No quantization - uses bilinear interpolation to sample exact values
    - **Process:** 
      1. Keep RoI coordinates as floating-point (no rounding)
      2. Divide into grid (e.g., $7 \times 7$ bins)
      3. For each bin, sample 4 points using bilinear interpolation
      4. Max-pool or average-pool the 4 samples per bin
    - **Advantage:** Preserves spatial alignment, crucial for mask prediction (used in Mask R-CNN)
    - **Why it matters:** Small misalignments can significantly hurt mask accuracy
  - **RPN architecture:**
    - Input: Feature map from backbone (e.g., VGG/ResNet)
    - Sliding window: $3 \times 3$ conv over feature map
    - Two parallel heads:
      - **Classification head:** $1 \times 1$ conv → outputs objectness score (object vs background) for $k$ anchors
      - **Regression head:** $1 \times 1$ conv → outputs box deltas $(t_x, t_y, t_w, t_h)$ for $k$ anchors
    - $k$ anchors per location (typically $k=9$: 3 scales × 3 aspect ratios)
  - **Training:**
    - **Positive anchors:** IoU > 0.7 with any GT box, OR highest IoU with a GT box
    - **Negative anchors:** IoU < 0.3 with all GT boxes
    - **Ignored:** 0.3 ≤ IoU ≤ 0.7 (not used in training)
    - Loss: Binary cross-entropy for objectness + Smooth L1 for box regression
    - **Smooth L1 loss (Huber loss variant):**
      ```math
      L_\text{smooth L1}(x) = \begin{cases}
      0.5 x^2 & \text{if } |x| < 1 \\
      |x| - 0.5 & \text{otherwise}
      \end{cases}
      ```
      - **Properties:** 
        - Smooth near zero (like L2) → stable gradients for small errors
        - Linear for large errors (like L1) → less sensitive to outliers than L2
        - Less aggressive than L1, more robust than L2
      - **Why for box regression:** Box coordinates can have large errors during early training; L2 would be dominated by outliers, L1 has non-smooth gradients at zero
  - **Inference:**
    - RPN generates ~2000 proposals (top-N by objectness score)
    - Proposals fed to RoI head (RoI Pool or RoI Align) for final classification and box refinement
  - **Key advantage:** End-to-end trainable, proposals generated on feature map (much faster than external methods)
- **FPN (Feature Pyramid Network, 2017):** top-down + lateral connections → multi-scale feature maps for small / large objects.
  - **Problem:** Single-scale feature maps struggle with objects of different sizes
    - High-resolution features (shallow layers): good for small objects, but lack semantic information
    - Low-resolution features (deep layers): good for large objects, rich semantics, but poor localization
  
  - **Architecture:**
    - **Bottom-up pathway:** Standard CNN backbone (e.g., ResNet) extracts features at multiple scales
      - Stages: $C_2, C_3, C_4, C_5$ (typically at resolutions $1/4, 1/8, 1/16, 1/32$ of input)
    - **Top-down pathway:** 
      - Start from highest level ($P_5$ from $C_5$)
      - Upsample $P_5$ by 2× → combine with $C_4$ via lateral connection → produce $P_4$
      - Repeat: $P_4$ upsampled + $C_3$ → $P_3$, $P_3$ upsampled + $C_2$ → $P_2$
    - **Lateral connections:**
      - $1 \times 1$ conv to reduce channels (match dimensions)
      - Element-wise addition: upsampled top-down + lateral bottom-up
    - **Output:** Multi-scale feature pyramid: $P_2, P_3, P_4, P_5$ (all with same channel depth, e.g., 256)
  
  - **Usage in detection:**
    - Assign objects to pyramid levels based on size
    - Small objects → $P_2$ or $P_3$ (high resolution)
    - Large objects → $P_4$ or $P_5$ (low resolution, rich semantics)
    - Each level handles objects in a specific size range
  
  - **Benefits:**
    - Single network handles all object scales
    - No need for image pyramids (faster)
    - Better accuracy, especially for small objects
- Trade-off: highest accuracy, slower than one-stage; easy exam point: proposal stage vs dense prediction.

---

## 6. One-Stage Detectors

- Goal: **skip proposal stage**, perform dense prediction directly on grid / feature maps.
- **SSD (2016):** multi-scale feature maps; default boxes (anchors) per location; class + box prediction per anchor.
- **RetinaNet (2017):** adds **focal loss** to fix extreme foreground / background imbalance; still anchor-based; uses FPN backbone.
  - **Problem:** One-stage detectors suffer from extreme class imbalance (e.g., 1000:1 background to foreground ratio).
  - **Standard CE loss:** $L_{CE} = -\log(p_t)$ where $p_t = p$ if $y=1$, else $1-p$.
  - **Focal loss derivation:**
    - Start with weighted CE: $L = -\alpha_t \log(p_t)$
    - Add modulating factor: $L_{focal} = -\alpha_t (1-p_t)^\gamma \log(p_t)$
    - $(1-p_t)^\gamma$ down-weights easy examples:
      - When $p_t \to 1$ (easy example), $(1-p_t)^\gamma \to 0$ → loss is small
      - When $p_t \to 0$ (hard example), $(1-p_t)^\gamma \to 1$ → loss is large
    - **Hyperparameters:** $\alpha$ balances class importance (typically 0.25), $\gamma$ focuses on hard examples (typically 2.0)
  - **Architecture:**
    - Backbone: ResNet + FPN (multi-scale features)
    - Two subnetworks:
      - Classification subnet: predicts class probability per anchor
      - Box regression subnet: predicts box deltas per anchor
    - Anchor design: 3 scales × 3 aspect ratios = 9 anchors per location
  - **Training:** Focal loss for classification, smooth L1 for box regression
  - **Result:** Achieves accuracy comparable to two-stage detectors while being faster.
- Strengths: simple pipeline, fast. Weaknesses: class imbalance, harder small-object recall (mitigated by FPN, focal loss, better anchors).

---

## 7. YOLO Lineage

- Philosophy: **single forward pass**, treat detection as regression.

- **YOLOv1 (You Only Look Once, 2016):**
  - **Core idea:** Divide image into $S \times S$ grid (VOC often $7 \times 7$), each cell predicts $B$ bounding boxes (typically 2) and $C$ class probabilities (VOC: 20 classes).
  - **Output tensor:** $S \times S \times (B \cdot 5 + C)$ where:
    - Each of $B$ boxes contributes 5 values: $(x, y, w, h, \text{confidence})$
    - $x, y$: center coordinates relative to cell (normalized 0-1)
    - $w, h$: width and height relative to entire image (normalized 0-1)
    - $\text{confidence} = P(\text{object}) \cdot \text{IoU}(\text{pred}, \text{truth})$
    - $C$ class probabilities shared across all boxes in the cell
  - **Loss function (multi-part):**
    ```math
    L = \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2 \right]
    ```
    ```math
    + \lambda_{\text{coord}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} \left[ (\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2 \right]
    ```
    ```math
    + \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{obj}} (C_i - \hat{C}_i)^2
    ```
    ```math
    + \lambda_{\text{noobj}} \sum_{i=0}^{S^2} \sum_{j=0}^{B} \mathbb{1}_{ij}^{\text{noobj}} (C_i - \hat{C}_i)^2
    ```
    ```math
    + \sum_{i=0}^{S^2} \mathbb{1}_{i}^{\text{obj}} \sum_{c \in \text{classes}} (p_i(c) - \hat{p}_i(c))^2
    ```
    - $\lambda_{\text{coord}} = 5$ (emphasizes localization), $\lambda_{\text{noobj}} = 0.5$ (reduces false positives)
    - $\sqrt{w}, \sqrt{h}$ used to reduce penalty for errors in large boxes
    - $\mathbb{1}_{ij}^{\text{obj}}$: indicator that cell $i$ has object and box $j$ is responsible (highest IoU with ground truth)
  - **Training details:**
    - Pre-train on ImageNet classification, then fine-tune on detection
    - Data augmentation: random scaling, translation, exposure/saturation adjustments
    - Learning rate schedule: start high, decay by factor of 10 at certain epochs
  - **Limitations:**
    - At most one object per cell (problematic for small objects or dense scenes)
    - Fixed aspect ratios (struggles with unusual shapes)
    - Coarse localization (7x7 grid limits precision)

- **YOLOv2 / YOLO9000:** use anchors via k-means, higher input resolution (e.g., 448), Darknet-19 backbone, multi-scale training.
- **YOLOv3:** multi-scale detection (e.g., $13 \times 13$, $26 \times 26$, $52 \times 52$), Darknet-53 with residuals, logistic classifiers for objectness and classes.
- **YOLOv4/5/6/7/8:** many engineering improvements (mosaic augmentation, SPP, PANet necks); PyTorch-based implementations (v5+); anchor-free options (e.g., in v8); lighter heads and NAS variants.
- Typical lab-style hyperparameters: confidence filter ≈ $0.4$–$0.5$, NMS IoU ≈ $0.5$, mAP at IoU $= 0.5$ (VOC) or averaged $0.5:0.95$ (COCO).

---

## 8. Anchors and Box Regression

- **Anchor design:** predefined widths / heights (scales and aspect ratios) per location; usually obtained via k-means on ground-truth boxes (distance $= 1 - \text{IoU}$).
- **Matching rules (typical heuristic):**
  - Positive if IoU $\geq 0.5$ with some ground-truth.
  - Negative if IoU $< 0.4$.
  - Ignore / neutral region in between.
- **Box delta parameterization (Faster R-CNN style):**

```math
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}
```
```math
t_w = \log\left(\frac{w}{w_a}\right), \quad t_h = \log\left(\frac{h}{h_a}\right)
```

  - At inference time, invert deltas to recover predicted box from anchor.
- **Anchor-free idea:** predict centers / offsets / distances per pixel (FCOS) or keypoints (CenterNet / CornerNet) to avoid anchor design and tuning.

---

## 9. Non-Maximum Suppression

- **Hard NMS algorithm (detailed):**
  1) Sort all boxes by confidence score (descending, per class if class-aware).
  2) Initialize empty list for kept boxes.
  3) While boxes remain:
     - Take highest-scoring box, add to kept list.
     - Remove all boxes with IoU > threshold (typically 0.5) with this box.
     - Repeat with remaining boxes.
  4) Return kept boxes.
  
  **Pseudocode:**
  ```python
  def nms(boxes, scores, iou_threshold):
      keep = []
      order = argsort(scores, descending=True)
      while order:
          i = order[0]
          keep.append(i)
          ious = compute_iou(boxes[i], boxes[order[1:]])
          order = order[1:][ious < iou_threshold]
      return keep
  ```
  
  **Time complexity:** $O(n^2)$ worst case, but typically much better with early stopping.

- **Soft-NMS:**
  - Instead of hard removal, decay scores of overlapping boxes:
    ```math
    s_i = \begin{cases}
    s_i \cdot e^{-\frac{\text{IoU}(b_i, b_m)^2}{\sigma}} & \text{if IoU} > \text{threshold} \\
    s_i & \text{otherwise}
    \end{cases}
    ```
  - Gradually reduces score rather than removing immediately.
  - **Advantage:** Preserves detections when objects are close together (avoids suppressing correct nearby detections).
  - **Parameter:** $\sigma$ controls decay rate (typically 0.5).

- **DIoU-NMS / CIoU-NMS:**
  - **DIoU (Distance IoU):** Considers center distance between boxes:
    ```math
    \text{DIoU} = \text{IoU} - \frac{\rho^2(b, b^{gt})}{c^2}
    ```
    where $\rho$ is center distance, $c$ is diagonal of smallest enclosing box.
  - **CIoU (Complete IoU):** Also considers aspect ratio:
    ```math
    \text{CIoU} = \text{IoU} - \frac{\rho^2}{c^2} - \alpha v
    ```
    where $v$ measures aspect ratio consistency, $\alpha$ is weight.
  - More accurate suppression when boxes have different aspect ratios.

- **Class-aware vs class-agnostic:**
  - **Class-aware:** Apply NMS separately per class (most common).
  - **Class-agnostic:** Apply NMS across all classes (faster, but may suppress boxes of different classes if they overlap).

- **Purpose:** Remove duplicate detections of the same object while keeping the best one. Critical for dense detectors that produce many overlapping predictions.

---

## 10. Modern Architectures

- **Anchor-free:** FCOS (per-pixel regression + center-ness), CenterNet (heatmap of object centers), CornerNet (corner heatmaps).
- **Transformers:** **DETR** reframes detection as set prediction; learned object queries, encoder–decoder attention; no anchors and no NMS; requires longer training but has a very clean pipeline.
- **Hybrid improvements:** Deformable DETR (faster convergence), DINO, Sparse R-CNN (learned proposals), RT-DETR (real-time focus).

---

## 11. From Lecture to Lab

- **Notebook: `lab8_YOLO/YOLO.ipynb`**
  - Output head matches YOLOv1: $S = 7$, $B = 2$, $C = 20$; prediction tensor $S \times S \times (B \cdot 5 + C)$.
  - Loss components:
    - Coordinate loss on $x, y$ (cell-relative) and $\sqrt{w}, \sqrt{h}$.
    - Objectness confidence loss for the responsible box.
    - No-object confidence penalty elsewhere.
    - Class probability loss.

- **Dataset pipeline: `lab8_YOLO/dataset.py`**
  - CSV lists image + label file; labels are normalized: `class, x_center, y_center, width, height` (all in [0,1]).
  - Builds `label_matrix[S, S, C + 5 * B]` where:
    - First $C$ channels: one-hot class encoding
    - Channel $C$: objectness indicator (1 if cell contains object, 0 otherwise)
    - Channels $C+1$ to $C+4$: box coordinates for first box $(x, y, w, h)$ relative to cell
    - Channels $C+5$ to $C+8$: box coordinates for second box (if $B=2$)
  - **Cell assignment:** Object with center $(x, y)$ assigned to cell $(i, j) = (\lfloor S \cdot y \rfloor, \lfloor S \cdot x \rfloor)$
  - **Coordinate conversion:**
    - $x_\text{cell} = S \cdot x - j$ (offset within cell, [0,1])
    - $y_\text{cell} = S \cdot y - i$ (offset within cell, [0,1])
    - $w_\text{cell} = w \cdot S$ (width in cell units, can be > 1 if box spans multiple cells)
    - $h_\text{cell} = h \cdot S$ (height in cell units)
  - **"One object per cell" constraint:** Only first object in a cell is encoded (checked via `label_matrix[i, j, 20] == 0`)

- **Utilities: `lab8_YOLO/utils.py`**
  - IoU computation for midpoint or corner format boxes.
  - NMS over `[class, score, x1, y1, x2, y2]`, sorted by score, class-aware suppression.
  - mAP loop: per-class AP aggregation with IoU threshold and class filtering.

---

## 12. Key Formulas

- IoU:

```math
\text{IoU} = \frac{\text{area(intersection)}}{\text{area(union)}}
```

- Precision / Recall:

```math
\text{Precision} = \frac{TP}{TP + FP}, \quad
\text{Recall} = \frac{TP}{TP + FN}
```

- mAP (VOC-style single IoU):

```math
\text{mAP} = \frac{1}{N}\sum_{\text{class}} \text{AP}_\text{class}
```

- Focal loss (RetinaNet):

```math
L_\text{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
```

- YOLO confidence target:

```math
\text{conf}^\ast = P(\text{object}) \cdot \text{IoU}
```

- Box deltas (again):

```math
t_x = \frac{x - x_a}{w_a}, \quad t_y = \frac{y - y_a}{h_a}, \quad
t_w = \log\left(\frac{w}{w_a}\right), \quad t_h = \log\left(\frac{h}{h_a}\right)
```

---

## 13. Rapid Q&A

1) **Two-stage vs one-stage?** Two-stage: generate proposals then refine / classify (higher accuracy, slower). One-stage: dense prediction (faster, needs class-imbalance handling).  
2) **Role of IoU?** Used to match predictions to ground truth, decide positives / negatives, and drive NMS and AP.  
3) **Why focal loss?** Down-weights easy negatives to address extreme class imbalance in dense detection heads.  
4) **Why anchors?** Cover typical aspect ratios and scales; predict small offsets rather than absolute coordinates.  
5) **What does NMS solve?** Removes duplicate detections of the same object; keeps the highest-scoring box.  
6) **How does FPN help?** Multi-scale feature maps improve detection of both small and large objects.  
7) **Key change in Faster R-CNN vs Fast R-CNN?** RPN generates proposals on the feature map, making the pipeline end-to-end and faster.  
8) **Why is DETR different?** Uses transformers and set prediction, has no anchors or NMS, and uses bipartite matching loss.

---

## 14. Exam-Style Questions

Use these to test yourself without notes; then check back against the sections above.

1. **IoU / AP computation (numeric)**  
   You have 3 predictions for one class on a single image with IoUs $\{0.8, 0.4, 0.6\}$ to the single ground-truth box and confidence scores $\{0.9, 0.7, 0.5\}$.  
   (a) At IoU threshold $0.5$, label each prediction as TP or FP after sorting by confidence.  
   (b) Sketch the precision–recall points and compute AP using the simple trapezoid approximation.

2. **Two-stage vs one-stage (short answer)**  
   (a) List two reasons why two-stage detectors (e.g., Faster R-CNN) tend to have higher AP on COCO than early one-stage detectors.  
   (b) List two reasons why one-stage detectors (e.g., YOLO, RetinaNet) are attractive for real-time applications.

3. **YOLOv1 tensor reasoning**  
   For YOLOv1 with $S = 7$, $B = 2$, $C = 20$:  
   (a) Write the full output tensor shape and explain what each term represents.  
   (b) For an image batch of size 16, what is the shape of the final prediction tensor?  
   (c) Why does YOLOv1 struggle with multiple small objects inside the same grid cell?

4. **Anchor design and matching**  
   (a) Explain why k-means clustering on ground-truth box width/height is a sensible way to design anchors.  
   (b) For a detector with 9 anchors per location, describe what happens if most training data consists of very small objects but anchors are designed for large objects. How would you fix it?

5. **Effect of focal loss (derivation-level)**  
   Starting from the standard CE loss for a binary classification head, derive the focal loss expression and explain intuitively how the $(1 - p_t)^\gamma$ term changes the gradient for easy vs hard examples.

6. **NMS variants and failure modes**  
   (a) Describe a scenario where standard hard NMS might suppress a correct detection.  
   (b) Explain how Soft-NMS or DIoU-NMS modifies this behavior and why that can improve mAP.

