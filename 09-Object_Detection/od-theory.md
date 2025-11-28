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
  - VOC11: 11-point interpolation.
  - COCO: integrates over all points with interpolation.

- **mAP:** mean of AP over classes (and over IoU thresholds for COCO).

- **Speed:** FPS / latency / FLOPs / parameters; always mention the speed–accuracy trade-off.

---

## 4. Classical Baselines

- **Sliding window:** multi-scale crops + classifier; huge redundancy and slow.
- **HOG + SVM, DPM:** hand-crafted features with part-based models; better than raw sliding but still heavy.
- **Region proposals (Selective Search, EdgeBoxes, BING):** generate ~2k candidate boxes + classify; set the stage for R-CNN.

---

## 5. Two-Stage Detectors

- **R-CNN (2014):** proposals via Selective Search → warp to fixed size → CNN features → SVM classify + box regressor; multi-stage training, slow inference.
- **Fast R-CNN (2015):** shared backbone feature map; **RoI Pool** to e.g., $7 \times 7$; single network with multi-task loss (class + box); still depends on external proposals.
- **Faster R-CNN (2016):** **RPN** generates proposals on the shared feature map using anchors; outputs objectness + box deltas; feeds RoI Align / Pool head.
- **FPN (2017):** top-down + lateral connections → multi-scale feature maps for small / large objects.
- Trade-off: highest accuracy, slower than one-stage; easy exam point: proposal stage vs dense prediction.

---

## 6. One-Stage Detectors

- Goal: **skip proposal stage**, perform dense prediction directly on grid / feature maps.
- **SSD (2016):** multi-scale feature maps; default boxes (anchors) per location; class + box prediction per anchor.
- **RetinaNet (2017):** adds **focal loss** to fix extreme foreground / background imbalance; still anchor-based; uses FPN backbone.
- Strengths: simple pipeline, fast. Weaknesses: class imbalance, harder small-object recall (mitigated by FPN, focal loss, better anchors).

---

## 7. YOLO Lineage

- Philosophy: **single forward pass**, treat detection as regression.

- **YOLOv1:**
  - Grid $S \times S$ (VOC often $7 \times 7$), $B$ boxes per cell (e.g., 2), $C$ classes (e.g., 20).
  - Output tensor: $S \times S \times (B \cdot 5 + C)$ (5 = $x, y, w, h, \text{confidence}$).
  - Loss terms:
    - Coordinate loss on $x, y$ (cell-relative), $\sqrt{w}, \sqrt{h}$ for stability.
    - Objectness confidence for “responsible” box (highest IoU).
    - No-object loss on boxes without objects.
    - Class probability loss per cell.
  - Limitation: at most one object per cell.

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

- **Hard NMS algorithm:**
  1) Sort boxes by score (per class).
  2) Keep top box, remove boxes with IoU > threshold.
  3) Repeat until no boxes remain.
- **Variations:** Soft-NMS (score decay instead of hard removal), DIoU / CIoU-NMS (penalize center distance / aspect differences), class-aware vs class-agnostic NMS.
- Purpose: remove duplicate detections of the same object while keeping the best one.

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
  - CSV lists image + label file; labels are normalized: `class, x_center, y_center, width, height`.
  - Builds `label_matrix[S, S, C + 5 * B]`; enforces “one object per cell” with an indicator (e.g., `label_matrix[i, j, 20]`).
  - Converts to cell coordinates: `x_cell = S * x - j`, `y_cell = S * y - i`, `w_cell = w * S`, `h_cell = h * S`.

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

