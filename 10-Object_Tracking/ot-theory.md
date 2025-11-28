# Unit 10: Object Tracking (Exam-Ready Guide)

Use this sheet to rehearse definitions, algorithms, and the lab wiring between detection and tracking. Equations are formatted with LaTeX-style math.

## Table of Contents
1. [Exam Checklist](#1-exam-checklist)
2. [Detection vs Tracking](#2-detection-vs-tracking)
3. [Challenges](#3-challenges)
4. [Classical Tracking Toolbox](#4-classical-tracking-toolbox)
5. [Correlation Filter Trackers](#5-correlation-filter-trackers)
6. [Deep Trackers](#6-deep-trackers)
7. [Multi-Object Tracking Pipeline](#7-multi-object-tracking-pipeline)
8. [Metrics](#8-metrics)
9. [From Lecture to Lab](#9-from-lecture-to-lab)
10. [Key Formulas](#10-key-formulas)
11. [Rapid Q&A](#11-rapid-qa)

---

## 1. Exam Checklist

- Define **tracking** vs **detection** (temporal continuity + IDs).
- Explain **tracking-by-detection** and why it is popular in practice.
- Describe **optical flow assumptions** and the basic equation.
- Summarize **MeanShift / CAMShift**, background subtraction, template matching.
- Explain **correlation filters** (FFT, circulant structure) and why they are fast.
- Outline **Siamese** and **transformer** trackers and their matching mechanisms.
- Walk through **MOT pipeline:** predict → associate (Hungarian / IoU) → update → manage tracks.
- Compute **MOTA, MOTP, IDF1** and interpret ID switches.

---

## 2. Detection vs Tracking

| Aspect | Detection | Tracking |
|---|---|---|
| Input | Single frame | Video sequence |
| Output | Boxes | Boxes + IDs over time |
| Temporal use | None | Motion + appearance |
| Speed need | Flexible | Often real-time |
| Init | Automatic | Manual box (SOT) or detector (MOT) |

**Tracking-by-detection:** run a detector on each frame → associate detections between frames → maintain track states. This leverages strong detectors and handles missed detections by using motion / appearance continuity.

---

## 3. Challenges

- Appearance changes: illumination, pose, deformation, and scale changes.
- Occlusion: partial / full, self / inter-object.
- Motion: fast, abrupt, low frame rate, camera motion.
- Background clutter, similar-looking objects, out-of-view exits / entries, motion blur.

---

## 4. Classical Tracking Toolbox

- **Color-based (HSV):** threshold in HSV + morphology + contour / centroid tracking; simple but sensitive to lighting and color changes.
- **Background subtraction:** frame differencing, MOG2 (Gaussian mixture), KNN; assumes mostly static background; fails with camera motion.
- **Optical flow:** brightness constancy

```math
I_x u + I_y v + I_t = 0
```

  - Assumes small motion and spatial coherence; Lucas–Kanade for sparse or dense flow.
- **MeanShift / CAMShift:** iteratively shift a window to the density peak (e.g., hue histogram); CAMShift also updates window size using image moments.
- **Template matching:** cross-correlation / NCC / SSD over a search window; works for rigid appearance, fails under large deformations.

---

## 5. Correlation Filter Trackers

- Learn a filter whose **response map peaks** at the target location; exploit **circulant matrices** and FFT for fast training / inference.
- **KCF:** kernelized correlation filter with multi-channel features (HOG, color).
- **MOSSE:** grayscale filters, extremely fast, robust to some appearance change.
- **DCF variants:** spatial regularization (SRDCF), efficient convolution operators (ECO) to reduce boundary effects and noise.
- Online update keeps filters fresh, but performance drops with large scale / appearance change or long-term occlusion.

---

## 6. Deep Trackers

- **Siamese family (SiamFC, SiamRPN, SiamMask):**
  - Shared CNN encodes a **template** (target crop) and a **search region**.
  - Cross-correlation produces a similarity map; RPN heads classify / regress boxes; SiamMask adds segmentation masks.
  - Pros: fast, robust to moderate appearance change; common in real-time tracking.
- **Transformer trackers (TransT, STARK):**
  - Use attention to model long-range dependencies and interactions between template and search features.
  - Improve robustness under occlusion and background clutter.
- Core idea: learn **similarity matching** between template and search regions rather than relying purely on motion models.

---

## 7. Multi-Object Tracking Pipeline

1) **Detect** objects in each frame using an object detector.  
2) **Predict** track states (e.g., with a Kalman filter tracking position, scale, and velocity).  
3) **Associate** detections and tracks using a cost (IoU distance, appearance embedding distance) and the Hungarian algorithm; gating removes impossible matches.  
4) **Update** matched tracks with detections; **init** new tracks for unmatched detections; **age out** stale tracks that miss too many frames.  
5) Manage track states: “tentative” → “confirmed” → “lost / deleted” depending on consecutive hits / misses.  

- **DeepSORT:** adds deep appearance embeddings to IoU for more robust association with similar objects.  
- **ByteTrack:** reuses low-score detections in a second pass to keep occluded / low-confidence objects alive, reducing ID switches.

---

## 8. Metrics

- **SOT (single-object tracking):**
  - Success rate (percentage of frames with IoU above a threshold).
  - Precision (center location error).
  - AUC of success plot (area under the curve over IoU thresholds).

- **MOT (CLEAR MOT metrics):**
  - **MOTA:**

    ```math
    \text{MOTA} = 1 - \frac{FN + FP + IDSW}{GT}
    ```

    Higher is better; penalizes misses, false positives, and ID switches.

  - **MOTP:** mean IoU of matched boxes (localization accuracy).

  - **IDF1:**

    ```math
    \text{IDF1} = \frac{2 \cdot IDTP}{2 \cdot IDTP + IDFN + IDFP}
    ```

    Focuses on identity preservation across time.

  - **IDSW:** number of track ID changes; lower is better.

---

## 9. From Lecture to Lab

- **Lab scripts (`lab9_Tracker`):**
  - `9.1_simple_object_tracking_with_color.py`: HSV thresholds (e.g., green / yellow), morphology to clean masks, contour / centroid drawing, deque to smooth trajectory, handles frame drops.
  - `9.2_single_object_tracking.py` / `9.3_multi_object_tracking.py`: extend color / simple feature tracking to multiple targets.
  - `9.4_detection_tracking.py`: demonstrates a tracking-by-detection pipeline connecting detectors and trackers.
  - `tracker_utils.py`: shared helper functions for drawing, filtering, and bookkeeping.

- **Kalman + SORT-style flow (lecture):**
  - State vector example: $[x, y, s, r, \dot x, \dot y, \dot s]$ (center, scale, aspect ratio, velocities).
  - Predict step:

    ```math
    x_{\text{pred}} = F x_{\text{prev}}
    ```

    Update step:

    ```math
    x = x_{\text{pred}} + K (z - H x_{\text{pred}})
    ```

    where $K$ is the Kalman gain.
  - Assignment: Hungarian algorithm on an IoU-based cost matrix with gating; unmatched detections spawn tentative tracks; unmatched tracks decay and are removed after several misses.

- **Handling similarity and occlusion:**
  - Use IoU gating plus appearance embeddings (DeepSORT) for ambiguous cases.
  - Use low-score detections and longer track age to bridge short occlusions (ByteTrack).

---

## 10. Key Formulas

- Optical flow (brightness constancy):

  ```math
  I_x u + I_y v + I_t = 0
  ```

- Correlation filter (frequency domain, simplified):

  ```math
  H = \frac{\overline{G} \cdot F^\ast}{\overline{F} \cdot F^\ast + \lambda}
  ```

- Kalman update:

  ```math
  x = x_{\text{pred}} + K (z - H x_{\text{pred}})
  ```

- MOTA:

  ```math
  \text{MOTA} = 1 - \frac{FN + FP + IDSW}{GT}
  ```

- IDF1:

  ```math
  \text{IDF1} = \frac{2 \cdot IDTP}{2 \cdot IDTP + IDFN + IDFP}
  ```

---

## 11. Rapid Q&A

1) **SOT vs MOT?** SOT tracks one initialized object; MOT tracks many objects with persistent IDs and automatic birth / death.  
2) **Why use a Kalman filter?** Predicts during occlusion, smooths noisy detections, and handles missing frames.  
3) **What does the Hungarian algorithm solve?** Global optimal assignment between detections and tracks given a cost matrix.  
4) **How to handle occlusion?** Motion prediction, appearance cues, low-score detections (ByteTrack), and track age thresholds.  
5) **Why are correlation filters fast?** They use FFT on circulant data, making per-frame updates and responses very cheap.  
6) **Why track instead of just detect every frame?** Provides smoother IDs, robustness to missed detections, and can reduce compute when detectors are heavy.

---

## 12. Exam-Style Questions

1. **Optical flow reasoning**  
   (a) State the brightness constancy assumption and derive the optical flow constraint equation $I_x u + I_y v + I_t = 0$.  
   (b) Explain why this is not enough to uniquely determine $(u, v)$ at each pixel and how Lucas–Kanade handles this.

2. **Kalman filter in MOT**  
   You model track state as $[x, y, s, r, \dot x, \dot y, \dot s]^\top$.  
   (a) Write down a reasonable linear motion model matrix $F$ (you can ignore $r$ for simplicity).  
   (b) Describe qualitatively what happens to a track state when detections are missing for a few frames (prediction-only).

3. **Assignment and cost matrices**  
   Consider 3 existing tracks and 4 detections with a given IoU matrix.  
   (a) Explain how you would build the cost matrix for the Hungarian algorithm.  
   (b) Describe how you would enforce a minimum IoU threshold (gating) before assignment.  
   (c) What do you do with unmatched tracks and unmatched detections?

4. **Correlation filter tracker behavior**  
   A KCF-style tracker drifts off the target after a severe occlusion.  
   (a) Explain why online filter updates can cause this.  
   (b) Propose at least two strategies to reduce drift (e.g., update scheduling, occlusion detection).

5. **MOTA vs IDF1 interpretation**  
   Give an example where two trackers have similar MOTA but very different IDF1. Explain what this says about their performance and which you would prefer for an application that cares about long-term identity.

