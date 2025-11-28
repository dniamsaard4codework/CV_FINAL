# Unit 11: Image Segmentation (Exam-Ready Guide)

This is a drill-friendly sheet: start with the checklist, then use the architecture / loss sections to practice short answers and quick derivations. Equations are formatted with LaTeX-style math.

## Table of Contents
1. [Exam Checklist](#1-exam-checklist)
2. [Segmentation Types](#2-segmentation-types)
3. [Classical Methods](#3-classical-methods)
4. [Deep Learning Motivation](#4-deep-learning-motivation)
5. [Semantic Architectures](#5-semantic-architectures)
6. [Instance and Panoptic Approaches](#6-instance-and-panoptic-approaches)
7. [Losses](#7-losses)
8. [Metrics](#8-metrics)
9. [From Lecture to Lab](#9-from-lecture-to-lab)
10. [Key Formulas](#10-key-formulas)
11. [Rapid Q&A](#11-rapid-qa)

---

## 1. Exam Checklist

- Distinguish **semantic vs instance vs panoptic** and give an example of each.
- Explain why **encoder–decoder + skip connections** help recover details.
- State how **dilated convolutions, ASPP, and pyramid pooling** enlarge receptive field without losing resolution.
- Compare **U-Net vs FCN vs DeepLab vs PSPNet** signatures.
- Choose / justify a **loss** for class imbalance or boundary quality (CE, weighted CE, Dice, focal).
- Compute **IoU / mIoU, Dice, pixel accuracy**; know what each metric is biased toward.
- Recall how **Mask R-CNN** differs from Faster R-CNN and why RoIAlign matters.

---

## 2. Segmentation Types

| Type | Per-pixel class? | Instance ID? | Example |
|---|---|---|---|
| Semantic | Yes | No | All dogs = “dog” |
| Instance | Yes | Yes | Dog1, Dog2 separated |
| Panoptic | Stuff via semantic; things via instance | For “things” | Roads merged, cars separated |

Applications: medical imaging, autonomous driving, satellite imagery, AR, industrial inspection, etc.

---

## 3. Classical Methods

- **Thresholding:** global / adaptive / Otsu;  
  $g(x, y) = 1$ if $f(x, y) > T$, else $0$.
- **Region-based:** region growing, split-and-merge; rely on local homogeneity.
- **Edge-based:** Canny / Sobel edges + contour closing + region filling.
- **Clustering:** K-means (color / position), Mean Shift (mode seeking, bandwidth-sensitive).
- **Graph-based:** Normalized Cuts; GrabCut (interactive GMM + graph cut).
- **Watershed:** gradient as topography; flood from markers; dams correspond to boundaries.
- **Morphology:** opening / closing / dilation / erosion to denoise or refine binary masks; often used as post-processing.

---

## 4. Deep Learning Motivation

- CNNs can learn **contextual + semantic** features and handle large appearance variation.
- Challenge: encoder downsamples via stride / pooling → lose spatial detail; solution: **decoder / upsampling + skip connections**.
- Need **multi-scale context** for small vs large objects → FPN / ASPP / pyramid pooling.

---

## 5. Semantic Architectures

- **FCN (Fully Convolutional Networks):**
  - Replace FC layers with $1 \times 1$ conv layers; use deconv / upsampling to reach input resolution.
  - Variants:
    - FCN-32s: direct upsampling from deepest feature map (coarse).
    - FCN-16s: skip from Pool4, better detail.
    - FCN-8s: skips from Pool3 + Pool4, best detail among FCN variants.

- **U-Net:**
  - **Architecture:**
    - **Encoder (contracting path):** 
      - Repeated: $3 \times 3$ conv → ReLU → $3 \times 3$ conv → ReLU → max-pool $2 \times 2$
      - Channels: 64 → 128 → 256 → 512 → 1024 (doubles at each downsampling)
      - Each level reduces spatial size by 2, increases channels by 2
    - **Decoder (expanding path):**
      - Repeated: upsampling → concatenate with skip → $3 \times 3$ conv → ReLU → $3 \times 3$ conv → ReLU
      - Channels: 1024 → 512 → 256 → 128 → 64 (halves at each upsampling)
      - Upsampling typically via transposed convolution or bilinear interpolation
    - **Transposed convolution (deconvolution):**
      - **Not true deconvolution:** It's a learnable upsampling operation
      - **How it works:**
        1. Insert zeros between input pixels (stride > 1)
        2. Apply standard convolution
        3. Output size: $H_\text{out} = (H_\text{in} - 1) \times S - 2P + K$ where $S$ is stride, $P$ is padding, $K$ is kernel size
      - **Example:** Input $4 \times 4$, kernel $3 \times 3$, stride $2$, padding $1$ → output $7 \times 7$ or $8 \times 8$ (depending on output padding)
      - **Advantage:** Learnable upsampling (can learn optimal interpolation)
      - **Disadvantage:** Can produce checkerboard artifacts if stride doesn't divide kernel size
      - **Alternative:** Bilinear/nearest upsampling + regular conv (simpler, fewer artifacts)
    - **Skip connections:** Concatenate encoder feature maps with decoder at same resolution
      - Provides fine-grained spatial information to decoder
      - Helps recover details lost during downsampling
  
  - **Key design choices:**
    - Symmetric architecture (encoder and decoder mirror each other)
    - Concatenation (not addition) for skip connections (preserves all information)
    - No fully connected layers (fully convolutional)
    - Output size matches input size (for dense prediction)
  
  - **Why it works:**
    - Encoder learns high-level semantic features (what/where)
    - Decoder upsamples to full resolution
    - Skip connections provide precise localization (exact boundaries)
    - Combination: semantic understanding + spatial precision
  
  - **Applications:** Very popular in medical imaging (precise boundaries crucial), low-data regimes (efficient use of data), and when input/output sizes must match.

- **SegNet:**
  - Stores pooling indices to perform max-unpooling in decoder.
  - More memory-efficient than full skip connections but less rich in details than U-Net.

- **DeepLab (v2 / v3 / v3+):**
  - **Atrous (dilated) convolutions:**
    - Standard conv: kernel samples every pixel
    - Dilated conv: kernel samples with gaps (dilation rate $r$)
    - **Receptive field:** $RF = (K - 1) \times r + 1$ where $K$ is kernel size
    - Example: $3 \times 3$ conv with $r=2$ has RF of $5 \times 5$ but only 9 parameters
    - **Advantage:** Larger RF without downsampling (preserves resolution) or more parameters
    - **Implementation:** Insert zeros between kernel elements, effectively sparse sampling
  
  - **ASPP (Atrous Spatial Pyramid Pooling):**
    - Parallel branches with different dilation rates: $r \in \{6, 12, 18, 24\}$
    - Each branch: $1 \times 1$ conv + $3 \times 3$ dilated conv (rate $r$)
    - Image-level pooling: Global Average Pooling → $1 \times 1$ conv → bilinear upsample
    - All branches concatenated → $1 \times 1$ conv to reduce channels
    - **Purpose:** Captures multi-scale context (small objects via small $r$, large objects via large $r$)
  
  - **DeepLab v3 improvements:**
    - Batch normalization in ASPP
    - Better dilation rate selection
    - Improved backbone (ResNet with atrous conv in last blocks)
  
  - **DeepLab v3+ additions:**
    - Light decoder: Uses low-level features (from early layers) to refine boundaries
    - Encoder-decoder structure: Encoder = DeepLab v3, Decoder = simple upsampling path
    - Better boundary accuracy while maintaining efficiency

- **PSPNet:**
  - **Pyramid pooling** (e.g., $1 \times 1$, $2 \times 2$, $3 \times 3$, $6 \times 6$ bins); features from each bin are upsampled and concatenated.
  - Explicitly mixes global context and local detail.

---

## 6. Instance and Panoptic Approaches

- **Mask R-CNN:**
  - Extends Faster R-CNN with a parallel mask head.
  - **Architecture:**
    - Backbone: ResNet + FPN (multi-scale features)
    - RPN: generates proposals (same as Faster R-CNN)
    - RoI head: three parallel branches
      - Classification branch: class prediction
      - Box regression branch: box refinement
      - **Mask branch:** $K$ binary masks (one per class) at $28 \times 28$ resolution
  - Uses **RoIAlign** (no quantization) instead of RoI Pool to preserve spatial alignment for masks.
    - Critical for masks: pixel-level accuracy requires precise alignment
    - RoI Pool quantization would cause misalignment and blurry masks
  - **Mask prediction:**
    - For each RoI, predicts $K$ binary masks (one per class)
    - Uses class prediction to select the appropriate mask
    - Upsampled to RoI size for final output
  - **Loss:** Multi-task: classification + box regression + mask segmentation (binary cross-entropy per pixel)
  - Outputs class, box, and mask per RoI.

- **YOLACT, SOLO / SOLOv2:**
  - YOLACT: learn a small set of prototype masks + per-instance coefficients, then linearly combine.
  - SOLO: treats instance segmentation as category-aware per-pixel instance prediction on a grid.

- **Panoptic FPN:**
  - Combines a semantic segmentation branch for “stuff” and an instance branch for “things”.
  - Merges them with rules to resolve overlaps, typically prioritizing instances.

---

## 7. Losses

- **Cross-Entropy (CE):**

  ```math
  L_\text{CE} = -\sum_c y_c \log p_c
  ```

  per pixel; often combined with class weights to handle imbalance.

- **Dice Loss:**

  **Dice coefficient (similarity metric):**
  ```math
  \text{Dice} = \frac{2|P \cap G| + \varepsilon}{|P| + |G| + \varepsilon} = \frac{2 \sum p \cdot g + \varepsilon}{\sum p + \sum g + \varepsilon}
  ```
  where $P$ is predicted mask, $G$ is ground truth mask, $p, g$ are pixel values (0 or 1), $\varepsilon$ prevents division by zero.

  **Dice loss:**
  ```math
  L_\text{Dice} = 1 - \text{Dice}
  ```

  **Properties:**
  - Range: $[0, 1]$ (0 = perfect match, 1 = no overlap)
  - **Advantages:**
    - Directly optimizes overlap (what we care about in segmentation)
    - Robust to class imbalance (small foreground regions contribute significantly)
    - Smooth gradient (differentiable)
  - **Disadvantages:**
    - Can be unstable when predictions and ground truth are both small
    - May not penalize boundary errors as strongly as pixel-wise losses

  **Multi-class extension:**
  ```math
  L_\text{Dice} = 1 - \frac{1}{C} \sum_{c=1}^{C} \frac{2 \sum p_c \cdot g_c + \varepsilon}{\sum p_c + \sum g_c + \varepsilon}
  ```
  where $C$ is number of classes, $p_c, g_c$ are class-specific masks.

  **Good for:** Overlap optimization, class imbalance (e.g., small organs in medical imaging), when precise boundaries are less critical.

- **Focal Loss:**

  ```math
  L_\text{focal} = -\alpha_t (1 - p_t)^\gamma \log(p_t)
  ```

  Focuses on hard-to-classify pixels.

- **Combined losses:**

  ```math
  L = \lambda_1 L_\text{CE} + \lambda_2 L_\text{Dice}
  ```

  Common in medical tasks to balance region coverage and boundary accuracy.

---

## 8. Metrics

- **Pixel Accuracy:** $\text{correct} / \text{total}$ (biased to majority class).
- **Mean Pixel Accuracy:** average per-class accuracy.
- **IoU per class:**

  ```math
  \text{IoU}_c = \frac{TP_c}{TP_c + FP_c + FN_c}
  ```

  **mIoU** is mean IoU across classes.

- **FWIoU:** frequency-weighted IoU; weights classes by frequency.
- **Dice / F1:**

  ```math
  \text{Dice} = \frac{2TP}{2TP + FP + FN}
  ```

  closely related to IoU, more focused on overlap.
- **Boundary metrics:** Boundary IoU or Boundary F1 emphasize edge quality and thin structures.

---

## 9. From Lecture to Lab

- **Classical (`lab10.1_Segmentation_Classical.ipynb`):**
  - Mean-shift / clustering in color space, thresholding, and morphological cleanup.
  - Visualize label images, cluster centers, and boundary effects.

- **Deep (`lab10.2_Segmentation_Deep_learning.ipynb`):**
  - **Model:** U-Net on a road / scene dataset; encoder–decoder with skip concats; conv blocks + transposed conv for upsampling.
  - **Transforms:** joint transforms for image and mask (flips, rotations) to preserve alignment.
  - **Loss:** CE or Dice depending on experiment; monitor validation loss and IoU.

- **Segment Anything (SAM) in `lab10.2`:**
  - Promptable model that takes points / boxes / masks as prompts.
  - Use SAM when you have few labels but can interact; use U-Net when you need a fully automatic, task-specific model.

---

## 10. Key Formulas

- Dice:

  ```math
  \text{Dice} = \frac{2TP}{2TP + FP + FN}
  ```

- IoU:

  ```math
  \text{IoU} = \frac{TP}{TP + FP + FN}
  ```

- mIoU:

  ```math
  \text{mIoU} = \frac{1}{C} \sum_c \text{IoU}_c
  ```

- Pixel accuracy:

  ```math
  \text{PixelAcc} = \frac{\text{correct}}{\text{total}}
  ```

- Cross-entropy:

  ```math
  L_\text{CE} = -\sum_c y_c \log(p_c)
  ```

- Dice loss:

  ```math
  L_\text{Dice} = 1 - \frac{2 \sum p g + \varepsilon}{\sum p + \sum g + \varepsilon}
  ```

---

## 11. Rapid Q&A

1) **Semantic vs instance segmentation?** Semantic assigns each pixel a class label shared across objects; instance also separates each object into its own mask.  
2) **Why encoder–decoder with skips?** Encoder builds high-level semantics; decoder upsamples; skips restore spatial detail and produce sharper boundaries.  
3) **Advantage of dilated convolutions?** Larger receptive field without further downsampling, cheaper than using very large kernels.  
4) **Why is Dice loss good for imbalance?** It directly optimizes overlap; small foreground regions contribute significantly to the loss.  
5) **Role of RoIAlign in Mask R-CNN?** Avoids quantization from RoI Pool, which is crucial for high-quality, well-aligned masks.  
6) **How does ASPP help?** Provides multi-scale context via different dilation rates and image-level pooling, improving performance on objects of various sizes.

---

## 12. Exam-Style Questions

1. **Metric comparison (conceptual + numeric)**  
   You segment a binary object class and obtain the following counts: $TP = 80, FP = 20, FN = 40$.  
   (a) Compute pixel accuracy, IoU, and Dice.  
   (b) Explain which metric is more forgiving when the foreground is very small compared to the background.

2. **Architecture design question**  
   For a road-scene semantic segmentation task:  
   (a) Sketch a U-Net style architecture (depth, skips, number of channels) and explain how skip connections improve thin structures like lane markings.  
   (b) Explain when you might prefer DeepLabv3+ over a vanilla U-Net.

3. **Loss choice in medical imaging**  
   You are segmenting small tumors that occupy $\ll 1\%$ of the image.  
   (a) Explain why plain cross-entropy can perform poorly in this setting.  
   (b) Propose a combined loss of the form $L = \lambda_1 L_\text{CE} + \lambda_2 L_\text{Dice}$ and discuss how you might choose $\lambda_1, \lambda_2$.

4. **Instance vs panoptic segmentation**  
   Given a street scene containing road, sidewalk, cars, and pedestrians:  
   (a) Describe what the output of a semantic segmentation model looks like for this scene.  
   (b) Describe the output of an instance segmentation model.  
   (c) Describe the output of a panoptic segmentation model and how it combines the two.

5. **Boundary quality emphasis**  
   Explain how you could modify training or evaluation to focus more on boundary quality (thin structures, edges) rather than overall region accuracy. Mention at least one boundary-aware loss or metric.

