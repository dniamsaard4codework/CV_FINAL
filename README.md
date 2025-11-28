# Computer Vision Final – Theory Summary (Units 8–13)

This file is your single entry point for theory notes. Each unit below links to the detailed, exam‑ready guide with full explanations, formulas (LaTeX-style), and exam‑style questions.

Use this `summary.md` as:
- A **map** of what to revise for the final.
- A **checklist** to mark off topics you are comfortable with.
- A quick way to jump to the full notes for any unit.

---

## Unit 8 – Image Classification with CNN

Detailed notes: `08-CNN/cnn-theory.md`

**Key topics**
- Why CNNs beat MLPs: sparse connectivity, parameter sharing, translation tolerance, hierarchical features.
- Core math: conv output shapes, receptive field growth, parameter counts, pooling and downsampling, BatchNorm.
- Building blocks: activations (ReLU, LeakyReLU), conv / pooling / stride, normalization, dropout, weight decay, label smoothing, initialization.
- Training: CE/BCE, SGD+momentum, Adam/AdamW, LR schedules (step, cosine, warmup), overfitting countermeasures.
- Architectures: LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, EfficientNet – signatures and differences.
- Transfer learning: freezing / unfreezing, LR ratios for head vs backbone, domain shift, augmentation.
- Failure modes: dying ReLUs, exploding/vanishing gradients, overfitting on small data, divergence, plateaus.

**In the CNN notes**
- Full derivations for:
  - Conv shape: $W_\text{out} = \left\lfloor \frac{W_\text{in} - K + 2P}{S} \right\rfloor + 1$.
  - RF growth: $RF_\text{new} = RF_\text{old} + (K - 1)\prod \text{(strides)}$.
  - Parameter counts for conv vs FC.
- A complete “From Lecture to Lab” mapping to the MLP, CNN on MNIST, and ResNet18 fine‑tuning labs.
- Rapid Q&A and exam‑style questions on shapes, RF, architectures, and transfer learning plans.

---

## Unit 9 – Object Detection

Detailed notes: `09-Object_Detection/od-theory.md`

**Key topics**
- Problem setup: detection vs classification / localization / instance segmentation; box parameterizations.
- Metrics: IoU, precision/recall, AP and mAP (VOC vs COCO; IoU 0.5 vs 0.5:0.95).
- Classical baselines: sliding windows, HOG+SVM, DPM, region proposals (Selective Search, EdgeBoxes, BING).
- Two‑stage detectors: R‑CNN → Fast R‑CNN → Faster R‑CNN; RPN; RoI Pool vs RoI Align; FPN.
- One‑stage detectors: SSD, RetinaNet; class imbalance and focal loss $L_\text{focal} = -\alpha_t(1 - p_t)^\gamma\log p_t$.
- YOLO family: YOLOv1 (grid $S \times S$, $B$ boxes, $C$ classes), YOLOv2/9000, YOLOv3, v4–v8; tensor shapes $S \times S \times (B\cdot 5 + C)$.
- Anchors and box regression: anchor design via k‑means; matching rules; Faster R‑CNN deltas $t_x, t_y, t_w, t_h$.
- NMS: hard NMS, Soft‑NMS, DIoU/CIoU‑NMS; class‑aware vs class‑agnostic.
- Modern detectors: anchor‑free (FCOS, CenterNet, CornerNet), DETR and its transformer-based set prediction.

**In the OD notes**
- LaTeX formulas for IoU, precision/recall, AP/mAP, focal loss, YOLO confidence target, box deltas.
- Clear comparison of two‑stage vs one‑stage detectors (speed vs accuracy, imbalance, design complexity).
- “From Lecture to Lab” mapping to `lab8_YOLO/YOLO.ipynb`, `dataset.py`, `utils.py` for IoU, NMS, mAP.
- Rapid Q&A + exam‑style questions on:
  - Manual IoU/AP computation.
  - YOLOv1 tensor and loss reasoning.
  - Anchor design errors and fixes.
  - Focal loss behavior and NMS variants.

---

## Unit 10 – Object Tracking

Detailed notes: `10-Object_Tracking/ot-theory.md`

**Key topics**
- Detection vs tracking; SOT vs MOT; tracking‑by‑detection pipeline.
- Challenges: appearance change, occlusion, camera motion, clutter, entry/exit, motion blur.
- Classical tools: color-based tracking (HSV), background subtraction, optical flow (brightness constancy $I_x u + I_y v + I_t = 0$), MeanShift/CAMShift, template matching.
- Correlation filter trackers: MOSSE, KCF, DCF variants; FFT and circulant structure; online updates and drift.
- Deep trackers: Siamese (SiamFC, SiamRPN, SiamMask) and transformer trackers (TransT, STARK); similarity matching between template and search region.
- MOT pipeline: detection → Kalman prediction → association (Hungarian, IoU / appearance) → update → track life-cycle (tentative, confirmed, lost).
- Metrics: SOT metrics (success, precision, AUC); MOT metrics (MOTA, MOTP, IDF1, IDSW).

**In the tracking notes**
- LaTeX formulas for optical flow, correlation filter, Kalman prediction/update, MOTA, IDF1.
- Detailed explanation of SORT/DeepSORT/ByteTrack ideas: IoU gating, appearance embeddings, use of low-score detections.
- “From Lecture to Lab” mapping to color trackers and detection+tracking scripts in `lab9_Tracker`.
- Rapid Q&A + exam‑style questions on:
  - Deriving optical flow constraint from brightness constancy.
  - Designing Kalman state and $F$ matrix.
  - Building and gating cost matrices for Hungarian assignment.
  - Correlation filter drift and mitigation strategies.
  - Understanding differences between high‑MOTA vs high‑IDF1 trackers.

---

## Unit 11 – Image Segmentation

Detailed notes: `11-Image_Segmentation/is-theory.md`

**Key topics**
- Segmentation types: semantic vs instance vs panoptic; “stuff” vs “things” categories.
- Classical methods: thresholding, region growing, edge‑based, clustering (K‑means, Mean Shift), graph‑based (Normalized Cuts, GrabCut), watershed, morphology.
- Deep motivation: why encoder‑decoder + skip connections; need for multi‑scale context (FPN/ASPP/PSP).
- Semantic segmentation architectures:
  - FCN (32s/16s/8s) with skip connections.
  - U‑Net: symmetric encoder‑decoder with concatenation skips.
  - SegNet: pooling indices, max‑unpooling.
  - DeepLab v2/v3/v3+: atrous conv, ASPP, light decoder for boundaries.
  - PSPNet: pyramid pooling for multi‑scale context.
- Instance / panoptic: Mask R‑CNN (RoIAlign, mask head), YOLACT, SOLO/SOLOv2, Panoptic FPN.
- Losses: CE, weighted CE, Dice loss, focal loss, combined losses like $L = \lambda_1 L_\text{CE} + \lambda_2 L_\text{Dice}$.
- Metrics: pixel accuracy, mean pixel accuracy, per-class IoU, mIoU, FWIoU, Dice/F1, boundary metrics.

**In the segmentation notes**
- LaTeX for CE, Dice, focal, combined loss, IoU/mIoU, pixel accuracy, Dice loss.
- Intuitive discussion of when to favor Dice/focal (e.g., small or highly imbalanced classes).
- “From Lecture to Lab” mapping to classical segmentation and U‑Net/SAM labs.
- Rapid Q&A + exam‑style questions on:
  - Numeric comparison of pixel acc vs IoU vs Dice.
  - U‑Net vs DeepLab architecture choices.
  - Loss design for tiny objects (tumors).
  - Semantic vs instance vs panoptic outputs for street scenes.
  - Focusing on boundary quality in training/evaluation.

---

## Unit 12 – Generative Models (AE, VAE, GAN)

Detailed notes: `12-Generative_Models/gm-theory.md`

**Key topics**
- Generative vs discriminative: modeling $p(x)$ or $p(x, z)$ vs $p(y \mid x)$; motivation in vision (synthesis, augmentation, anomaly detection).
- Autoencoders (AE):
  - Encoder $f_\phi(x)$, decoder $g_\theta(z)$, bottleneck latent $z$.
  - Reconstruction loss (MSE/BCE), compression vs overfitting trade‑off.
- Variational Autoencoders (VAE):
  - Latent variable model $p(z)$, $p_\theta(x \mid z)$; approximate posterior $q_\phi(z \mid x)$.
  - ELBO derivation and interpretation: reconstruction term + KL regularizer.
  - Reparameterization trick; closed-form KL for Gaussians; $\beta$‑VAE idea (control KL weight).
- GANs:
  - Generator $G(z)$, discriminator $D(x)$; minimax objective and non‑saturating variant.
  - Optimal discriminator $D^\ast(x)$ and connection to JS divergence.
  - Conditional GAN (cGAN), DCGAN architecture guidelines.
- Training issues: mode collapse, vanishing gradients, discriminator/generator imbalance, evaluation metrics (IS, FID).

**In the generative-model notes**
- LaTeX ELBO, GAN objectives, KL formulas, non‑saturating loss, etc.
- “From Lecture to Lab” mapping to `lab11.1_GAN.ipynb` and `lab11.2_VAE.ipynb` (GAN and VAE implementations).
- Rapid Q&A + exam‑style questions on:
  - Deriving ELBO and interpreting KL term.
  - $\beta$‑VAE behavior.
  - GAN optimal discriminator and JS divergence.
  - Diagnosing GAN failures and proposing fixes.
  - Designing a conditional GAN for class‑conditioned generation.

---

## Unit 13 – 3D Vision

Detailed notes: `13-3D_Vision/3d-theory.md`

**Key topics**
- 3D representations: depth maps, point clouds, meshes, voxels, implicit fields; pros/cons of each.
- Depth estimation:
  - Stereo: disparity $d = x_L - x_R$, depth $Z = \frac{fB}{d}$; effect of noise on depth.
  - Monocular depth networks and scale ambiguity.
  - Active sensors: structured light, ToF, LiDAR.
- 3D reconstruction:
  - SfM: feature matching, relative pose, triangulation, bundle adjustment.
  - MVS: dense matching, surface reconstruction.
  - SLAM: visual/RGB‑D/LiDAR SLAM, front‑end vs back‑end vs mapping, loop closure.
- Point clouds:
  - Properties, traditional processing (downsampling, normals, registration/ICP).
  - Why grids/CNNs struggle; need for permutation invariance.
- PointNet / PointNet++:
  - Shared MLP + symmetric max-pooling; T‑Net for alignment.
  - Hierarchical set abstraction in PointNet++.
- Meshes and rendering:
  - Mesh data structures, face and vertex normals, mesh operations.
  - Pinhole camera model, intrinsics matrix, Phong shading equation.
  - Differentiable rendering and common numerical issues (face winding, z‑fighting, near/far planes).
- NeRF:
  - Neural radiance field as MLP $f(x, d) \to (c, \sigma)$.
  - Volume rendering equations with transmittance $T_i$, hierarchical sampling.

**In the 3D notes**
- LaTeX for stereo depth, projection, normals, epipolar constraint $x'^\top F x = 0$, Phong shading, NeRF rendering.
- “From Lecture to Lab” mapping to PyTorch3D rendering, Open3D mesh work, and PointNet classification labs.
- Rapid Q&A + exam‑style questions on:
  - Stereo geometry derivation and uncertainty.
  - PointNet permutation invariance and PointNet++ hierarchies.
  - Full rendering pipeline and intrinsics usage.
  - Comparing differentiable rasterization vs NeRF volume rendering.
  - Relating SfM and SLAM tasks and assumptions.

---

## How to Use This Summary for Final Prep

- Step 1: For each unit (8–13), read the **Exam Checklist** in the corresponding `*-theory.md` file and tick off items you can explain from memory.
- Step 2: Use the **Key Formulas** sections to quickly refresh equations, then immediately attempt related **Exam‑Style Questions** without looking.
- Step 3: When you get stuck on a question, jump back to the detailed explanation in that unit’s theory file and connect it to the labs via the **From Lecture to Lab** section.
- Step 4: In the last days before the exam, use only this `summary.md` + the Rapid Q&A sections as a high‑speed revision loop.
