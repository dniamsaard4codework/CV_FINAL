Gotcha — let’s switch to English 😊
Here’s a **big exam-style question bank with answers**, grouped by topic.

You can treat this as a “likely questions” list rather than literally every possible question in the universe, but it covers the common final-exam stuff for:

* Optimization & basic neural nets / CNNs
* Image classification
* Object detection
* Object tracking & motion/change detection
* Image segmentation
* 3D vision
* Generative models (AE, VAE, GAN)

---

## 1. Optimization, Neural Networks, CNN & Image Classification

### Q1. Describe the standard training pipeline for deep learning image classification.

**A:**

1. Collect and preprocess dataset, split into train / validation / test.
2. Choose model architecture (linear classifier, MLP, CNN, ResNet, etc.).
3. Choose loss function (e.g., cross-entropy for classification).
4. Initialize weights (random, Xavier, He, etc.).
5. For each batch:

   * Forward pass → compute predictions.
   * Compute loss between predictions and labels.
   * Backward pass (backprop) → compute gradients.
   * Use optimizer (SGD, Adam, etc.) to update weights.
6. Repeat for many epochs, monitor validation loss/accuracy to detect overfitting.
7. After training, evaluate on test set.

---

### Q2. What is Gradient Descent? Show the update rule.

**A:**
Gradient Descent is an optimization method that iteratively moves parameters in the opposite direction of the gradient of the loss to minimize it:
[
w := w - \eta \nabla_w J(w)
]
where (w) are the parameters, (\eta) is the learning rate, and (J(w)) is the loss.

---

### Q3. Compare Gradient Descent, Stochastic Gradient Descent (SGD), and Mini-batch SGD.

**A:**

* **Full-batch GD**: Uses **all** training examples to compute each gradient step.

  * Pros: Accurate gradient estimate.
  * Cons: Very slow and memory-heavy for large datasets.
* **SGD**: Uses **one** example per update.

  * Pros: Very fast per step, can escape local minima.
  * Cons: Very noisy updates.
* **Mini-batch SGD**: Uses a small batch (e.g. 32, 64).

  * Pros: Balance between speed and stability; vectorization works well on GPU.
  * This is the most commonly used in practice.

---

### Q4. What happens if the learning rate is too small or too large?

**A:**

* Too small:

  * Training is very slow, may get stuck in poor local minima or plateaus.
* Too large:

  * Loss may oscillate or diverge (explode), never converging.

So we tune learning rate or use schedules (decay, cosine) or adaptive methods (Adam, RMSProp, etc.).

---

### Q5. What is backpropagation and how does it relate to the chain rule?

**A:**
Backpropagation is the algorithm used to efficiently compute gradients of the loss with respect to all parameters in a neural network.

* It uses the **chain rule** of calculus to propagate gradients from the output layer backwards through each layer, reusing intermediate derivatives so we don’t recompute everything from scratch.

---

### Q6. Why do we need activation functions? What happens if we remove them?

**A:**

* Activation functions introduce **non-linearity**, allowing the network to approximate complex non-linear functions.
* Without activations, each layer is linear, and a stack of linear layers collapses into a **single linear transformation**. The model would be equivalent to a linear classifier, no matter how many layers.

---

### Q7. Why are CNNs better than fully-connected networks for images?

**A:**

* Images have a **spatial structure** (neighboring pixels are related).
* Fully-connected layers ignore this and require huge numbers of parameters.
* CNNs use:

  * **Local receptive fields**: filters look at local patches.
  * **Weight sharing**: the same filter is used across the whole image.
  * **Translation invariance**: detection of a pattern is location-agnostic.
    This results in far fewer parameters and better performance on visual tasks.

---

### Q8. How do you compute the output size of a 1D or 2D convolution?

**A:**
For each spatial dimension (e.g., width (W)):
[
W_\text{out} = \frac{W - K + 2P}{S} + 1
]
where:

* (W) = input size
* (K) = kernel size
* (P) = padding
* (S) = stride

If you have (C_\text{out}) filters, output shape is (C_\text{out} \times W_\text{out} \times H_\text{out}).

---

### Q9. What is the receptive field of a neuron in a CNN?

**A:**
The receptive field is the region in the **input image** that affects the value of a particular feature (neuron) in some layer.

* As you stack layers, the receptive field grows.
* Example (1D for simplicity, no stride):

  * With kernel size (K) and (L) layers:
    [
    \text{RF} = 1 + L (K - 1)
    ]

---

### Q10. Why is normalization (BatchNorm, LayerNorm) used?

**A:**

* Stabilizes the distribution of activations across layers.
* Makes optimization easier, allows higher learning rates.
* Often speeds up convergence and improves generalization.

---

## 2. Object Detection (R-CNN, Fast/Faster R-CNN, NMS, IoU, mAP, YOLO/SSD)

### Q11. How is object detection different from image classification?

**A:**

* **Classification**: Predicts a single label (or multiple labels) for the whole image.
* **Detection**: Finds **where** and **what** objects are in the image.

  * Outputs: a set of **bounding boxes** and **class labels**, possibly many per image.

---

### Q12. Define IoU (Intersection over Union). How is it used?

**A:**
Given predicted box (B_\text{pred}) and ground-truth box (B_\text{gt}):
[
\text{IoU} = \frac{\text{Area}(B_\text{pred} \cap B_\text{gt})}{\text{Area}(B_\text{pred} \cup B_\text{gt})}
]
Uses:

* To determine if a predicted box is a **True Positive** (e.g. IoU ≥ 0.5).
* For evaluation metrics like mAP.
* Inside NMS to measure overlap between boxes.

---

### Q13. Explain the main steps of R-CNN.

**A:**
**Training / Inference pipeline:**

1. Use a region proposal method (e.g., Selective Search) to generate ~2000 candidate regions.
2. For each region:

   * Crop and resize to fixed size (e.g. 224×224).
   * Feed through a CNN (often pretrained).
   * Extract features from some layer (e.g. fc7).
3. Train one or more SVM classifiers on these features for object classes vs background.
4. Train a bounding-box regressor to refine the coordinates.
5. At test time, repeat steps 1–4 and apply NMS to remove redundant boxes.

Drawback: CNN must run for each region → extremely slow.

---

### Q14. How does Fast R-CNN improve over R-CNN?

**A:**
Fast R-CNN:

1. Run the CNN **once** on the whole image to get a feature map.
2. Project region proposals onto this feature map.
3. Use **ROI Pooling / ROI Align** to get a fixed-size feature for each proposal.
4. Pass those through fully-connected layers to jointly predict class scores and bounding box offsets.

Benefits:

* Shared convolutional computation for all proposals → much faster.

---

### Q15. What is the difference between ROI Pooling and ROI Align?

**A:**

* **ROI Pooling**:

  * Quantizes ROI coordinates to integer bins.
  * Applies max pooling per bin.
  * Causes **quantization errors** → misalignment between proposals and features.
* **ROI Align**:

  * Avoids quantization; uses bilinear interpolation to sample exact float coordinates.
  * More accurate localization; used in Mask R-CNN.

---

### Q16. What does Faster R-CNN add compared to Fast R-CNN?

**A:**
Faster R-CNN replaces external region proposal methods (Selective Search) with a **Region Proposal Network (RPN)**:

* RPN operates on the shared CNN feature map.
* At each spatial location, it uses **anchor boxes** of different scales/ratios.
* For each anchor, predicts:

  * Objectness score (object vs background).
  * Bounding box regression offsets.
* Top proposals are passed to Fast R-CNN head for final classification and bbox refinement.

Result: end-to-end trainable, much faster region proposals.

---

### Q17. Explain Non-Maximum Suppression (NMS).

**A:**
Given many detections of the same class:

1. Sort boxes by confidence score in descending order.
2. Take the highest score box, add it to the final list.
3. Remove all remaining boxes with IoU above a threshold (e.g. 0.5 or 0.7) with this box.
4. Repeat step 2–3 until no boxes remain.

NMS removes redundant overlapping detections.

---

### Q18. What is mAP in object detection?

**A:**

* For each class:

  1. Collect all predicted boxes across the dataset and sort by confidence.
  2. Mark each prediction as TP/FP using IoU and matching rules.
  3. Compute precision–recall curve.
  4. **Average Precision (AP)** = area under this precision–recall curve.
* **mAP** = mean of AP values over all classes.
  Modern COCO-style mAP averages AP over multiple IoU thresholds (e.g. 0.5 to 0.95).

---

### Q19. In one-stage detectors (YOLO, SSD), what is the main idea vs two-stage detectors?

**A:**

* **Two-stage (Faster R-CNN)**:

  1. Generate proposals,
  2. Classify/refine them.
* **One-stage (YOLO, SSD)**:

  * Directly predict class probabilities and bounding boxes **on a dense grid or anchors**, without an intermediate proposal step.
* Pros: Much faster, suitable for real-time detection.
* Cons: Historically somewhat lower accuracy for small/complex objects (but newer versions improved).

---

## 3. Object Tracking & Change Detection

### Q20. Define the object tracking problem.

**A:**
Given a sequence of frames (video), object tracking aims to **estimate the state** (position, velocity, size, etc.) of one or more objects over time, maintaining **consistent identities** across frames.

---

### Q21. Why do we need tracking if we already have object detection?

**A:**
Detectors can:

* Miss objects in some frames (occlusion, blur, lighting).
* Be computationally expensive to run on every frame at high resolution.

Tracking:

* Fills in missing detections (short-term occlusion).
* Provides motion trajectories and velocities.
* Can reduce computation by focusing on predicted areas.

---

### Q22. What are the main components of a tracking system?

**A:**

1. **Detection**: Find object candidates per frame.
2. **Prediction**: Use a motion model (e.g., Kalman filter) to predict the object’s next state.
3. **Data Association**: Match detections to existing tracks (who is who).
4. **Update / Manage tracks**: Update state with matched detection, create new tracks, delete lost ones.

---

### Q23. Explain simple Background Subtraction and its limitations.

**A:**

* Have a background model (B(x,y)) (empty scene).
* For each frame (I_t(x,y)):

  * Compute difference (|I_t - B|).
  * If above threshold → foreground; else → background.

**Limitations:**

* Objects that stop moving become part of the background (or remain forever foreground depending on update).
* Background changes (lighting, moving leaves) cause false detections.
* Shadows and reflections cause problems.
* Single global threshold may not work well on the entire image.

---

### Q24. How is frame differencing different from background subtraction?

**A:**

* **Frame differencing** uses previous frame as reference:
  [
  |I_t(x,y) - I_{t-1}(x,y)|
  ]
* Pros:

  * Adapts quickly to gradual changes.
  * Objects that become stationary disappear from the foreground.
* Cons:

  * Often only detects moving edges (leading/trailing edges).
  * Very small motions or slow global motion may be missed.

---

### Q25. What is adaptive background subtraction using an exponential moving average?

**A:**
Update rule per pixel:
[
B_t = (1 - \alpha) B_{t-1} + \alpha I_t
]

* (\alpha) controls how fast the background adapts.

  * Large (\alpha): adapts quickly but may absorb slow-moving objects into background.
  * Small (\alpha): more stable but adapts slowly to lighting changes.

---

### Q26. Why use Gaussian Mixture Models (GMMs) for background modeling?

**A:**

* A single pixel’s intensity over time may be multi-modal (e.g., swaying tree, periodic motion).
* GMM models the distribution at each pixel as a mixture of several Gaussians.
* Gaussians with high weight and low variance are treated as **background**; others as **foreground**.
  This handles dynamic backgrounds better than simple thresholding.

---

### Q27. What is “tracking-by-detection”?

**A:**

* Run an object detector (or feature detector) in each frame.
* Use tracking methods to **link detections over time** (data association).
* For feature-based: detect interest points (SIFT/ORB), match them between frames, and estimate motion or track objects.

---

### Q28. What is GOTURN (high-level idea)?

**A:**
GOTURN is a deep learning–based tracker that:

* Takes as input:

  * A crop of the target object from the previous frame.
  * A search region from the current frame.
* A regression CNN then outputs the bounding box of the object in the current frame.
  It doesn’t update online per target; it is trained offline and runs very fast.

---

## 4. Image Segmentation

### Q29. How is image segmentation different from classification and detection?

**A:**

* **Classification**: One label for the whole image.
* **Detection**: Bounding boxes + label for each object.
* **Segmentation**: A label for **every pixel**.

Types:

* **Semantic segmentation**: pixels of the same class share one label, instances not separated.
* **Instance segmentation**: differentiates individual instances (e.g., three different persons).
* **Panoptic segmentation**: combines semantic and instance.

---

### Q30. What does “segmentation as clustering” mean? What features can we use per pixel?

**A:**
We represent each pixel as a feature vector, e.g.:
[
f = [R, G, B, x, y, \dots]
]
Then apply clustering (e.g., k-means, mean shift) in this feature space so that similar pixels are grouped into segments.
Features can include color, position, texture, depth, etc.

---

### Q31. Describe the k-means clustering algorithm and its pros/cons for segmentation.

**A:**
**Algorithm:**

1. Choose number of clusters (k), initialize centroids randomly.
2. Assign each data point to the nearest centroid.
3. Update each centroid as the mean of assigned points.
4. Repeat steps 2–3 until convergence.

**Pros:**

* Simple, fast, easy to implement.

**Cons:**

* Must choose (k) beforehand.
* Sensitive to initialization and outliers.
* Does not enforce spatial smoothness; far-apart pixels of same color may be clustered together.

---

### Q32. What is mean shift clustering and how is it different from k-means?

**A:**
Mean shift:

* Views points as samples from a density.
* Iteratively moves each point (or window center) towards the nearest **mode** (peak) of the density by averaging nearby points.
* Points that converge to the same mode form a cluster.

Differences:

* Does **not** require specifying number of clusters (k).
* Can handle arbitrarily shaped clusters.
* More computationally expensive and depends on bandwidth (window size).

---

### Q33. Briefly explain graph-based segmentation with cuts (Min-Cut / Normalized Cut).

**A:**

* Represent image as a graph:

  * Pixels (or superpixels) = nodes.
  * Edges connect neighboring pixels; weights = similarity (e.g., color similarity, distance).
* **Min-Cut**: partition into two sets by cutting edges with minimum total weight.
* **Normalized Cut**: normalizes cut cost by association inside each segment, reducing bias for tiny segments.
  Solving exactly is hard, so eigenvalue-based approximations are used.

---

### Q34. What is a Fully Convolutional Network (FCN) for semantic segmentation?

**A:**

* Replace fully-connected layers in a classification CNN with **convolutional** layers.
* The network outputs a **spatial feature map** where each location corresponds to a region in the input.
* Upsampling (deconvolution, interpolation) is used to return to original resolution and provide per-pixel class scores.
* Compared to sliding-window, FCN is much more efficient because computations are shared.

---

### Q35. Name common upsampling methods in segmentation and their differences.

**A:**

* **Interpolation**:

  * Nearest neighbor, bilinear, bicubic.
  * No learnable parameters; purely geometric.
* **Unpooling**:

  * Uses indices from max-pooling (“switches”) to restore feature positions.
* **Transposed Convolution (Deconvolution)**:

  * Learnable upsampling; a convolution-like operation that increases spatial resolution.

---

### Q36. Intuition for transposed convolution?

**A:**

* Regular convolution: multiplies and sums neighborhood patches, often reducing resolution (with stride > 1).
* Transposed convolution: reverses this process; spreads each input value over a larger output region using a kernel, summing where overlaps occur, so resolution increases.
* Used in decoders of segmentation networks and in generators of GANs.

---

## 5. 3D Vision & 3D Deep Learning

### Q37. What is the difference between 2.5D and full 3D?

**A:**

* **2.5D (depth map)**: assigns a depth value to each pixel in an image; only visible surfaces from one viewpoint.
* **3D representation**: describes the full volume/shape in space (even hidden parts): point clouds, meshes, voxels, implicit functions, etc.

---

### Q38. List common 3D shape representations and their pros/cons.

**A:**

* **Point Cloud**:

  * Pros: Simple, directly from sensors (LiDAR, depth cameras).
  * Cons: No explicit surface connectivity; rendering and surface operations are harder.
* **Polygon Mesh (vertices + faces)**:

  * Pros: Great for graphics; clear surfaces; efficient rendering.
  * Cons: Complex structure; difficult to handle topological changes.
* **Voxel Grid**:

  * Pros: Regular grid; can use 3D CNNs; simple occupancy representation.
  * Cons: Memory grows cubically with resolution.
* **Implicit Representation (e.g. Signed Distance Function)**:

  * Pros: Encodes inside/outside and distance; easy to test occupancy.
  * Cons: Requires extra work for rendering/meshing (marching cubes, etc.).

---

### Q39. What is the difference between explicit and implicit 3D shape representation?

**A:**

* **Explicit**: Represent the surface/volume directly (points, mesh faces, voxels where occupied).
* **Implicit**: Represent the shape via a function (f(x,y,z)) where the surface is the level set (e.g., (f(x,y,z) = 0)).

  * Example: sphere defined by (x^2 + y^2 + z^2 - r^2 = 0).

---

### Q40. What is a Signed Distance Function (SDF)?

**A:**
An SDF is a scalar function (f(p)) that gives the signed distance from point (p) to the surface:

* (f(p) = 0) on the surface.
* (f(p) < 0) inside the object.
* (f(p) > 0) outside the object.

SDFs are widely used in shape reconstruction, collision detection, and implicit neural representations.

---

### Q41. What problem does PointNet solve?

**A:**
PointNet is designed to process point clouds directly, which are **unordered sets** of points.

* It uses shared MLPs on each point and a symmetric aggregation function (e.g., max pooling) across points to produce global features.
* This makes the network invariant to permutations of the input points and suitable for classification/segmentation on point clouds.

---

## 6. Generative Models: AE, VAE, GAN

### Q42. What is an Autoencoder (AE) and what is it used for?

**A:**

* An AE is a neural network with:

  * **Encoder**: maps input (x) to a latent code (z).
  * **Decoder**: maps (z) back to reconstruction (\hat{x}).
* Training objective: minimize reconstruction error (e.g. MSE, cross-entropy).
* Uses: dimensionality reduction, denoising, feature learning, anomaly detection.

---

### Q43. How is a Variational Autoencoder (VAE) different from a standard AE?

**A:**

* AE: encoder outputs a deterministic code (z).
* VAE: encoder outputs parameters of a probability distribution (q_\phi(z|x)) (e.g., Gaussian with mean and variance).
* We sample (z) using the **reparameterization trick**:
  [
  z = \mu(x) + \sigma(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I).
  ]
* Loss = reconstruction loss + KL divergence between (q_\phi(z|x)) and prior (p(z)) (often (\mathcal{N}(0,I))).
* This makes the latent space smooth and allows **sampling** new data (generative model).

---

### Q44. What is the reparameterization trick and why do we need it?

**A:**

* Directly sampling (z \sim q_\phi(z|x)) makes backpropagation through sampling difficult.
* Reparameterization writes:
  [
  z = \mu + \sigma \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I),
  ]
  where (\epsilon) is independent of parameters.
* This allows gradients to flow through (\mu) and (\sigma) while still sampling stochastically.

---

### Q45. Explain GAN architecture and training objective.

**A:**
GAN has two networks:

* **Generator (G(z))**: maps random noise (z) to synthetic samples (e.g., images).
* **Discriminator (D(x))**: outputs probability that (x) is real (from data) vs fake (from (G)).

Training is a minimax game:

* (D) tries to **maximize** correct classification of real vs fake.
* (G) tries to **fool (D)** so that (D(G(z)) \approx 1) (real).

Typical loss:
[
\min_G \max_D , \mathbb{E}*{x \sim p*{\text{data}}}[\log D(x)] + \mathbb{E}_{z \sim p(z)}[\log (1 - D(G(z)))].
]

When training succeeds, (G) learns to generate samples that follow the data distribution.

---

### Q46. List common problems when training GANs.

**A:**

* **Mode collapse**: generator produces limited variety of samples.
* **Training instability**: sensitive to architecture and hyperparameters.
* **Vanishing gradients**: if discriminator becomes too strong.
  Solutions include Wasserstein GAN, gradient penalties, feature matching, etc.

---

If you want, next step I can:

* Turn these into **short flashcards** (Q on one line, 1–3 line answer).
* Or give you **calculation practice** (e.g., compute conv output size, IoU, simple mAP example) with step-by-step solutions.

Alright, let’s explode your question bank with **a lot more conceptual exam questions + compact answers** 🚀

I’ll continue numbering from last time (we stopped at Q46).

---

## A. General Deep Learning & Optimization

### Q47. What is overfitting? How can you detect it?

**A:**

* Overfitting = model learns training data too specifically (including noise) and performs poorly on unseen data.
* Detection: training loss ↓ while validation loss stops improving or ↑; training accuracy ≫ validation accuracy.

---

### Q48. List common techniques to reduce overfitting in deep learning.

**A:**

* More data / data augmentation
* Regularization (L2, L1)
* Dropout
* Early stopping
* Smaller model / fewer parameters
* Batch normalization (indirectly helps)

---

### Q49. What is L2 regularization and how does it affect the weights?

**A:**

* Add term (\lambda |w|_2^2) to the loss.
* Penalizes large weights, pushing them toward smaller values.
* Encourages smoother, less complex models and reduces overfitting.

---

### Q50. Difference between L1 and L2 regularization?

**A:**

* **L1 (Lasso)**: uses (\lambda |w|_1); encourages sparsity (many weights exactly 0).
* **L2 (Ridge)**: uses (\lambda |w|_2^2); shrinks weights continuously but rarely to exactly 0.
* L1 good for feature selection; L2 good for smoothness and stability.

---

### Q51. What is dropout and what problem does it solve?

**A:**

* During training, each neuron is randomly “dropped” (set to 0) with probability p.
* Prevents co-adaptation of neurons, acts like training an ensemble of subnetworks.
* Reduces overfitting and improves generalization.

---

### Q52. Why is cross-entropy loss used for classification instead of MSE?

**A:**

* Cross-entropy is derived from maximum likelihood for categorical distributions.
* Provides stronger gradients when predictions are wrong.
* Works naturally with softmax outputs; MSE can lead to slower learning and poor probability calibration.

---

### Q53. Explain vanishing gradients and how we mitigate them.

**A:**

* In deep networks, backpropagated gradients may become very small, so earlier layers learn extremely slowly.
* Causes: repeated multiplication by small derivatives (e.g., sigmoid).
* Mitigation: ReLU-type activations, residual connections, careful initialization, batch normalization.

---

### Q54. What does batch normalization do and why is it helpful?

**A:**

* Normalizes intermediate activations per mini-batch to have zero mean and unit variance (with learned scale/shift).
* Reduces internal covariate shift, stabilizes learning, allows larger learning rates, often speeds convergence and improves accuracy.

---

### Q55. Explain early stopping and how it works in practice.

**A:**

* Monitor validation loss during training.
* If it stops improving (or gets worse) for N epochs (“patience”), stop training and keep the best model.
* It’s an effective and simple form of regularization.

---

### Q56. What is a confusion matrix and what can it tell you?

**A:**

* Table with rows = true classes, columns = predicted classes.
* Shows counts of TP, FP, FN per class.
* Helps visualize which classes are confused with each other (e.g., cat→dog).

---

### Q57. Define Precision, Recall, and F1-score for binary classification.

**A:**

* Precision = TP / (TP + FP) → “Of predicted positives, how many are correct?”
* Recall = TP / (TP + FN) → “Of all real positives, how many did we find?”
* F1 = harmonic mean of precision and recall:
  [
  F1 = \frac{2PR}{P+R}
  ]

---

### Q58. What is class imbalance and how can you handle it?

**A:**

* Class imbalance: some classes have many more samples than others.
* Solutions:

  * Resampling (oversample minority, undersample majority)
  * Class-weighted loss
  * Focal loss (for detection)
  * Data augmentation on minority classes.

---

### Q59. Why might accuracy be a bad metric for imbalanced data?

**A:**

* A classifier can get very high accuracy by always predicting the majority class, but completely fail on rare classes (e.g., 99% accuracy but 0% recall for minority class).
* Metrics like precision, recall, F1, ROC-AUC, or per-class accuracy are more informative.

---

## B. CNN & Image Classification (Conceptual)

### Q60. Why do we use pooling layers in CNNs?

**A:**

* Reduce spatial resolution → fewer parameters and less computation.
* Introduce invariance to small translations and noise.
* Capture more abstract features at coarser resolution.

---

### Q61. Compare max pooling and average pooling.

**A:**

* **Max pooling**: outputs the maximum value in each window; keeps strong activations; good for detecting presence of features.
* **Average pooling**: outputs the mean; smoother, may blur small strong responses.
* Max pooling is more common in classification CNNs.

---

### Q62. What is “same” padding vs “valid” padding?

**A:**

* **Same padding**: add padding so output size ≈ input size (for stride 1).
* **Valid padding**: no padding; output size shrinks as convolution reduces border pixels.

---

### Q63. What is global average pooling and why use it instead of fully-connected layers?

**A:**

* Global average pooling averages each channel over all spatial positions → converts C×H×W feature map into a C-dimensional vector.
* Reduces parameters (no large FC layer), lowers overfitting, enforces connection between channels and classes in classification architectures (e.g., in Network-in-Network, ResNet variants).

---

### Q64. Describe the main ideas of AlexNet, VGG, and ResNet at a high level.

**A:**

* **AlexNet**: first very deep CNN that won ImageNet; used ReLU, dropout, large conv filters and pooling.
* **VGG**: uses many 3×3 conv layers stacked deeply; simple, uniform architecture.
* **ResNet**: introduces residual (skip) connections to train very deep networks by easing gradient flow.

---

### Q65. Why are residual connections helpful?

**A:**

* They allow layers to learn a residual function (F(x) = H(x) - x) and add it to the input: (y = x + F(x)).
* Makes it easier to learn identity mappings.
* Improves gradient flow, enabling training of very deep networks (e.g., 50, 101, 152 layers).

---

### Q66. What is data augmentation and why is it important in image classification?

**A:**

* Create new training samples by applying label-preserving transformations, e.g., flips, rotations, crops, color jitter, random noise.
* Increases effective dataset size, reduces overfitting, improves robustness to variations in real-world data.

---

### Q67. What is transfer learning and when is it useful?

**A:**

* Use a model pretrained on a large dataset (e.g. ImageNet) as a starting point for a new task.
* You can fine-tune all layers or only train new classification head.
* Very helpful when your dataset is small or expensive to label.

---

### Q68. What is the difference between top-1 and top-5 accuracy (like on ImageNet)?

**A:**

* **Top-1**: the highest probability prediction must match the true label.
* **Top-5**: the true label must appear in the 5 classes with highest predicted probabilities.
* Top-5 is more forgiving and useful when many classes are similar.

---

### Q69. Why do we often use ReLU instead of sigmoid or tanh in CNNs?

**A:**

* ReLU: simple, non-saturating for positive inputs, gradient does not vanish when x > 0.
* Sigmoid/tanh saturate for large |x| → tiny gradients; training becomes difficult in deep networks.
* ReLU also encourages sparse activations (many zeros).

---

## C. Object Detection – More Conceptual

### Q70. What are anchor boxes and why do detectors use them?

**A:**

* Predefined bounding boxes (with certain scales and aspect ratios) placed densely on feature maps.
* The model predicts offsets and class scores relative to each anchor.
* They allow the network to handle objects of multiple scales and shapes efficiently.

---

### Q71. How are positive and negative samples defined for anchors?

**A:**

* Positive: anchors whose IoU with a ground-truth box is above a certain threshold (e.g., > 0.7), and sometimes the best-matching anchor per ground-truth.
* Negative: anchors whose IoU with all ground-truth boxes is below a lower threshold (e.g., < 0.3).
* Anchors in between might be ignored (neutral).

---

### Q72. What loss functions are typically used in object detection?

**A:**

* **Classification loss**: cross-entropy or focal loss (for foreground/background or multi-class).
* **Box regression loss**: L1 loss, smooth L1 (Huber), or IoU-based loss (GIoU, DIoU, CIoU).
* Total loss = weighted sum of classification + regression (sometimes + other terms, e.g., center-ness).

---

### Q73. What problem does focal loss solve?

**A:**

* In one-stage detectors, there are many more easy negatives than positives; this class imbalance dominates the loss.
* Focal loss down-weights easy examples and focuses training on hard, misclassified examples.
* Greatly improves training for dense detectors like RetinaNet.

---

### Q74. What is a Feature Pyramid Network (FPN) and why is it used?

**A:**

* FPN builds a multi-scale feature pyramid from top (high-level) to bottom (low-level) layers with lateral connections.
* Outputs feature maps at multiple resolutions but all semantically strong.
* Helps detect objects at different scales (small, medium, large) more effectively.

---

### Q75. Why does object detection often use multi-scale training/testing?

**A:**

* Real objects appear at various sizes in images.
* Multi-scale training: resize images randomly to different sizes; improves robustness to scale.
* Multi-scale testing: run detector at several scales, combine results; can improve accuracy (at higher computational cost).

---

### Q76. What are common failure modes in object detection?

**A:**

* Missing small or heavily occluded objects.
* Confusing visually similar classes (e.g., cat vs dog).
* Incorrect localization (box shifted or too small/large).
* Duplicate detections (imperfect NMS) or wrong class labels.

---

### Q77. Difference between two-stage and one-stage detectors (in terms of speed and accuracy).

**A:**

* **Two-stage** (Faster R-CNN): region proposal stage + classification/refinement stage; generally higher accuracy, slower.
* **One-stage** (YOLO, SSD, RetinaNet): directly predict boxes and classes on dense grid; faster, often slightly lower accuracy (but modern ones are very strong).

---

## D. Object Tracking – More Conceptual

### Q78. What is the difference between single-object tracking and multi-object tracking?

**A:**

* **Single-object**: track one target once it is given in the first frame.
* **Multi-object**: track multiple objects simultaneously, maintain identities (ID 1, 2, 3, …) through time; more complex data association.

---

### Q79. What is a Kalman filter used for in tracking?

**A:**

* Estimates object state (position, velocity, etc.) over time under a linear Gaussian model.
* Uses prediction + measurement update loop.
* Provides smoothed, robust estimates and uncertainty, useful for prediction and data association.

---

### Q80. What is data association in multi-object tracking?

**A:**

* The process of matching current detections to existing tracks.
* Typically formulated as an assignment problem (tracks ↔ detections) using cost based on distance, appearance similarity, IoU, etc.
* Solved by algorithms like Hungarian (Kuhn–Munkres) or greedy matching.

---

### Q81. What are ID switches in multi-object tracking?

**A:**

* When the identity label of a tracked object changes (e.g., track 1 becomes track 2, or two IDs swap).
* This is a typical tracking error and is penalized in metrics like MOTA.

---

### Q82. Why is appearance modeling (e.g., deep features) important in tracking?

**A:**

* Motion alone may be ambiguous if objects cross paths or move similarly.
* Appearance features help distinguish objects that are close together or overlapping.
* Improves data association robustness, reducing ID switches.

---

## E. Image Segmentation – More Conceptual

### Q83. What is the main difference between semantic segmentation and instance segmentation?

**A:**

* Semantic: all pixels of the same class share the same label (no separation between instances).
* Instance: distinguishes different objects of the same class (e.g., person #1 vs person #2), assigning separate masks.

---

### Q84. Why is segmentation often harder than detection?

**A:**

* Requires classification of every pixel, not just bounding boxes.
* Needs accurate boundaries at fine resolution.
* Small errors near edges or thin structures can significantly degrade metrics like IoU.

---

### Q85. What loss functions are used for segmentation when classes are imbalanced (e.g., small objects)?

**A:**

* Dice loss (or soft IoU loss)
* Focal loss
* Weighted cross-entropy (class weights inversely proportional to frequency)
  These focus more on rare or small classes.

---

### Q86. What is U-Net and why is it popular in segmentation?

**A:**

* An encoder–decoder CNN with skip connections between matching resolutions.
* Encoder: downsampling with conv+pool; Decoder: upsampling with conv+upsample.
* Skip connections pass high-resolution features to decoder → better localization and boundary detail, especially in medical imaging.

---

### Q87. What are conditional random fields (CRFs) used for in segmentation?

**A:**

* Post-processing step to refine segmentation masks.
* Encourages label smoothness and alignment with image edges (pixels with similar color/nearby positions tend to share labels).
* Can improve fine details and sharp boundaries.

---

### Q88. How is IoU used as a metric for segmentation?

**A:**

* Compare predicted mask vs ground-truth mask for each class.
* IoU = |Intersection| / |Union| of the two sets of pixels.
* Mean IoU (mIoU) = average IoU over all classes; a common benchmark.

---

### Q89. What is a “skip connection” in encoder–decoder segmentation networks and why is it helpful?

**A:**

* Link from an early (high-resolution) feature map directly to a later decoder layer of the same resolution.
* Provides fine spatial details that might be lost during downsampling.
* Helps produce sharper segmentation boundaries.

---

### Q90. Why do segmentation networks often downsample then upsample instead of operating at full resolution all the time?

**A:**

* Full-resolution convs with many channels are very expensive in memory and computation.
* Downsampling lets the network capture larger receptive fields with fewer layers and parameters.
* Upsampling reconstructs the details at the end.

---

## F. 3D Vision – More Conceptual

### Q91. What is the pinhole camera model?

**A:**

* Idealized model where 3D points project through a single point (camera center) onto an image plane.
* Described by camera intrinsics (focal length, principal point) and extrinsics (rotation and translation).
* Basis for most projective geometry in computer vision.

---

### Q92. What are camera intrinsics vs extrinsics?

**A:**

* **Intrinsics**: internal parameters like focal lengths, principal point, skew; map 3D camera coordinates to 2D pixel coordinates.
* **Extrinsics**: rotation and translation that transform points from world coordinates to camera coordinates.

---

### Q93. What is stereo vision and how does it estimate depth?

**A:**

* Two cameras (left, right) capture images from slightly different viewpoints.
* For a point visible in both images, difference in horizontal position (disparity) is inversely proportional to depth.
* Depth ∝ baseline × focal length / disparity.

---

### Q94. What is epipolar geometry?

**A:**

* Describes the geometric relationship between two camera views of the same scene.
* A point in one image corresponds to a line (epipolar line) in the other image where its match must lie.
* Encoded by fundamental or essential matrix; used to constrain correspondence search.

---

### Q95. Compare depth map, point cloud, and mesh.

**A:**

* **Depth map**: 2D image where each pixel holds depth value from one viewpoint.
* **Point cloud**: set of 3D points (x,y,z) in space, no connectivity.
* **Mesh**: 3D surface with vertices and faces; explicitly defines surface topology.

---

### Q96. What is SLAM (Simultaneous Localization and Mapping)?

**A:**

* Algorithm where a moving camera/robot builds a map of an unknown environment while simultaneously estimating its own pose in that map.
* Combines sensor data (e.g., camera, lidar) with motion models and optimization.

---

## G. Generative Models – More Conceptual

### Q97. What is the main difference between discriminative and generative models?

**A:**

* **Discriminative**: model (p(y|x)) or direct mapping from input to label (e.g., classifiers).
* **Generative**: model data distribution (p(x)) or joint (p(x,y)), can generate new samples similar to training data.

---

### Q98. Why can VAEs sometimes generate blurry images?

**A:**

* The likelihood term (e.g., Gaussian with L2 loss) encourages averaging over multiple possible outputs.
* The KL regularization also pulls latent codes towards a simple prior.
* This combination can favor smooth, blurry reconstructions.

---

### Q99. What is mode collapse in GANs?

**A:**

* Generator produces only a small variety of outputs (few “modes”) even though real data has many diverse modes.
* Different input noise vectors result in similar or identical outputs.
* Discriminator may not penalize this strongly enough.

---

### Q100. What is a conditional GAN and why is it useful?

**A:**

* GAN where both generator and discriminator get extra conditioning information (e.g., class label, text, image).
* Generator learns (G(z, c)) to produce outputs consistent with condition c.
* Useful for tasks such as image-to-image translation, class-conditional generation.

---

### Q101. How do you evaluate generative image models (GANs, VAEs)?

**A:**

* **Qualitative**: visual inspection (realism, diversity, artifacts).
* **Quantitative**:

  * Inception Score (IS)
  * Fréchet Inception Distance (FID)
  * Precision/recall in feature space
  * Likelihood-based metrics for VAEs, flows.

---

### Q102. Compare AE, VAE, GAN briefly in terms of strengths/weaknesses.

**A:**

* **AE**: simple, good for reconstruction/compression; not great for realistic sampling distribution.
* **VAE**: probabilistic, easy sampling, smooth latent space; outputs can be blurry.
* **GAN**: often produces very sharp, realistic images; but hard to train, no explicit likelihood, mode collapse.

---

### Q103. What is the latent space in generative models?

**A:**

* A lower-dimensional space (z) from which the model generates data.
* Ideally, similar z’s produce similar x’s; structure in latent space captures semantic properties (e.g., interpolation, arithmetic on attributes).

---

### Q104. Why is it useful that z in VAE is constrained to follow a simple prior (like N(0,I))?

**A:**

* Makes sampling easy: just draw z ~ N(0, I) to generate new data.
* Encourages smooth, well-behaved latent space; nearby points correspond to similar outputs.
* Prevents latent codes from spreading arbitrarily, which would make sampling difficult.

---

If you still have energy, I can:

* Build a **mock conceptual exam** (e.g., 30 questions with NO answers first), then another message with all answers (so you can test yourself).
* Or zoom in on **any one topic** (e.g., “only object detection” or “only segmentation”) and make a giant conceptual bank just for that.


Nice, let’s turn this into **calculation-heavy exam practice** with clear explanations.
Format:

> **Q… (Exam-style question)**
> **Solution:** step-by-step, so you can see *exactly* how to do similar ones.

Try to solve each question yourself first, then check the solution.

---

## 1. CNN & Image Classification – Sizes, Parameters, Receptive Field

### Q1. (Conv output size + number of parameters)

We have an RGB image:

* Input: shape **3 × 32 × 32** (C × H × W)

Network:

1. **Conv1**: 16 filters, kernel 5×5, stride 1, padding 2
2. MaxPool1: kernel 2×2, stride 2
3. **Conv2**: 32 filters, kernel 5×5, stride 1, padding 2
4. MaxPool2: kernel 2×2, stride 2

(a) Compute the spatial size and channels after each layer.
(b) Compute number of parameters (weights + bias) of Conv1 and Conv2.
(c) Suppose we flatten the final feature map and use a fully-connected layer to 10 classes. How many weights and biases in that FC layer?

---

#### Solution Q1

**Formula for conv output (for each spatial dim):**

[
W_\text{out} = \frac{W - K + 2P}{S} + 1
]

---

**Step 1 – Conv1**

* Input: C = 3, H = 32, W = 32
* K = 5, P = 2, S = 1

[
W_\text{out} = \frac{32 - 5 + 2\times 2}{1} + 1 = \frac{32 - 5 + 4}{1} + 1 = (31) + 1 = 32
]

So **H_out = 32, W_out = 32**, channels = number of filters = 16.
→ Output: **16 × 32 × 32**

**Parameters Conv1**

Each filter:

* Size = 3 (in channels) × 5 × 5 = (3 \times 25 = 75) weights
* * 1 bias = 76 parameters per filter

16 filters →

* (16 \times 76) parameters
* (16 \times 70 = 1120)
* (16 \times 6 = 96)
* Total = **1120 + 96 = 1216** parameters.

---

**Step 2 – MaxPool1 (2×2, stride 2)**

Pool:
[
W_\text{out} = \frac{32 - 2}{2} + 1 = \frac{30}{2} + 1 = 15 + 1 = 16
]

So after MaxPool1: **16 × 16 × 16** (C × H × W).

---

**Step 3 – Conv2**

Input to Conv2: C = 16, H = W = 16

* K = 5, P = 2, S = 1

[
W_\text{out} = \frac{16 - 5 + 2\times2}{1} + 1 = 16
]

So spatial size stays **16 × 16**, number of output channels = 32.
→ Output: **32 × 16 × 16**

**Parameters Conv2**

Each filter:

* 16 input channels, kernel 5×5
* Weights per filter = (16 \times 5 \times 5 = 16 \times 25 = 400)
* * 1 bias = 401

32 filters:

* (32 \times 401 = 32 \times 400 + 32 \times 1 = 12800 + 32 = 12832) parameters.

**Total conv parameters** = 1216 + 12832 = **14048**.

---

**Step 4 – MaxPool2**

Pool 2×2, stride 2 on 16×16:

[
W_\text{out} = \frac{16 - 2}{2} + 1 = \frac{14}{2} + 1 = 7 + 1 = 8
]

So final feature map: **32 × 8 × 8**.

---

**Step 5 – FC layer to 10 classes**

Flattened size:

* (32 \times 8 \times 8)
* (8 \times 8 = 64)
* (32 \times 64 = 2048)

So FC weight matrix: size **10 × 2048** →

* Weights: (10 \times 2048 = 20480)
* Biases: 10

Total FC params = **20480 + 10 = 20490**.

---

**Exam memory:**

* You MUST know the conv output formula.
* Parameter count = (kernel_h × kernel_w × in_channels + 1) × out_channels.

---

### Q2. (Receptive field calculation)

Consider this network (all convolutions are “same” style with padding so sizes stay):

1. Conv1: kernel 3×3, stride 1
2. Conv2: kernel 3×3, stride 1
3. MaxPool: kernel 2×2, stride 2
4. Conv3: kernel 3×3, stride 1

Compute the **receptive field size** (in the input image) for 1 neuron in Conv3 output. Assume square receptive fields (so answer like 10×10).

---

#### Solution Q2

Use the standard iterative method:

Track two quantities as we go from input → output:

* **RF** (receptive field size)
* **J** (“jump” – how many input pixels you move when you move by 1 step in this layer)

Initialize at the **input**:

* RF₀ = 1 (a single pixel sees itself)
* J₀ = 1

---

**Layer 1: Conv1 (K=3, S=1)**

Formula:

* RF₁ = RF₀ + (K - 1) × J₀
* J₁ = J₀ × S

Compute:

* RF₁ = 1 + (3 - 1) × 1 = 1 + 2 = 3
* J₁ = 1 × 1 = 1

---

**Layer 2: Conv2 (K=3, S=1)**

* RF₂ = RF₁ + (3 - 1) × J₁ = 3 + 2 × 1 = 5
* J₂ = 1 × 1 = 1

---

**Layer 3: MaxPool (K=2, S=2)**

* RF₃ = RF₂ + (2 - 1) × J₂ = 5 + 1 × 1 = 6
* J₃ = J₂ × 2 = 1 × 2 = 2

---

**Layer 4: Conv3 (K=3, S=1)**

* RF₄ = RF₃ + (3 - 1) × J₃ = 6 + 2 × 2 = 6 + 4 = 10
* J₄ = 2 × 1 = 2

So each neuron in Conv3 sees a **10×10 patch** in the original input.

---

**Exam tip:**
This “RF + (K-1)*J” and “J *= stride” pattern is standard. The teacher can give you any small stack and ask for receptive field.

---

## 2. Object Detection – IoU & mAP Calculations

### Q3. (IoU of two boxes)

Two bounding boxes (x1, y1, x2, y2):

* Box A = (10, 20, 50, 60)
* Box B = (20, 40, 60, 80)

Compute IoU.

---

#### Solution Q3

1. **Intersection coordinates**

* Left = max(10, 20) = 20
* Top  = max(20, 40) = 40
* Right = min(50, 60) = 50
* Bottom = min(60, 80) = 60

Intersection width = 50 - 20 = 30
Intersection height = 60 - 40 = 20

Intersection area = 30 × 20 = **600**

---

2. **Individual areas**

* Area(A) = (50 - 10) × (60 - 20) = 40 × 40 = **1600**
* Area(B) = (60 - 20) × (80 - 40) = 40 × 40 = **1600**

---

3. **Union area**

[
\text{Union} = \text{Area}(A) + \text{Area}(B) - \text{Area}(\text{Intersection}) = 1600 + 1600 - 600 = 2600
]

---

4. **IoU**

[
\text{IoU} = \frac{600}{2600}
]

Simplify: divide numerator and denominator by 100:

[
\frac{600}{2600} = \frac{6}{26} = \frac{3}{13} \approx 0.2308
]

So **IoU ≈ 0.23 (23%)**.

---

### Q4. (Simple AP / mAP calculation for one class)

You have one class and 3 ground-truth objects in the dataset.
Your model makes 5 detections for this class (already after NMS), sorted by confidence:

| Rank | Detection | Correct? (TP/FP) |
| ---- | --------- | ---------------- |
| 1    | D1        | TP               |
| 2    | D2        | FP               |
| 3    | D3        | TP               |
| 4    | D4        | FP               |
| 5    | D5        | TP               |

Assume **each TP matches a different ground-truth** (no double-use).

1. Compute precision and recall at each rank.
2. Compute AP using the “interpolated precision” method (VOC-style, with integration over recall).

---

#### Solution Q4

There are **3 ground-truth objects**, so max recall = 3/3 = 1.

We go through predictions one by one.

Let TP(k) = number of TPs in top k
Let FP(k) = number of FPs in top k

---

**Rank 1 (D1 = TP)**

* TP(1) = 1
* FP(1) = 0

Precision P₁ = TP(1) / 1 = 1 / 1 = **1.0**
Recall R₁ = TP(1) / 3 = 1 / 3 ≈ **0.3333**

---

**Rank 2 (D2 = FP)**

* TP(2) = 1
* FP(2) = 1

P₂ = 1 / 2 = **0.5**
R₂ = 1 / 3 ≈ **0.3333**

---

**Rank 3 (D3 = TP)**

* TP(3) = 2
* FP(3) = 1

P₃ = 2 / 3 ≈ **0.6667**
R₃ = 2 / 3 ≈ **0.6667**

---

**Rank 4 (D4 = FP)**

* TP(4) = 2
* FP(4) = 2

P₄ = 2 / 4 = **0.5**
R₄ = 2 / 3 ≈ **0.6667**

---

**Rank 5 (D5 = TP)**

* TP(5) = 3
* FP(5) = 2

P₅ = 3 / 5 = **0.6**
R₅ = 3 / 3 = **1.0**

---

So we have pairs (Recall, Precision):

* (R₁, P₁) = (1/3, 1.0)
* (R₂, P₂) = (1/3, 0.5)
* (R₃, P₃) = (2/3, 2/3)
* (R₄, P₄) = (2/3, 0.5)
* (R₅, P₅) = (1, 0.6)

Unique recall levels: **1/3, 2/3, 1**.

---

**Interpolated precision**

Define (P_{\text{interp}}(r) = \max_{r' \ge r} P(r')).

* For r = 1: consider all recall ≥ 1 → that’s only R₅:

  * (P_{\text{interp}}(1) = \max(0.6) = 0.6)

* For r = 2/3: consider recall ≥ 2/3 → R₃, R₄, R₅ with precisions (0.6667, 0.5, 0.6)
  → max = **0.6667** (≈ 2/3).

* For r = 1/3: consider recall ≥ 1/3 → R₁…R₅ with precisions (1.0, 0.5, 0.6667, 0.5, 0.6)
  → max = **1.0**.

So:

* (P_{\text{interp}}(1/3) = 1)
* (P_{\text{interp}}(2/3) = 2/3)
* (P_{\text{interp}}(1) = 0.6)

---

**Integrate over recall (discrete sum)**

We do:

[
AP = \sum_{i} (R_i - R_{i-1}) P_{\text{interp}}(R_i)
]

Let R₀ = 0, R₁ = 1/3, R₂ = 2/3, R₃ = 1.

So:

* Contribution 1: ((R_1 - R_0) P_{\text{interp}}(R_1) = (1/3 - 0) \times 1 = 1/3)
* Contribution 2: ((R_2 - R_1) P_{\text{interp}}(R_2) = (2/3 - 1/3) \times (2/3) = (1/3) \times (2/3) = 2/9)
* Contribution 3: ((R_3 - R_2) P_{\text{interp}}(R_3) = (1 - 2/3) \times 0.6 = (1/3) \times 0.6)

Now compute the last term exactly: 0.6 = 3/5

[
(1/3) \times (3/5) = 3 / 15 = 1/5
]

So:

[
AP = 1/3 + 2/9 + 1/5
]

Bring to common denominator 45:

* (1/3 = 15/45)
* (2/9 = 10/45)
* (1/5 = 9/45)

Sum = (15 + 10 + 9 = 34) → **34/45 ≈ 0.756**

So **AP ≈ 0.756** for this class.

If there was only this class, mAP = AP.

---

## 3. Segmentation / Clustering – K-means Example

### Q5. (One k-means iteration)

We want to cluster 4 pixels using only 2D “color” features (R,G).

Points:

* P₁ = (1, 1)
* P₂ = (2, 1)
* P₃ = (4, 3)
* P₄ = (5, 4)

We run k-means with **k = 2**.

Initial centroids:

* m₁⁰ = (1, 1)
* m₂⁰ = (5, 4)

1. Assign each point to the nearest centroid (using squared Euclidean distance).
2. Compute new centroids m₁¹ and m₂¹.

---

#### Solution Q5

We compute squared distances (d^2) – no need for square roots, because sqrt is monotonic.

---

**Distances to initial centroids**

1. **P₁ = (1,1)**

* To m₁⁰ = (1,1):
  ((1-1)^2 + (1-1)^2 = 0 + 0 = 0)

* To m₂⁰ = (5,4):
  ((1-5)^2 + (1-4)^2 = (-4)^2 + (-3)^2 = 16 + 9 = 25)

→ Closest to m₁⁰.

---

2. **P₂ = (2,1)**

* To m₁⁰ = (1,1):
  ((2-1)^2 + (1-1)^2 = 1^2 + 0 = 1)

* To m₂⁰ = (5,4):
  ((2-5)^2 + (1-4)^2 = (-3)^2 + (-3)^2 = 9 + 9 = 18)

→ Closest to m₁⁰.

---

3. **P₃ = (4,3)**

* To m₁⁰ = (1,1):
  ((4-1)^2 + (3-1)^2 = 3^2 + 2^2 = 9 + 4 = 13)

* To m₂⁰ = (5,4):
  ((4-5)^2 + (3-4)^2 = (-1)^2 + (-1)^2 = 1 + 1 = 2)

→ Closest to m₂⁰.

---

4. **P₄ = (5,4)**

* To m₁⁰ = (1,1):
  ((5-1)^2 + (4-1)^2 = 4^2 + 3^2 = 16 + 9 = 25)

* To m₂⁰ = (5,4):
  ((5-5)^2 + (4-4)^2 = 0 + 0 = 0)

→ Closest to m₂⁰.

---

**Cluster assignments**

* Cluster 1 (C₁): P₁, P₂
* Cluster 2 (C₂): P₃, P₄

---

**Compute new centroids as mean of points in cluster**

* (m_1^1 =) mean of P₁ and P₂:

[
x: \frac{1 + 2}{2} = \frac{3}{2} = 1.5, \quad
y: \frac{1 + 1}{2} = \frac{2}{2} = 1
]

So (m_1^1 = (1.5, 1)).

* (m_2^1 =) mean of P₃ and P₄:

[
x: \frac{4 + 5}{2} = \frac{9}{2} = 4.5, \quad
y: \frac{3 + 4}{2} = \frac{7}{2} = 3.5
]

So (m_2^1 = (4.5, 3.5)).

That’s **one full k-means iteration** (assign + update).

---

## 4. 3D Vision – Voxel Memory Calculation

### Q6. (Voxel grid memory)

You store a 3D voxel grid using 1 byte per voxel.

(a) For a grid of size (128 \times 128 \times 128), how many bytes and approximately how many megabytes?
(b) If you double the resolution in each dimension to (256 \times 256 \times 256), how many bytes and megabytes?
(c) What is the ratio of memory usage between the two grids?

---

#### Solution Q6

(a) **128³ voxels**

First compute 128³:

* 128² = 128 × 128 = 16384
* 128³ = 16384 × 128

Multiply 16384 × 128:

* 16384 × 100 = 1,638,400
* 16384 × 20 = 327,680
* 16384 × 8 = 131,072

Sum:
1,638,400 + 327,680 = 1,966,080
1,966,080 + 131,072 = **2,097,152**

So there are **2,097,152 voxels**.
Each voxel = 1 byte → **2,097,152 bytes**.

1 MiB (mebibyte) = 1,048,576 bytes.

[
\frac{2,097,152}{1,048,576} = 2
]

So memory = **2 MiB**.

---

(b) **256³ voxels**

128 → 256 is doubling each dimension. Volume scales by (2^3 = 8).

So:

* 256³ = 8 × 128³ = 8 × 2,097,152 = **16,777,216 bytes**.

Check multiplication:

* 2,097,152 × 8 =

  * 2,000,000 × 8 = 16,000,000
  * 97,152 × 8 = 777,216
  * Sum = 16,777,216 correct.

Convert to MiB:

[
\frac{16,777,216}{1,048,576} = 16
]

So memory = **16 MiB**.

---

(c) **Ratio**

[
\text{Ratio} = \frac{16 \text{ MiB}}{2 \text{ MiB}} = 8:1
]

So doubling resolution in each dimension multiplies memory by **8**.

---

## 5. Tracking / Background Modeling – Numeric Example

### Q7. (Adaptive background subtraction update)

We use exponential moving average for the background at each pixel:

[
B_t = (1 - \alpha) B_{t-1} + \alpha I_t
]

Given a single pixel:

* Initial background (B_0 = 100) (intensity)
* (\alpha = 0.1)

Observed intensities:

* At t = 1: (I_1 = 130)
* At t = 2: (I_2 = 160)

Compute (B_1) and (B_2).

---

#### Solution Q7

**At t = 1:**

[
B_1 = (1 - 0.1) B_0 + 0.1 I_1 = 0.9 \times 100 + 0.1 \times 130
]

Compute:

* 0.9 × 100 = 90
* 0.1 × 130 = 13

So:

[
B_1 = 90 + 13 = 103
]

---

**At t = 2:**

[
B_2 = 0.9 B_1 + 0.1 I_2 = 0.9 \times 103 + 0.1 \times 160
]

Compute:

* 0.9 × 103 = 92.7
* 0.1 × 160 = 16

So:

[
B_2 = 92.7 + 16 = 108.7
]

So background slowly moves from 100 → 103 → 108.7 as the pixel becomes brighter.

---

If you like this style, I can:

* Add **more conv / IoU / mAP** numeric questions.
* Or make a **mini mock exam** (e.g. 10 questions: 5 conceptual, 5 calculation) with no solutions first, then a second version with detailed solutions so you can time yourself.
