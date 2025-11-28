# Unit 8: Image Classification with CNN (Exam-Ready Guide)

How to use this sheet: skim bolded phrases for a 2-minute refresh, then drill sections with worked examples and exam-style questions. The formatting (including LaTeX equations) is designed to print or export cleanly to PDF.

## Table of Contents
1. [Exam Checklist](#1-exam-checklist)
2. [Why CNNs Beat MLPs](#2-why-cnns-beat-mlps)
3. [Core Math and Shapes](#3-core-math-and-shapes)
4. [Building Blocks](#4-building-blocks)
5. [Training, Regularization, and Schedules](#5-training-regularization-and-schedules)
6. [Architectures to Recognize](#6-architectures-to-recognize)
7. [Transfer Learning Playbook](#7-transfer-learning-playbook)
8. [From Lecture to Lab](#8-from-lecture-to-lab)
9. [Failure Modes + Fixes](#9-failure-modes--fixes)
10. [Key Formulas](#10-key-formulas)
11. [Rapid Q&A](#11-rapid-qa)
12. [Exam-Style Questions](#12-exam-style-questions)

---

## 1. Exam Checklist

Be able to:

- Explain **sparse + shared weights** → translation tolerance and fewer parameters.
- Quickly compute **output shapes** and **receptive fields** for a stack of conv / pool layers.
- Choose and justify **activation functions** (ReLU, Leaky ReLU, GELU, etc.).
- Map each **regularizer / augmentation** to its effect (dropout, weight decay, label smoothing, data augmentation).
- Recognize **major architectures + signature ideas** (LeNet, AlexNet, VGG, Inception, ResNet, DenseNet, EfficientNet).
- Outline **transfer learning steps + LR ratios** (head vs backbone, freezing, unfreezing).
- Diagnose typical **training problems** (overfitting, vanishing gradients, dying ReLUs) and propose fixes.

---

## 2. Why CNNs Beat MLPs

**Key intuition:** images have strong local structure (nearby pixels are related). CNNs exploit this; vanilla MLPs do not.

- **Spatial structure kept**
  - MLP: flatten $H \times W \times C$ into a long vector → destroys local neighborhood structure.
  - CNN: uses small kernels (e.g., $3 \times 3$) sliding over the image → preserves locality.

- **Sparse connections**
  - Each output neuron connects only to a local patch of the input (receptive field), not to every pixel.
  - Reduces parameters and overfitting risk.

- **Parameter sharing**
  - The same kernel weights are used at every spatial location.
  - Greatly reduces the number of learnable parameters compared to FC layers.

- **Translation tolerance**
  - If an object shifts slightly, the same kernel still responds because it slides across positions.
  - Pooling and stride further help small translation invariance.

- **Hierarchical features**
  - Early layers: detect edges, corners, simple textures.
  - Middle layers: detect parts (eyes, wheels, textures).
  - Deep layers: detect object concepts (faces, cars, animals).

- **Historical context (good short answer)**
  - Before deep CNNs: hand-crafted features (SIFT, HOG) + SVM / shallow models.
  - AlexNet (2012): first deep CNN to win ImageNet by a large margin.
  - ResNet (2015): residual connections allow very deep networks.
  - EfficientNet (2019): systematic scaling of depth, width, and resolution.

**Example exam question (conceptual)**  
Explain why applying a large fully connected layer directly to a $224 \times 224$ RGB image is usually a bad idea, and how convolution + pooling addresses this.

---

## 3. Core Math and Shapes

### 3.1 Convolution output shape

For one spatial dimension (width or height):

```math
W_\text{out} = \left\lfloor \frac{W_\text{in} - K + 2P}{S} \right\rfloor + 1
```

Where:
- $W_\text{in}$ = input size (width or height).
- $K$ = kernel size.
- $P$ = padding.
- $S$ = stride.

**Worked example (shape):**  
Input: $32 \times 32$, kernel $5 \times 5$, padding $0$, stride $1$.
- $W_\text{out} = (32 - 5 + 0)/1 + 1 = 28$
- Output: $28 \times 28$ feature map.

**Worked example (keeping size with padding):**  
Input: $32 \times 32$, kernel $3 \times 3$, stride $1$, padding $1$.
- $W_\text{out} = (32 - 3 + 2\cdot 1)/1 + 1 = 32$
- Output size preserved ($32 \times 32$).

### 3.2 Parameter counts

- **Conv layer parameters:**

```math
\text{params} = K_h \cdot K_w \cdot C_\text{in} \cdot C_\text{out} + C_\text{out}
```

where $+ C_\text{out}$ is the bias term (one bias per output channel).

- **FC layer parameters:**

```math
\text{params} = (H \cdot W \cdot C_\text{in}) \cdot C_\text{out} + C_\text{out}
```

**Worked example (parameter comparison):**  
Input: $32 \times 32 \times 3$, want $64$ output channels.
- Conv with $3 \times 3$ kernel:
  - $\text{params} = 3 \cdot 3 \cdot 3 \cdot 64 + 64 = 1728 + 64 = 1792$
- FC from $32 \times 32 \times 3$ to $64$:
  - $H \cdot W \cdot C_\text{in} = 32 \cdot 32 \cdot 3 = 3072$
  - $\text{params} = 3072 \cdot 64 + 64 = 196608 + 64 \approx 196672$
- Conv uses **two orders of magnitude fewer parameters**.

### 3.3 Receptive field

Receptive field (RF) is the region in the input that affects one output neuron. Understanding RF is crucial for designing architectures that can capture objects of different sizes.

**Definition:** The receptive field of a neuron is the size of the region in the input image that can affect its value.

For a sequence of conv layers:

```math
RF_\text{new} = RF_\text{old} + (K - 1)\cdot \prod(\text{previous\_strides})
```

**Derivation intuition:** Each new layer adds $(K-1)$ pixels to the RF, but this addition is scaled by the product of all previous strides (since stride effectively "zooms out" the view).

**Worked example (RF growth):**  
Three $3 \times 3$ conv layers, all stride 1.
- Start: $RF_0 = 1$ (a single pixel).
- Layer 1: $RF_1 = 1 + (3 - 1)\cdot 1 = 3$ (sees $3 \times 3$ patch)
- Layer 2: $RF_2 = 3 + (3 - 1)\cdot 1 = 5$ (sees $5 \times 5$ patch)
- Layer 3: $RF_3 = 5 + (3 - 1)\cdot 1 = 7$ (sees $7 \times 7$ patch)
- After 3 layers, each neuron "sees" a $7 \times 7$ region of the input.

**Worked example (with stride):**  
Two $3 \times 3$ conv layers, stride 1, followed by one $3 \times 3$ conv with stride 2.
- Start: $RF_0 = 1$
- Layer 1 (stride 1): $RF_1 = 1 + (3-1)\cdot 1 = 3$
- Layer 2 (stride 1): $RF_2 = 3 + (3-1)\cdot 1 = 5$
- Layer 3 (stride 2): $RF_3 = 5 + (3-1)\cdot (1 \cdot 1 \cdot 2) = 5 + 4 = 9$
- The stride-2 layer multiplies the effective RF growth.

**Pooling and RF:** Max-pooling with kernel size $K_p$ and stride $S_p$ effectively increases RF by approximately $K_p$ and scales by $S_p$. For a $2 \times 2$ max-pool with stride 2: RF increases by 2 and is scaled by 2.

**Practical importance:** 
- Small RF → captures local features (edges, textures)
- Large RF → captures global context (object parts, full objects)
- Modern architectures use multiple RF sizes (e.g., Inception modules, dilated convolutions) to handle objects at different scales.

### 3.4 Pooling and downsampling

- $2 \times 2$ max-pooling with stride $2$ halves spatial dimensions:
  - $32 \times 32 \rightarrow 16 \times 16 \rightarrow 8 \times 8 \rightarrow \dots$
- Global Average Pooling (GAP):
  - For each channel, take the average over all $H \times W$ positions → a single scalar.
  - Replaces large FC layers at the end with a much smaller linear layer.

### 3.5 Batch Normalization (BN)

Batch Normalization normalizes activations across a mini-batch, making training more stable and allowing higher learning rates.

**Mathematical formulation:**

For each channel $c$ in a feature map:

```math
\mu_c = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{i=1}^{H} \sum_{j=1}^{W} x_{b,i,j,c}
```

```math
\sigma_c^2 = \frac{1}{B \cdot H \cdot W} \sum_{b=1}^{B} \sum_{i=1}^{H} \sum_{j=1}^{W} (x_{b,i,j,c} - \mu_c)^2
```

```math
\hat{x}_{b,i,j,c} = \frac{x_{b,i,j,c} - \mu_c}{\sqrt{\sigma_c^2 + \varepsilon}}
```

```math
y_{b,i,j,c} = \gamma_c \cdot \hat{x}_{b,i,j,c} + \beta_c
```

Where:
- $B$ = batch size, $H \times W$ = spatial dimensions
- $\mu_c, \sigma_c^2$ = mean and variance computed over batch and spatial dimensions
- $\gamma_c, \beta_c$ = learnable scale and shift parameters (one per channel)
- $\varepsilon$ = small constant (typically $10^{-5}$) to prevent division by zero

**Key effects:**
1. **Reduces internal covariate shift:** Normalizes the distribution of inputs to each layer, making training more stable.
2. **Allows higher learning rates:** Smoother loss landscape enables faster convergence.
3. **Acts as mild regularization:** The batch statistics add noise during training (different batches have different statistics).
4. **Enables deeper networks:** Helps with gradient flow in very deep architectures.

**Training vs inference:**
- **Training:** Uses batch statistics (mean/variance computed from current batch).
- **Inference:** Uses running averages of batch statistics (exponential moving average) to ensure consistent behavior regardless of batch size.

**Placement:** Typically placed **after** convolution but **before** activation (Conv → BN → ReLU), though some architectures place it after activation.

**Variants:**
- **LayerNorm:** Normalizes across channels for each sample (useful for RNNs/transformers).
- **GroupNorm:** Normalizes across groups of channels (useful when batch size is very small).
- **InstanceNorm:** Normalizes across spatial dimensions for each channel (useful for style transfer).

**Example exam question (calculation)**  
Given a $64 \times 64 \times 32$ input feature map, a $3 \times 3$ conv with stride $2$ and padding $1$ outputs what spatial size?  
Answer: $(64 - 3 + 2\cdot 1)/2 + 1 = 63/2 + 1 = 31 + 1 = 32$, so $32 \times 32 \times C_\text{out}$.

---

## 4. Building Blocks

### 4.1 Activations

- **Sigmoid / tanh**
  - Smooth, but saturate → gradients vanish for very large or small inputs.
  - Mostly used in output layers (e.g., binary classification) rather than in deep CNN hidden layers.

- **ReLU**
  - $\mathrm{ReLU}(x) = \max(0, x)$.
  - Advantages: simple, cheap, avoids saturation on positive side, often faster convergence.
  - Disadvantage: neurons can “die” (always output 0) if weights push them permanently negative.

- **Leaky / Parametric ReLU**
  - $\text{LeakyReLU}(x) = x$ if $x > 0$, otherwise $\alpha x$ with small $\alpha$ (e.g., $0.01$).
  - Allows a small gradient for negative inputs → fewer dead neurons.
  - **PReLU:** Learnable $\alpha$ parameter (one per channel or shared)

- **GELU (Gaussian Error Linear Unit)**
  - $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi(x)$ is CDF of standard normal
  - Approximation: $\text{GELU}(x) \approx 0.5x(1 + \tanh(\sqrt{2/\pi}(x + 0.044715x^3)))$
  - **Properties:** Smooth, non-monotonic (has negative region), used in transformers (BERT, GPT)
  - **Intuition:** Probabilistic version of ReLU - multiplies input by probability it would be kept

- **ELU (Exponential Linear Unit)**
  - $\text{ELU}(x) = x$ if $x > 0$, else $\alpha(e^x - 1)$ with $\alpha \approx 1$
  - Smooth, negative values help push mean activations toward zero

- **Swish / SiLU**
  - $\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}$
  - Self-gated activation, smooth, non-monotonic
  - Used in EfficientNet and some modern architectures

- **Softmax**
  - Turns logits into probabilities for multi-class outputs:
  - $\mathrm{softmax}(x_i) = \dfrac{e^{x_i}}{\sum_j e^{x_j}}$.
  - **Numerical stability:** Subtract max before exp: $\mathrm{softmax}(x_i) = \dfrac{e^{x_i - \max(x)}}{\sum_j e^{x_j - \max(x)}}$

### 4.2 Convolution layers

- **Kernel sizes**
  - $3 \times 3$ is the standard building block.
  - $1 \times 1$ conv mixes channels without changing spatial resolution (useful for bottlenecks).
  - Large $5 \times 5$ or $7 \times 7$ kernels can often be approximated by stacked $3 \times 3$ convs (more nonlinearity, fewer parameters).

- **Padding**
  - `"same"`: choose padding to keep spatial size unchanged (when $S = 1$, $P \approx K/2$).
  - `"valid"`: no padding; spatial size shrinks after each conv.

- **Stride**
  - Stride $> 1$ downsamples feature maps directly (alternative to pooling).

### 4.3 Pooling / Striding

- **Max-pooling**
  - Retains the strongest activation in a region → some translation invariance.

- **Average pooling**
  - Averages activations; smoother but may blur sharp responses.

- **GAP**
  - Global average pooling over full $H \times W$; reduces each channel to one value.

### 4.4 Weight Initialization

Proper initialization is crucial for training deep networks.

- **Xavier / Glorot initialization:**
  - For linear layer: $W \sim \mathcal{N}(0, \frac{2}{n_\text{in} + n_\text{out}})$ or uniform $U(-\sqrt{\frac{6}{n_\text{in} + n_\text{out}}}, \sqrt{\frac{6}{n_\text{in} + n_\text{out}}})$
  - Designed for tanh/sigmoid activations
  - Keeps variance of activations and gradients roughly constant across layers

- **He / Kaiming initialization:**
  - For ReLU: $W \sim \mathcal{N}(0, \frac{2}{n_\text{in}})$ or uniform $U(-\sqrt{\frac{6}{n_\text{in}}}, \sqrt{\frac{6}{n_\text{in}}})$
  - Accounts for ReLU killing half the activations (variance is halved)
  - Standard for modern CNNs with ReLU

- **Why it matters:**
  - Too small: gradients vanish, slow learning
  - Too large: gradients explode, training unstable
  - Proper initialization enables training very deep networks

### 4.5 Normalization and regularization

- **BatchNorm, LayerNorm, GroupNorm**
  - BN is most common in CNNs; LN and GN help when batch size is very small.

- **Dropout**
  - Randomly zeroes activations with probability $p$ (e.g., $0.2$–$0.5$) during training.
  - Reduces co-adaptation of neurons and overfitting.
  - **Spatial dropout:** Drops entire feature maps (for conv layers)
  - **Dropout placement:** Usually after activation, before or after pooling

- **Weight decay (L2 regularization)**
  - Penalizes large weights; typical values: $10^{-4}$ to $5 \cdot 10^{-4}$ for CNNs.
  - **L1 regularization:** $L_1 = \lambda \sum |w|$ (sparsity-inducing, less common in CNNs)

- **Label smoothing**
  - Replaces hard one-hot labels with softened ones (e.g., $0.9$ for correct class, $0.1/(C-1)$ for others).
  - Prevents over-confident predictions and improves calibration.
  - **Formula:** $y_\text{smooth} = (1 - \alpha) y_\text{one-hot} + \frac{\alpha}{C}$ where $\alpha$ is smoothing factor (typically $0.1$)

**Example exam question (short)**  
Why might you replace a $7 \times 7$ conv with three stacked $3 \times 3$ convs? Discuss in terms of parameters and expressiveness.

---

## 5. Training, Regularization, and Schedules

### 5.1 Losses

- **Multi-class cross-entropy (CE)**

```math
L = -\sum_i y_i \log p_i
```

where $y$ is one-hot (or smoothed) and $p$ comes from softmax.

- **Binary cross-entropy (BCE)**

```math
L = -(y \log p + (1 - y)\log(1 - p))
```

used for binary classification or per-class independent outputs.

### 5.2 Optimizers

**SGD (Stochastic Gradient Descent):**
- Basic update: $\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)$
- Simple but can be slow and noisy.

**SGD + Momentum:**
- Accumulates velocity in the gradient direction:
  ```math
  v_t = \mu v_{t-1} + \eta \nabla_\theta L(\theta_t)
  ```
  ```math
  \theta_{t+1} = \theta_t - v_t
  ```
- Momentum coefficient $\mu$ (typically $0.9$ or $0.99$) controls how much past gradients influence current update.
- **Intuition:** Like a ball rolling downhill with friction; helps escape local minima and speeds up convergence in consistent directions.
- **Nesterov momentum:** Uses gradient at $\theta_t + \mu v_{t-1}$ instead of $\theta_t$ for better convergence.

**Adam (Adaptive Moment Estimation):**
- Combines momentum with adaptive learning rates per parameter.
- Maintains exponential moving averages of gradients ($m$) and squared gradients ($v$):

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

- Bias correction (important early in training):

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

- Update:

```math
\theta_{t+1} = \theta_t - \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}
```

- Typical hyperparameters: $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$
- **Advantages:** Adaptive per-parameter learning rates, works well with sparse gradients, requires little tuning.
- **Disadvantages:** Can converge to suboptimal solutions, memory overhead (stores $m$ and $v$ for each parameter).

**AdamW:**
- Decouples weight decay from gradient-based updates:
  ```math
  \theta_{t+1} = \theta_t - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon} + \lambda \theta_t \right)
  ```
- In standard Adam, weight decay is applied to gradients, which interacts with adaptive learning rates. AdamW applies it directly to parameters, making regularization cleaner and more effective.

**Comparison:**
- **SGD + momentum:** Best for convex problems, can generalize better, requires careful LR tuning.
- **Adam/AdamW:** Better for non-convex problems, faster convergence, less sensitive to LR, but may not generalize as well as SGD in some cases.

### 5.3 Learning rate schedules

- Step decay (e.g., divide LR by $10$ at epochs $30 / 60 / 90$).
- Cosine decay (smoothly reduce LR from initial to near-zero).
- Linear warmup + cosine or inverse-sqrt (commonly used for deep or transformer-like models).
- Fine-tuning: often combine a short warmup phase + slower decay to avoid damaging pretrained weights.

### 5.4 Data Augmentation

Data augmentation artificially increases dataset size by applying transformations that preserve semantic meaning.

**Geometric transformations:**
- **Random crops:** Extract random patches (e.g., $224 \times 224$ from $256 \times 256$), forces model to be location-invariant.
- **Horizontal flips:** Mirror images left-right, doubles dataset size, common for natural images.
- **Rotations:** Small rotations ($\pm 15°$), useful for objects that can appear at any angle.
- **Scaling:** Random resize and crop, helps with scale invariance.

**Color/Photometric transformations:**
- **Color jitter:** Random adjustments to brightness, contrast, saturation, hue.
- **Normalization:** Subtract mean and divide by std (e.g., ImageNet: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).
- **Grayscale:** Randomly convert to grayscale with small probability.

**Advanced augmentations:**
- **CutMix:** Replace random rectangular region with patch from another image, label is weighted average.
- **MixUp:** Linearly interpolate between two images and their labels: $\tilde{x} = \lambda x_i + (1-\lambda)x_j$, $\tilde{y} = \lambda y_i + (1-\lambda)y_j$.
- **AutoAugment:** Learn optimal augmentation policies via reinforcement learning.
- **RandAugment:** Simplified version with random selection from augmentation pool.

**Domain-specific:**
- **Medical imaging:** Intensity normalization, elastic deformations.
- **Text recognition:** Perspective transforms, noise addition.
- **Satellite imagery:** No flips (orientation matters), but rotations and crops are useful.

### 5.5 Handling overfitting

**Symptoms:** High training accuracy but low validation accuracy, large gap between train and val loss.

**Strategies:**
1. **Increase / strengthen data augmentation** (crops, flips, jitter, CutMix, MixUp).
2. **Use dropout** on dense layers (typically $p=0.5$) and possibly on later conv layers (lower $p$, e.g., $0.2$).
3. **Apply weight decay** (L2 regularization): typical values $10^{-4}$ to $5 \cdot 10^{-4}$ for CNNs.
4. **Reduce model capacity:** Fewer layers, fewer channels, or use smaller classifier heads.
5. **Use early stopping:** Monitor validation loss, stop when it stops improving.
6. **Freeze more layers** in transfer learning to reduce trainable parameters.
7. **Label smoothing:** Prevents overconfident predictions, improves calibration.
8. **Ensemble methods:** Train multiple models and average predictions (reduces variance).

**Example exam question (training)**  
Your CNN achieves $99\%$ train accuracy but only $75\%$ validation accuracy. List three concrete changes you could make (with reasoning) to reduce overfitting.

---

## 6. Architectures to Recognize

You should be able to give 2–3 sentences describing each.

- **LeNet-5 (1998)**
  - Early CNN for digit recognition.  
  - Conv → pool → conv → pool → FC → FC; designed for $32 \times 32$ grayscale inputs (e.g., early digit datasets).

- **AlexNet (2012)**
  - 5 conv + 3 FC layers.
  - Uses ReLU, dropout, heavy data augmentation, and GPU training.
  - Won ImageNet 2012 by a large margin, popularizing deep CNNs.

- **VGG-16 / VGG-19 (2014)**
  - Deep stacks of $3 \times 3$ conv layers with occasional max-pooling.
  - Illustrates that depth and small kernels can yield strong performance.
  - Very large FC head at the end (often replaced with GAP in modern variants).

- **GoogLeNet / Inception (2014)**
  - Inception modules with parallel $1 \times 1$, $3 \times 3$, $5 \times 5$ branches and pooling.
  - $1 \times 1$ bottleneck convs reduce channel dimensions before expensive convs.
  - Includes auxiliary classifiers for regularization and additional supervision.

- **ResNet (2015)**
  - Residual blocks with identity shortcuts: $y = F(x) + x$.
  - Solves degradation / vanishing gradient issues for very deep networks.
  - **Key insight:** If optimal mapping is $H(x)$, let the network learn residual $F(x) = H(x) - x$ instead. This makes it easier to learn identity mapping (just set $F(x) = 0$).
  - **Basic block:** Two $3 \times 3$ convs with BN and ReLU, plus skip connection.
  - **Bottleneck block:** $1 \times 1$ conv (reduce) → $3 \times 3$ conv → $1 \times 1$ conv (expand), more efficient for deeper networks.
  - **Gradient flow:** Skip connections provide direct gradient paths, preventing vanishing gradients.
  - Versions: ResNet-18/34 (basic blocks), 50/101/152 (bottleneck blocks).
  - **Pre-activation variant (ResNet-v2):** BN and ReLU before conv, improves gradient flow further.

- **DenseNet (2017)**
  - Dense connectivity: each layer receives all previous feature maps via concatenation.
  - Encourages feature reuse and reduces number of parameters for similar accuracy.

- **EfficientNet (2019)**
  - Uses a compound scaling rule to jointly scale depth ($d$), width ($w$), and resolution ($r$).
  - **Scaling equations:**
    ```math
    \text{depth}: d = \alpha^\phi
    ```
    ```math
    \text{width}: w = \beta^\phi
    ```
    ```math
    \text{resolution}: r = \gamma^\phi
    ```
    where $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (constraint from FLOPs), and $\phi$ is a compound coefficient.
  - **Key insight:** Balancing all three dimensions is more efficient than scaling only one.
  - **Base architecture (EfficientNet-B0):** Uses mobile inverted bottleneck (MBConv) blocks with squeeze-and-excitation.
  - Series B0–B7 with different $\phi$ values; good accuracy–efficiency trade-offs.
  - Achieves better accuracy than ResNet/Inception with fewer parameters and FLOPs.

**Example exam question (architecture comparison)**  
Compare VGG and ResNet: how do their building blocks differ, and why does ResNet scale to much deeper networks?

---

## 7. Transfer Learning Playbook

Typical workflow when you have limited labeled data:

1. **Start from an ImageNet-pretrained backbone.**  
   This provides generic low- and mid-level features (edges, textures, shapes).

2. **Replace the classifier head.**  
   Swap the original FC / GAP + FC head with a new head sized for your number of classes.

3. **Freeze most backbone layers.**  
   - Train only the new head first.  
   - Use a relatively higher LR for the head (e.g., $10^{-3}$) and lower LR for any unfrozen layers (e.g., $10^{-4}$).

4. **Progressively unfreeze.**  
   - If validation accuracy saturates, unfreeze deeper (later) blocks.
   - Keep LR small on pretrained layers to avoid destroying useful features.

5. **Augment and regularize.**  
   - Use augmentations that match your domain (e.g., flips for natural images, intensity jitter for medical).
   - Add label smoothing, dropout, and weight decay as needed.

6. **Monitor validation metrics and adjust schedule.**  
   - Use cosine or step LR decay.
   - Save best-performing checkpoints based on validation accuracy / loss.

**Example exam question (short)**  
Under what circumstances is transfer learning clearly preferable to training a CNN from scratch? Give two concrete scenarios.

---

## 8. From Lecture to Lab

Connect theory to specific lab tasks:

- **Lab 7.1 – Linear / MLP**
  - Goal: show parameter explosion and spatial information loss when flattening images.
  - Pipeline: initialize weights → matrix multiply → activation → CE loss → gradient descent update.
  - Compare performance and parameter counts to CNN-based models.

- **Lab 7.2 – CNN on MNIST**
  - **Model architecture:** Small 2-layer CNN + pooling + FC head.
    - Typical structure: Conv(32 filters, 3x3) → ReLU → MaxPool(2x2) → Conv(64 filters, 3x3) → ReLU → MaxPool(2x2) → Flatten → FC(128) → ReLU → FC(10) → Softmax
    - Input: $28 \times 28$ grayscale images (MNIST digits)
    - Output: 10 class logits (digits 0-9)
  - **Data preprocessing:** 
    - Normalize pixel values to [0,1] or use mean/std normalization
    - Convert to tensors, create DataLoader with batch size (typically 64-128)
  - **Training loop:**  
    ```python
    for epoch in range(num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()      # Clear gradients
            outputs = model(inputs)    # Forward pass
            loss = criterion(outputs, labels)  # Compute loss
            loss.backward()            # Backward pass (compute gradients)
            optimizer.step()           # Update weights
    ```
  - **Loss function:** Cross-entropy loss (combines LogSoftmax + NLLLoss)
  - **Optimizer:** Typically Adam or SGD with momentum, learning rate around $10^{-3}$ to $10^{-4}$
  - **Visualization:** 
    - Inspect learned filters (first conv layer) to see low-level edge detectors (horizontal, vertical, diagonal)
    - Visualize feature maps at different layers to understand what the network learns
    - Early layers: edges, strokes, simple patterns
    - Later layers: more complex digit parts and shapes

- **ResNet18 Fine-Tuning (take-home style)**
  - Start from ImageNet-pretrained ResNet18 backbone.
  - Replace final FC layer for your specific number of classes.
  - Use random resized crop, horizontal flip, and ImageNet mean / std normalization.
  - Use smaller LR for backbone, larger LR for head; add a scheduler (step or cosine) as in lecture.

---

## 9. Failure Modes + Fixes

- **Dying ReLUs**
  - Symptom: many activations are zero and never recover.
  - Fixes: use Leaky / Parametric ReLU, lower LR, check initialization.

- **Exploding / vanishing gradients**
  - Symptoms: gradients become huge (NaNs) or tiny (no learning).
  - Fixes: residual connections, BatchNorm, proper initialization, gradient clipping, smaller LR.

- **Overfitting (small data)**
  - Symptom: train accuracy high, validation accuracy much lower.
  - Fixes: stronger augmentation, dropout around $0.3$–$0.5$, weight decay $10^{-4}$–$5\cdot10^{-4}$, freeze more layers in transfer learning.

- **Training diverges or loss becomes NaN**
  - Fixes: reduce LR, add LR warmup, check data normalization, ensure BN is placed before activation (for most architectures).

- **Plateaued accuracy**
  - Fixes: adjust LR schedule (e.g., step down), unfreeze more layers, switch optimizer (e.g., SGD ↔ AdamW), add label smoothing.

---

## 10. Key Formulas

- **Convolution output size**

```math
W_\text{out} = \left\lfloor \frac{W_\text{in} - K + 2P}{S} \right\rfloor + 1
```

- **Cross-entropy loss (multi-class)**

```math
L = -\sum_i y_i \log(\hat y_i)
```

- **Softmax**

```math
\mathrm{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
```

- **Batch Normalization**

```math
y = \gamma \cdot \frac{x - \mu}{\sqrt{\mathrm{var} + \varepsilon}} + \beta
```

- **Residual connection**

```math
y = F(x) + x
```

- **Receptive field growth**

```math
RF_\text{new} = RF_\text{old} + (K - 1)\cdot \prod(\text{previous\_strides})
```

---

## 11. Rapid Q&A

1) **Why ReLU over sigmoid in deep CNNs?**  
   ReLU avoids saturation on the positive side, reduces vanishing gradients, is cheap to compute, and often leads to faster convergence.

2) **Why add pooling or use stride > 1?**  
   To downsample feature maps, gain translation tolerance, and reduce computation and memory.

3) **What problem does ResNet solve?**  
   The degradation problem: deeper networks performing worse due to optimization difficulties. Residual shortcuts stabilize gradients and enable very deep networks.

4) **How does BatchNorm help training?**  
   It normalizes intermediate activations, smooths the loss landscape, allows higher learning rates, and adds mild regularization.

5) **When is transfer learning recommended?**  
   When your dataset is limited or similar to large pretraining datasets (e.g., ImageNet), giving faster convergence and better generalization than training from scratch.

6) **How to size the classifier head?**  
   Prefer global average pooling followed by a small FC layer: fewer parameters, less overfitting, and often similar accuracy to large FC stacks.

---

## 12. Exam-Style Questions

Use these to test yourself without looking at notes, then check answers above.

1. **Conceptual (short answer)**  
   (a) Define sparse connectivity and parameter sharing in CNNs.  
   (b) Explain how they jointly reduce the risk of overfitting compared to MLPs on images.

2. **Shape and parameter calculation**  
   You have an input of size $64 \times 64 \times 3$. It passes through:
   - Conv1: $5 \times 5$, stride $1$, padding $2$, $32$ filters.  
   - MaxPool: $2 \times 2$, stride $2$.  
   - Conv2: $3 \times 3$, stride $1$, padding $1$, $64$ filters.  
   (a) Compute the spatial size and channel count after each layer.  
   (b) Compute the total number of parameters in Conv1 and Conv2.

3. **Receptive field reasoning**  
   Consider three $3 \times 3$ conv layers with stride $1$ followed by a $2 \times 2$ max-pooling with stride $2$.  
   (a) What is the receptive field of a neuron in the 3rd conv layer?  
   (b) How does pooling change the effective receptive field relative to the input?

4. **Architecture comparison**  
   Compare AlexNet, VGG, and ResNet with respect to:
   - Depth and use of small vs large kernels.  
   - Use of residual connections.  
   - How easy they are to scale to deeper networks.

5. **Transfer learning scenario**  
   You have 2,000 labeled images for a 5-class medical imaging task.  
   (a) Describe a complete transfer-learning plan using a pretrained ResNet18.  
   (b) Include which layers you would freeze initially, LR choices for head vs backbone, and how you would decide when to unfreeze more layers.

6. **Diagnosing failure cases**  
   A student trains a CNN from scratch on CIFAR-10. Training loss decreases slowly, gradients are often near zero, and deeper layers appear inactive.  
   List at least three likely causes and corresponding fixes (e.g., activation choice, initialization, architecture changes).

If you can answer these without looking, your CNN theory is in very strong shape for the exam.  

