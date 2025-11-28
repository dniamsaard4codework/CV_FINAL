# Unit 12: Generative Models (AE, VAE, GAN) – Exam-Ready Guide

How to use this sheet: skim bolded phrases and diagrams in your head for a 2-minute refresh, then drill sections with the derivations and exam-style questions. The structure matches Units 8–13 and uses LaTeX-style math.

## Table of Contents
1. [Exam Checklist](#1-exam-checklist)  
2. [Generative vs Discriminative](#2-generative-vs-discriminative)  
3. [Autoencoders (AE)](#3-autoencoders-ae)  
4. [Variational Autoencoders (VAE)](#4-variational-autoencoders-vae)  
5. [Generative Adversarial Networks (GAN)](#5-generative-adversarial-networks-gan)  
6. [Variants: Conditional GAN, DCGAN, etc.](#6-variants-conditional-gan-dcgan-etc)  
7. [Training Issues and Diagnostics](#7-training-issues-and-diagnostics)  
8. [From Lecture to Lab](#8-from-lecture-to-lab)  
9. [Key Formulas](#9-key-formulas)  
10. [Rapid Q&A](#10-rapid-qa)  
11. [Exam-Style Questions](#11-exam-style-questions)  

---

## 1. Exam Checklist

Be able to:

- Explain **generative vs discriminative** models and give examples of each in vision.
- Draw the **AE architecture**, state its loss, and explain bottleneck intuition.
- Derive the **VAE ELBO** at a high level and interpret each term.
- Write the **VAE loss** (reconstruction + KL) and explain why we add noise / reparameterization.
- Write the **GAN minimax objective**, derive the optimal discriminator, and interpret the Jensen–Shannon connection.
- Explain core **GAN failure modes** (mode collapse, vanishing gradients, instability) and common fixes.
- Describe at least one **conditional generative model** (e.g., cGAN) and how labels are injected.
- Connect theory to **labs**: GAN on images and VAE implementation details.

---

## 2. Generative vs Discriminative

- **Discriminative models**
  - Model $p(y \mid x)$ directly (classification, detection, segmentation).
  - Examples: logistic regression, standard CNN classifiers, object detectors.

- **Generative models**
  - Model $p(x)$ or joint $p(x, z)$ such that we can **sample** $x$.
  - Aim: learn data distribution (e.g., images) and generate new samples that “look like” training data.
  - In practice, we approximate sampling via:
    - Explicit likelihood (e.g., VAE, flows) or
    - Implicit adversarial training (GANs).

- **Why for vision?**
  - Image synthesis, super-resolution, inpainting, style transfer, data augmentation, anomaly detection, representation learning.

**Example exam question (short)**  
Contrast discriminative and generative models in terms of what probability distributions they model and give one vision task for each.

---

## 3. Autoencoders (AE)

**Goal:** learn a compressed representation (latent code) that can reconstruct the input.

### 3.1 Architecture

- **Encoder** $f_\phi(x)$: maps input image $x$ to latent vector $z$ (bottleneck).
- **Decoder** $g_\theta(z)$: maps latent $z$ back to reconstruction $\hat x$.
- End-to-end: $x \rightarrow z = f_\phi(x) \rightarrow \hat x = g_\theta(z)$.

### 3.2 Loss

- Reconstruction loss (for images often MSE or cross-entropy):

  ```math
  L_\text{AE}(\theta, \phi)
    = \mathbb{E}_{x \sim p_\text{data}(x)}
      \bigl[ \lVert x - g_\theta(f_\phi(x)) \rVert^2 \bigr]
  ```

- No explicit probabilistic generative model; AEs mainly learn **compressive representations**.

### 3.3 Bottleneck and latent space

- Bottleneck dimensionality $\dim(z)$ controls how much information must be compressed.
- Too small: underfitting, blurry reconstructions, loss of detail.
- Too large: can memorize inputs (identity function), poor generalization.
- Latent space can be used for:
  - Clustering.
  - Nearest-neighbor search.
  - Anomaly detection (high reconstruction error → out-of-distribution).

**Example exam question (AE)**  
Why might a simple convolutional AE not be a good *generative* model, even if reconstruction error is low?

---

## 4. Variational Autoencoders (VAE)

**Goal:** make AEs probabilistic and **generative**, learning a latent variable model $p_\theta(x, z)$.

### 4.1 Latent variable model

- Prior over latent variables: $p(z)$ (usually $\mathcal{N}(0, I)$).
- Likelihood (decoder): $p_\theta(x \mid z)$, e.g., Gaussian or Bernoulli with mean from a neural net.
- True posterior: $p_\theta(z \mid x)$ is intractable → approximate with encoder $q_\phi(z \mid x)$.

### 4.2 ELBO (Evidence Lower BOund)

We want to maximize log-likelihood:

```math
\log p_\theta(x) = \log \int p_\theta(x, z)\, dz
```

Introduce $q_\phi(z \mid x)$ and derive:

```math
\log p_\theta(x)
= \mathcal{L}_\text{ELBO}(\theta, \phi; x)
  + D_\text{KL}\bigl(q_\phi(z \mid x)\ \lVert\ p_\theta(z \mid x)\bigr)
```

Since KL is non-negative, ELBO is a lower bound:

```math
\mathcal{L}_\text{ELBO}(\theta, \phi; x)
  = \mathbb{E}_{z \sim q_\phi(z \mid x)}
      \bigl[\log p_\theta(x \mid z)\bigr]
    - D_\text{KL}\bigl(q_\phi(z \mid x)\ \lVert\ p(z)\bigr)
```

We **maximize ELBO** ≈ maximize log-likelihood.

- First term: reconstruction log-likelihood.
- Second term: KL regularizer pulling $q_\phi(z \mid x)$ towards prior $p(z)$.

### 4.3 Gaussian VAE and reparameterization

- Assume:

  ```math
  q_\phi(z \mid x) = \mathcal{N}\bigl(z \mid \mu_\phi(x), \operatorname{diag}(\sigma^2_\phi(x))\bigr)
  ```

- Reparameterization trick:

  ```math
  \epsilon \sim \mathcal{N}(0, I), \quad
  z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon
  ```

  This makes sampling differentiable w.r.t. $\mu_\phi, \sigma_\phi$.

- For Gaussian prior $p(z) = \mathcal{N}(0, I)$ and diagonal Gaussian $q_\phi$:

  ```math
  D_\text{KL}\bigl(q_\phi(z \mid x)\ \lVert\ p(z)\bigr)
    = -\frac{1}{2}\sum_i \bigl(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2 \bigr)
  ```

### 4.4 VAE loss (minimization form)

Typically we **minimize** negative ELBO:

```math
L_\text{VAE}(\theta, \phi)
  = \mathbb{E}_{x \sim p_\text{data}}
    \left[
      - \mathbb{E}_{z \sim q_\phi(z \mid x)} \log p_\theta(x \mid z)
      + D_\text{KL}\bigl(q_\phi(z \mid x)\ \lVert\ p(z)\bigr)
    \right]
```

- First term ≈ reconstruction loss.
- Second term: regularization encouraging latent codes to match the prior and be **smooth / continuous**.

**Interpretation:** VAE trades off **good reconstructions** and **latent space regularity**, enabling smooth interpolation and sampling from $p(z)$.

**Common issues:**
- Blurry samples (due to simple likelihood, e.g., Gaussian with MSE).
- KL collapse when decoder is too strong.

---

## 5. Generative Adversarial Networks (GAN)

**Goal:** learn a generator $G$ that maps noise to realistic samples, using a discriminator $D$ as an adversary instead of an explicit likelihood.

### 5.1 Basic setup

- Latent prior: $z \sim p_z(z)$ (e.g., $\mathcal{N}(0, I)$ or uniform).
- Generator: $G_\theta(z)$ outputs fake image $\tilde x$.
- Discriminator: $D_\phi(x)$ outputs probability “image is real”.

### 5.2 Minimax objective

Original GAN objective (data distribution $p_\text{data}$):

```math
\min_G \max_D V(D, G)
  = \mathbb{E}_{x \sim p_\text{data}(x)} \bigl[ \log D(x) \bigr]
  + \mathbb{E}_{z \sim p_z(z)} \bigl[ \log (1 - D(G(z))) \bigr]
```

- $D$ tries to maximize correct classification of real vs fake.
- $G$ tries to minimize it (fool $D$).

### 5.3 Optimal discriminator and JS divergence

For fixed $G$, the optimal discriminator is:

```math
D^\ast(x)
  = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}
```

Substitute into the value function:

```math
V(D^\ast, G)
  = -\log 4 + 2 \cdot \text{JS}\bigl(p_\text{data} \,\|\, p_G\bigr)
```

So minimizing the GAN objective wrt $G$ ≈ minimizing the Jensen–Shannon divergence between $p_\text{data}$ and $p_G$.

### 5.4 Generator loss variants

- Original “saturating” loss for $G$:

  ```math
  L_G^\text{sat} = \mathbb{E}_{z}[\log (1 - D(G(z)))]
  ```

  This can cause vanishing gradients when $D$ is strong.

- Non-saturating heuristic (commonly used):

  ```math
  L_G^\text{ns} = - \mathbb{E}_z [\log D(G(z))]
  ```

  Gives stronger gradients early in training.

**Intuition:** $D$ provides a learned loss landscape telling $G$ where its samples differ from real data.

---

## 6. Variants: Conditional GAN, DCGAN, etc.

### 6.1 Conditional GAN (cGAN)

- Condition both $G$ and $D$ on label $y$ or additional input (e.g., class, segmentation mask).
- Generator: $G(z, y)$; discriminator: $D(x, y)$.
- Objective:

  ```math
  \min_G \max_D
  \mathbb{E}_{x, y \sim p_\text{data}}
    [\log D(x, y)]
  + \mathbb{E}_{z \sim p_z, y \sim p(y)}
    [\log (1 - D(G(z, y), y))]
  ```

### 6.2 DCGAN

- Deep convolutional GAN with architectural guidelines:
  - Use strided conv / transposed conv instead of pooling.
  - Use BatchNorm in generator and discriminator (carefully).
  - Use ReLU in generator (except Tanh in output), LeakyReLU in discriminator.

### 6.3 Other variants (slide-level knowledge)

- WGAN / WGAN-GP: use Wasserstein distance with gradient penalty.
- LS-GAN: least-squares loss for more stable gradients.
- StyleGAN: style-based generator, controls attributes via latent codes.

---

## 7. Training Issues and Diagnostics

**Mode collapse**
- Generator maps many $z$ values to very similar outputs (few modes).
- Symptoms: diversity low, samples look repetitive.
- Partial fixes:
  - Minibatch discrimination, feature matching.
  - Using WGAN-GP / improved objectives.

**Vanishing / unstable gradients**
- Discriminator too strong → $D(x) \approx 1$, $D(G(z)) \approx 0$ → small gradients for $G$.
- Remedies:
  - Non-saturating loss, label smoothing, noisy labels.
  - Spectral normalization, gradient penalty.

**Evaluation**
- Visual inspection (qualitative).
- Quantitative: Inception Score, FID (Fréchet Inception Distance).

**VAE-specific issues**
- Blurry outputs due to pixel-wise losses.
- KL collapse: KL term becomes too small; posterior collapses to prior.

---

## 8. From Lecture to Lab

The slides and labs for this unit mention:
- **Generative Models:** AE, VAE, GAN.
- **Labs:** `lab11.1_GAN.ipynb`, `lab11.2_VAE.ipynb`.

### 8.1 `lab11.1_GAN.ipynb` (GAN lab)

- Implements a DCGAN-style architecture:
  - Generator: transposed convolutions (ConvTranspose2d) to upsample noise $z$ into images.
  - Discriminator: strided conv layers to classify real vs fake.
- Typical training loop:
  - Update $D$ with real images (label = 1) and generated images (label = 0).
  - Update $G$ to maximize $D(G(z))$ (non-saturating loss).
- Visualize generated images every few epochs to see training progress.

### 8.2 `lab11.2_VAE.ipynb` (VAE lab)

- Encoder network outputs $\mu_\phi(x)$ and $\log \sigma^2_\phi(x)$.
- Use reparameterization: $z = \mu + \sigma \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.
- Decoder network maps $z$ back to reconstruction $\hat x$.
- Loss:
  - Reconstruction term (e.g., MSE or BCE).
  - KL term using closed form for Gaussian vs standard normal.
- Plot samples by drawing $z \sim p(z)$ and passing through decoder.
- Interpolate in latent space: linearly interpolate between two $z$ vectors and decode each point.

---

## 9. Key Formulas

- **AE reconstruction loss (MSE form)**

  ```math
  L_\text{AE} = \mathbb{E}_x \bigl[\lVert x - \hat x \rVert^2\bigr]
  ```

- **VAE ELBO**

  ```math
  \mathcal{L}_\text{ELBO}
  = \mathbb{E}_{z \sim q_\phi(z \mid x)}
      [\log p_\theta(x \mid z)]
    - D_\text{KL}\bigl(q_\phi(z \mid x)\ \lVert\ p(z)\bigr)
  ```

- **Gaussian KL (diagonal case)**

  ```math
  D_\text{KL}\bigl(\mathcal{N}(\mu, \sigma^2)
                   \,\|\, \mathcal{N}(0, 1)\bigr)
    = -\frac{1}{2}\sum_i \bigl(1 + \log \sigma_i^2 - \mu_i^2 - \sigma_i^2\bigr)
  ```

- **GAN minimax objective**

  ```math
  \min_G \max_D
  \mathbb{E}_{x \sim p_\text{data}}[\log D(x)]
  + \mathbb{E}_{z \sim p_z}[\log(1 - D(G(z)))]
  ```

- **Optimal discriminator**

  ```math
  D^\ast(x) = \frac{p_\text{data}(x)}{p_\text{data}(x) + p_G(x)}
  ```

- **Non-saturating generator loss**

  ```math
  L_G = -\mathbb{E}_z[\log D(G(z))]
  ```

---

## 10. Rapid Q&A

1) **How is a VAE different from a plain AE?**  
   VAE is probabilistic: it learns a distribution $q_\phi(z \mid x)$ and maximizes a likelihood lower bound, enabling principled sampling from $p(z)$; an AE just learns a deterministic code and reconstruction.

2) **Why use the reparameterization trick?**  
   To make sampling from $q_\phi(z \mid x)$ differentiable w.r.t. encoder parameters, allowing gradients to flow through the sampling operation.

3) **Why can GANs produce sharper images than VAEs?**  
   GANs do not rely on simple pixel-wise likelihoods; instead they use an adversarial loss that can capture complex, perceptual differences between real and fake images.

4) **What is mode collapse?**  
   When the generator maps many different $z$ values to a small set of outputs (few modes), reducing diversity in generated samples.

5) **How does a conditional GAN differ from a vanilla GAN?**  
   In a cGAN, both $G$ and $D$ receive additional conditioning information (e.g., labels), so $G$ learns $p(x \mid y)$ and can generate specific classes on demand.

---

## 11. Exam-Style Questions

Use these to check deep understanding of Unit 12.

1. **AE vs VAE (concept + math)**  
   (a) Explain conceptually how a VAE turns an autoencoder into a generative model.  
   (b) Starting from $\log p_\theta(x) = \log \int p_\theta(x, z)\, dz$, derive the ELBO and identify the reconstruction and KL terms.  
   (c) Why does minimizing the KL term encourage a “smooth” latent space?

2. **VAE loss decomposition (numeric)**  
   Suppose for a single data point you observe: reconstruction loss (negative log-likelihood) $= 50$, KL divergence $= 2$.  
   (a) What is the negative ELBO value for this point?  
   (b) If you multiply the KL term by a factor $\beta = 4$ (as in $\beta$-VAE), how does the loss change?  
   (c) Qualitatively, what effect does this have on reconstructions vs disentanglement?

3. **GAN optimal discriminator**  
   (a) Starting from the GAN objective, derive the formula for the optimal discriminator $D^\ast(x)$.  
   (b) Show that plugging $D^\ast$ back in yields a cost involving $\text{JS}(p_\text{data} \,\|\, p_G)$.  
   (c) Explain why, in practice, we often switch to the non-saturating generator loss instead of strictly following the minimax formulation.

4. **Diagnosing GAN training problems**  
   You train a GAN on faces. After some epochs:
   - Discriminator accuracy is very high.
   - Generator samples barely change and look like noise.  
   (a) Give three potential issues that may be happening (e.g., LR choices, architecture imbalance, loss choice).  
   (b) Propose corresponding concrete fixes for each issue.

5. **Conditional GAN application**  
   You want to generate clothing images conditioned on categories (shirt, pants, shoes).  
   (a) Explain how you would modify the architecture of $G$ and $D$ to incorporate the categorical label.  
   (b) Describe how the training data and loss change relative to a vanilla GAN.  
   (c) Give one advantage and one potential drawback of using a cGAN in this setting.

6. **Lab interpretation**  
   From the VAE lab (`lab11.2_VAE.ipynb`), you plot latent space interpolations between digits 1 and 9.  
   (a) Explain why linear interpolation in $z$ space can produce a smooth morphing between classes.  
   (b) Describe what you would expect to see if the KL term were removed entirely (i.e., using a plain AE instead).

