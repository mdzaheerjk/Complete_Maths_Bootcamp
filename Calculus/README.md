# 📐 Calculus for Machine Learning & Deep Learning
### Complete Notes — From Basics to Advanced

> **"To understand ML/DL deeply, you must master Calculus. Every gradient, every loss function, every optimization step is Calculus in disguise."**

---

## 📚 Table of Contents

1. [Foundation: Functions & Limits](#1-foundation-functions--limits)
2. [Derivatives — The Core Engine](#2-derivatives--the-core-engine)
3. [Partial Derivatives & Multivariable Calculus](#3-partial-derivatives--multivariable-calculus)
4. [The Chain Rule — Heart of Backpropagation](#4-the-chain-rule--heart-of-backpropagation)
5. [Gradients & Gradient Vectors](#5-gradients--gradient-vectors)
6. [Jacobians & Hessians](#6-jacobians--hessians)
7. [Taylor Series & Approximations](#7-taylor-series--approximations)
8. [Optimization Theory](#8-optimization-theory)
9. [Gradient Descent & Variants](#9-gradient-descent--variants)
10. [Integration & Probability Connection](#10-integration--probability-connection)
11. [Matrix Calculus](#11-matrix-calculus)
12. [Backpropagation — Full Derivation](#12-backpropagation--full-derivation)
13. [Activation Functions & Their Derivatives](#13-activation-functions--their-derivatives)
14. [Loss Functions & Their Gradients](#14-loss-functions--their-gradients)
15. [Advanced Topics: Lagrange Multipliers, KKT](#15-advanced-topics-lagrange-multipliers-kkt)
16. [Calculus in Attention & Transformers](#16-calculus-in-attention--transformers)
17. [NumPy & Calculus Connections](#17-numpy--calculus-connections)

---

## 1. Foundation: Functions & Limits

### 1.1 What is a Function?

A function **f** maps input **x** to output **y**:

$$f: \mathbb{R}^n \rightarrow \mathbb{R}^m$$

In ML, a neural network is literally a composed function:

$$\hat{y} = f(x) = f_L(f_{L-1}(\cdots f_1(x)))$$

### 1.2 Limits

The limit of f(x) as x approaches a:

$$\lim_{x \to a} f(x) = L$$

**Formal definition (ε-δ):**

$$\forall \varepsilon > 0,\ \exists \delta > 0 \text{ such that } |x - a| < \delta \Rightarrow |f(x) - L| < \varepsilon$$

**Important limit for ML:**

$$\lim_{h \to 0} \frac{f(x+h) - f(x)}{h} = f'(x) \quad \text{(definition of derivative)}$$

### 1.3 Continuity

f(x) is continuous at a if:

$$\lim_{x \to a} f(x) = f(a)$$

> **ML Note:** Activation functions like ReLU are continuous everywhere but not differentiable at x = 0. This is handled with subgradients.

### 1.4 Key Limits to Know

| Limit | Value |
|-------|-------|
| $\lim_{x \to 0} \frac{\sin x}{x}$ | 1 |
| $\lim_{x \to \infty} \left(1 + \frac{1}{x}\right)^x$ | $e$ |
| $\lim_{x \to 0} \frac{e^x - 1}{x}$ | 1 |
| $\lim_{x \to 0} \frac{\ln(1+x)}{x}$ | 1 |

---

## 2. Derivatives — The Core Engine

### 2.1 Definition of Derivative

The derivative measures the **instantaneous rate of change**:

$$f'(x) = \frac{df}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Geometric interpretation:** Slope of the tangent line to f(x) at point x.

**ML interpretation:** How much does the output change when the input changes slightly? This is the sensitivity of the model.

### 2.2 Basic Differentiation Rules

| Rule | Formula |
|------|---------|
| Constant | $\frac{d}{dx}[c] = 0$ |
| Power | $\frac{d}{dx}[x^n] = nx^{n-1}$ |
| Exponential | $\frac{d}{dx}[e^x] = e^x$ |
| Natural Log | $\frac{d}{dx}[\ln x] = \frac{1}{x}$ |
| Sine | $\frac{d}{dx}[\sin x] = \cos x$ |
| Cosine | $\frac{d}{dx}[\cos x] = -\sin x$ |
| Sum | $\frac{d}{dx}[f + g] = f' + g'$ |
| Product | $\frac{d}{dx}[f \cdot g] = f'g + fg'$ |
| Quotient | $\frac{d}{dx}\left[\frac{f}{g}\right] = \frac{f'g - fg'}{g^2}$ |
| Chain | $\frac{d}{dx}[f(g(x))] = f'(g(x)) \cdot g'(x)$ |

### 2.3 Higher-Order Derivatives

**Second derivative** (measures curvature):

$$f''(x) = \frac{d^2f}{dx^2} = \frac{d}{dx}\left[\frac{df}{dx}\right]$$

**n-th derivative:**

$$f^{(n)}(x) = \frac{d^n f}{dx^n}$$

> **ML Note:** Second derivatives are used in second-order optimization methods like Newton's method.

### 2.4 Example: Derivative of Sigmoid

The sigmoid function used in neural networks:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Derivative derivation:**

$$\sigma'(x) = \frac{d}{dx}\left[\frac{1}{1+e^{-x}}\right] = \frac{e^{-x}}{(1+e^{-x})^2}$$

**Elegant form:**

$$\boxed{\sigma'(x) = \sigma(x)(1 - \sigma(x))}$$

This is crucial — the derivative of sigmoid can be expressed using sigmoid itself, making backprop efficient.

---

## 3. Partial Derivatives & Multivariable Calculus

### 3.1 Functions of Multiple Variables

In ML, loss is a function of many parameters:

$$\mathcal{L}(w_1, w_2, \ldots, w_n, b_1, b_2, \ldots)$$

### 3.2 Partial Derivative

The partial derivative of f with respect to $x_i$ holds all other variables constant:

$$\frac{\partial f}{\partial x_i} = \lim_{h \to 0} \frac{f(x_1, \ldots, x_i + h, \ldots, x_n) - f(x_1, \ldots, x_i, \ldots, x_n)}{h}$$

**Example:** Let $f(x, y) = x^2 y + 3xy^2$

$$\frac{\partial f}{\partial x} = 2xy + 3y^2 \quad \text{(treat y as constant)}$$

$$\frac{\partial f}{\partial y} = x^2 + 6xy \quad \text{(treat x as constant)}$$

### 3.3 Directional Derivative

The rate of change in direction of unit vector **u**:

$$D_{\mathbf{u}} f(\mathbf{x}) = \nabla f(\mathbf{x}) \cdot \mathbf{u} = \|\nabla f\| \cos\theta$$

where θ is the angle between ∇f and **u**.

**Maximum rate of change** occurs in the direction of the gradient ∇f.

### 3.4 Second-Order Partial Derivatives

$$\frac{\partial^2 f}{\partial x^2}, \quad \frac{\partial^2 f}{\partial y^2}, \quad \frac{\partial^2 f}{\partial x \partial y}$$

**Schwarz's theorem (symmetry of mixed partials):**

$$\frac{\partial^2 f}{\partial x \partial y} = \frac{\partial^2 f}{\partial y \partial x}$$

(valid when second partials are continuous)

### 3.5 Example: Loss as a Multivariable Function

For linear regression, loss:

$$\mathcal{L}(w, b) = \frac{1}{n}\sum_{i=1}^{n}(y_i - (wx_i + b))^2$$

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{-2}{n}\sum_{i=1}^{n} x_i(y_i - wx_i - b)$$

$$\frac{\partial \mathcal{L}}{\partial b} = \frac{-2}{n}\sum_{i=1}^{n} (y_i - wx_i - b)$$

---

## 4. The Chain Rule — Heart of Backpropagation

### 4.1 Single Variable Chain Rule

If $y = f(u)$ and $u = g(x)$, then:

$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Example:** $y = \sin(x^2)$

Let $u = x^2$, so $y = \sin(u)$:

$$\frac{dy}{dx} = \cos(u) \cdot 2x = 2x\cos(x^2)$$

### 4.2 Multivariable Chain Rule

If $z = f(x, y)$ and $x = g(t)$, $y = h(t)$:

$$\frac{dz}{dt} = \frac{\partial z}{\partial x}\frac{dx}{dt} + \frac{\partial z}{\partial y}\frac{dy}{dt}$$

**General form:** If $z = f(x_1, \ldots, x_n)$ and each $x_i = g_i(t_1, \ldots, t_m)$:

$$\frac{\partial z}{\partial t_j} = \sum_{i=1}^{n} \frac{\partial z}{\partial x_i} \cdot \frac{\partial x_i}{\partial t_j}$$

### 4.3 Computational Graph View

Consider: $\mathcal{L} = f(g(h(x)))$

```
x → [h] → u → [g] → v → [f] → L
```

**Forward pass (compute values):**

$$u = h(x), \quad v = g(u), \quad \mathcal{L} = f(v)$$

**Backward pass (chain rule):**

$$\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial v} \cdot \frac{\partial v}{\partial u} \cdot \frac{\partial u}{\partial x}$$

> **This IS backpropagation.** Neural networks are just very deep compositions of functions.

### 4.4 Chain Rule for Neural Network Layer

For a layer: $a^{(l)} = \sigma(z^{(l)})$ where $z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \frac{\partial \mathcal{L}}{\partial a^{(l)}} \cdot \frac{\partial a^{(l)}}{\partial z^{(l)}} \cdot \frac{\partial z^{(l)}}{\partial W^{(l)}}$$

Define **delta** (error signal): $\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$

Then:

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} \cdot (a^{(l-1)})^\top$$

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

---

## 5. Gradients & Gradient Vectors

### 5.1 Definition of Gradient

The gradient is the vector of all partial derivatives:

$$\nabla f(\mathbf{x}) = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix}$$

The gradient **points in the direction of steepest ascent**.

**For descent:** move in direction $-\nabla f$

### 5.2 Properties of Gradients

| Property | Formula |
|----------|---------|
| Linearity | $\nabla(af + bg) = a\nabla f + b\nabla g$ |
| Product rule | $\nabla(fg) = f\nabla g + g\nabla f$ |
| Chain rule | $\nabla f(g(\mathbf{x})) = f'(g(\mathbf{x})) \cdot \nabla g(\mathbf{x})$ |

### 5.3 Gradient of Common ML Operations

**Dot product:** $f(\mathbf{w}) = \mathbf{w}^\top \mathbf{x}$

$$\nabla_\mathbf{w} f = \mathbf{x}$$

**Quadratic form:** $f(\mathbf{w}) = \mathbf{w}^\top A \mathbf{w}$

$$\nabla_\mathbf{w} f = (A + A^\top)\mathbf{w} = 2A\mathbf{w} \quad \text{(if A symmetric)}$$

**L2 norm squared:** $f(\mathbf{w}) = \|\mathbf{w}\|^2 = \mathbf{w}^\top \mathbf{w}$

$$\nabla_\mathbf{w} f = 2\mathbf{w}$$

**L2 regularization term:** $f(\mathbf{w}) = \lambda \|\mathbf{w}\|^2$

$$\nabla_\mathbf{w} f = 2\lambda \mathbf{w}$$

### 5.4 Example: MSE Gradient

$$\mathcal{L}(\mathbf{w}) = \frac{1}{n}\|X\mathbf{w} - \mathbf{y}\|^2 = \frac{1}{n}(X\mathbf{w} - \mathbf{y})^\top(X\mathbf{w} - \mathbf{y})$$

$$\nabla_\mathbf{w} \mathcal{L} = \frac{2}{n} X^\top (X\mathbf{w} - \mathbf{y})$$

Setting gradient to zero → **Normal Equations:**

$$X^\top X \mathbf{w} = X^\top \mathbf{y} \implies \mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$$

---

## 6. Jacobians & Hessians

### 6.1 The Jacobian Matrix

For a vector-valued function $\mathbf{f}: \mathbb{R}^n \rightarrow \mathbb{R}^m$:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Key property:**

$$J_{ij} = \frac{\partial f_i}{\partial x_j}$$

### 6.2 Jacobian in Neural Networks

For a layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$:

$$\frac{\partial \mathbf{z}}{\partial \mathbf{x}} = W \quad \text{(Jacobian = weight matrix)}$$

For softmax $\mathbf{s} = \text{softmax}(\mathbf{z})$:

$$\frac{\partial s_i}{\partial z_j} = s_i(\delta_{ij} - s_j)$$

where $\delta_{ij}$ is the Kronecker delta.

In matrix form:

$$\frac{\partial \mathbf{s}}{\partial \mathbf{z}} = \text{diag}(\mathbf{s}) - \mathbf{s}\mathbf{s}^\top$$

### 6.3 The Hessian Matrix

The Hessian contains all second-order partial derivatives:

$$H = \nabla^2 f = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \frac{\partial^2 f}{\partial x_1 \partial x_2} & \cdots \\ \frac{\partial^2 f}{\partial x_2 \partial x_1} & \frac{\partial^2 f}{\partial x_2^2} & \cdots \\ \vdots & & \ddots \end{bmatrix} \in \mathbb{R}^{n \times n}$$

**Symmetry:** $H_{ij} = H_{ji}$ (when second partials are continuous)

### 6.4 Hessian and Curvature

The Hessian encodes local curvature information:

| Hessian property | Interpretation |
|-----------------|----------------|
| All eigenvalues > 0 (positive definite) | Local minimum |
| All eigenvalues < 0 (negative definite) | Local maximum |
| Mixed sign eigenvalues | Saddle point |
| Some eigenvalues = 0 | Degenerate case |

**Example:** $f(x, y) = x^2 + 4y^2$

$$H = \begin{bmatrix} 2 & 0 \\ 0 & 8 \end{bmatrix}$$

Eigenvalues: 2, 8 — both positive → local (global) minimum at (0, 0).

### 6.5 Newton's Method (Second-Order)

Uses Hessian to update parameters:

$$\mathbf{x}_{t+1} = \mathbf{x}_t - H^{-1} \nabla f(\mathbf{x}_t)$$

> **Why not used in deep learning?** For a model with $n$ parameters, the Hessian is $n \times n$. For GPT-3 with 175 billion parameters, that's $175\text{B} \times 175\text{B}$ matrix — computationally impossible. Hence we use first-order methods.

---

## 7. Taylor Series & Approximations

### 7.1 Taylor Series (Single Variable)

Any smooth function can be approximated around point $a$:

$$f(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \frac{f'''(a)}{3!}(x-a)^3 + \cdots$$

$$\boxed{f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n}$$

When $a = 0$ → **Maclaurin Series**

### 7.2 Common Maclaurin Series

| Function | Series |
|---------|--------|
| $e^x$ | $1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$ |
| $\sin x$ | $x - \frac{x^3}{3!} + \frac{x^5}{5!} - \cdots$ |
| $\cos x$ | $1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \cdots$ |
| $\ln(1+x)$ | $x - \frac{x^2}{2} + \frac{x^3}{3} - \cdots$ |
| $\frac{1}{1-x}$ | $1 + x + x^2 + x^3 + \cdots$ |
| $(1+x)^\alpha$ | $1 + \alpha x + \frac{\alpha(\alpha-1)}{2!}x^2 + \cdots$ |

### 7.3 Multivariate Taylor Expansion

For $f: \mathbb{R}^n \rightarrow \mathbb{R}$ around point $\mathbf{a}$:

**First order (linear approximation):**

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a})$$

**Second order (quadratic approximation):**

$$f(\mathbf{x}) \approx f(\mathbf{a}) + \nabla f(\mathbf{a})^\top (\mathbf{x} - \mathbf{a}) + \frac{1}{2}(\mathbf{x} - \mathbf{a})^\top H(\mathbf{a}) (\mathbf{x} - \mathbf{a})$$

### 7.4 ML Application: Why Gradient Descent Works

The linear Taylor approximation tells us that near $\mathbf{w}$:

$$\mathcal{L}(\mathbf{w} + \Delta\mathbf{w}) \approx \mathcal{L}(\mathbf{w}) + \nabla \mathcal{L}(\mathbf{w})^\top \Delta\mathbf{w}$$

To decrease $\mathcal{L}$, choose:

$$\Delta\mathbf{w} = -\eta \nabla \mathcal{L}(\mathbf{w})$$

This gives:

$$\mathcal{L}(\mathbf{w} + \Delta\mathbf{w}) \approx \mathcal{L}(\mathbf{w}) - \eta \|\nabla \mathcal{L}(\mathbf{w})\|^2$$

Since $\eta > 0$ and $\|\nabla \mathcal{L}\|^2 \geq 0$, the loss decreases. ✓

### 7.5 Learning Rate & Trust Region

The Taylor approximation is only valid in a **small neighborhood** of $\mathbf{w}$. The learning rate $\eta$ controls how far we step — a large $\eta$ violates the linear approximation, causing divergence.

---

## 8. Optimization Theory

### 8.1 Unconstrained Optimization

**Problem:** Find $\mathbf{x}^* = \arg\min_{\mathbf{x}} f(\mathbf{x})$

**First-order necessary condition (FONC):**

$$\nabla f(\mathbf{x}^*) = \mathbf{0}$$

Points where $\nabla f = \mathbf{0}$ are called **critical points** (stationary points).

**Second-order sufficient condition:**

- Local minimum: $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H(\mathbf{x}^*)$ is positive definite
- Local maximum: $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H(\mathbf{x}^*)$ is negative definite
- Saddle point: $\nabla f(\mathbf{x}^*) = \mathbf{0}$ and $H(\mathbf{x}^*)$ has mixed eigenvalues

### 8.2 Convexity

A function $f$ is **convex** if for all $\mathbf{x}, \mathbf{y}$ and $\lambda \in [0,1]$:

$$f(\lambda \mathbf{x} + (1-\lambda)\mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda)f(\mathbf{y})$$

**Equivalent condition:** $f$ is convex iff $H(\mathbf{x}) \succeq 0$ (positive semi-definite) everywhere.

**For convex functions:** Any local minimum is a global minimum.

### 8.3 Convex vs Non-Convex

| Property | Convex | Non-Convex |
|---------|--------|-----------|
| Examples | Linear regression, Logistic regression | Neural networks |
| Local minima | = Global minima | Many local minima |
| Saddle points | No saddle points | Ubiquitous |
| Optimization guarantee | Always find global opt | No guarantee |

> **Deep learning insight:** Despite non-convexity, deep networks train well in practice. Many local minima have similar loss values. Saddle points are the bigger challenge.

### 8.4 Lipschitz Continuity of Gradients

The gradient is **L-smooth** if:

$$\|\nabla f(\mathbf{x}) - \nabla f(\mathbf{y})\| \leq L\|\mathbf{x} - \mathbf{y}\|$$

This ensures gradient descent converges with step size $\eta \leq \frac{1}{L}$.

**Convergence bound:**

$$f(\mathbf{x}_T) - f(\mathbf{x}^*) \leq \frac{\|\mathbf{x}_0 - \mathbf{x}^*\|^2}{2\eta T}$$

---

## 9. Gradient Descent & Variants

### 9.1 Vanilla Gradient Descent

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w} \mathcal{L}(\mathbf{w}_t)$$

where $\eta$ is the learning rate.

### 9.2 Stochastic Gradient Descent (SGD)

Uses a single sample or mini-batch:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \nabla_\mathbf{w} \mathcal{L}_i(\mathbf{w}_t)$$

where $\mathcal{L}_i$ is loss on sample $i$ (or mini-batch).

**Unbiased estimator:** $\mathbb{E}[\nabla \mathcal{L}_i] = \nabla \mathcal{L}$

### 9.3 Momentum

Accumulates velocity in directions of consistent gradient:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + (1-\beta) \nabla \mathcal{L}(\mathbf{w}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{v}_{t+1}$$

where $\beta \in [0, 1)$ is the momentum coefficient (typically 0.9).

**Physics analogy:** Like a ball rolling down a hill — it gains momentum in consistent directions.

### 9.4 Nesterov Accelerated Gradient (NAG)

"Look ahead" before computing gradient:

$$\mathbf{v}_{t+1} = \beta \mathbf{v}_t + \eta \nabla \mathcal{L}(\mathbf{w}_t - \beta \mathbf{v}_t)$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \mathbf{v}_{t+1}$$

**Convergence rate:** $O(1/T^2)$ vs $O(1/T)$ for vanilla GD on convex problems.

### 9.5 AdaGrad

Adapts learning rate per parameter — larger updates for infrequent parameters:

$$G_t = G_{t-1} + (\nabla \mathcal{L}_t)^2 \quad \text{(element-wise)}$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{G_t + \varepsilon}} \odot \nabla \mathcal{L}_t$$

**Problem:** $G_t$ keeps growing → learning rate → 0 (training stops)

### 9.6 RMSProp

Exponentially weighted average of squared gradients:

$$v_t = \beta v_{t-1} + (1-\beta)(\nabla \mathcal{L}_t)^2$$

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{v_t + \varepsilon}} \odot \nabla \mathcal{L}_t$$

Fixes AdaGrad's diminishing learning rate problem.

### 9.7 Adam (Adaptive Moment Estimation)

Combines momentum + RMSProp:

**First moment (mean of gradients):**

$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla \mathcal{L}_t$$

**Second moment (mean of squared gradients):**

$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla \mathcal{L}_t)^2$$

**Bias correction:**

$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$

**Update:**

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t$$

**Typical hyperparameters:** $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\varepsilon = 10^{-8}$

### 9.8 AdamW (Adam with Weight Decay)

Decouples L2 regularization from gradient-based update:

$$\mathbf{w}_{t+1} = \mathbf{w}_t - \frac{\eta}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t - \eta \lambda \mathbf{w}_t$$

where $\lambda$ is the weight decay coefficient. Used in modern transformers (BERT, GPT).

### 9.9 Learning Rate Schedules

**Step decay:**

$$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/s \rfloor}$$

**Exponential decay:**

$$\eta_t = \eta_0 \cdot e^{-\lambda t}$$

**Cosine annealing:**

$$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{t}{T}\pi\right)\right)$$

**Warmup + Decay (Transformers):**

$$\eta_t = d_{\text{model}}^{-0.5} \cdot \min(t^{-0.5},\ t \cdot t_{\text{warmup}}^{-1.5})$$

---

## 10. Integration & Probability Connection

### 10.1 Fundamental Theorem of Calculus

If $F'(x) = f(x)$, then:

$$\int_a^b f(x)\, dx = F(b) - F(a)$$

### 10.2 Probability Density Functions

For a continuous random variable $X$ with PDF $p(x)$:

$$\int_{-\infty}^{\infty} p(x)\, dx = 1 \quad \text{(normalization)}$$

$$P(a \leq X \leq b) = \int_a^b p(x)\, dx$$

**Expected value:**

$$\mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot p(x)\, dx$$

**Variance:**

$$\text{Var}(X) = \mathbb{E}[(X - \mu)^2] = \int_{-\infty}^{\infty} (x-\mu)^2 p(x)\, dx$$

### 10.3 KL Divergence (Integral Form)

Measures how much distribution Q differs from P:

$$D_{KL}(P \| Q) = \int_{-\infty}^{\infty} p(x) \ln\frac{p(x)}{q(x)}\, dx \geq 0$$

$$D_{KL}(P \| Q) = 0 \iff P = Q \text{ almost everywhere}$$

Used in VAEs, knowledge distillation, RL.

### 10.4 Cross-Entropy (Discrete)

$$H(P, Q) = -\sum_x P(x) \log Q(x) = H(P) + D_{KL}(P\|Q)$$

### 10.5 Expected Loss & Empirical Risk Minimization

**True risk (population):**

$$R(f) = \mathbb{E}_{(x,y)\sim \mathcal{D}}[\mathcal{L}(f(x), y)] = \int \mathcal{L}(f(x), y)\, p(x, y)\, dx\, dy$$

**Empirical risk (training set approximation):**

$$\hat{R}(f) = \frac{1}{n}\sum_{i=1}^{n} \mathcal{L}(f(x_i), y_i)$$

The fundamental goal: minimize empirical risk so true risk is also small.

### 10.6 Gaussian Distribution

$$p(x; \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Key integral:**

$$\int_{-\infty}^{\infty} e^{-x^2}\, dx = \sqrt{\pi} \quad \text{(Gaussian integral)}$$

---

## 11. Matrix Calculus

### 11.1 Notation Conventions

For $f: \mathbb{R}^n \rightarrow \mathbb{R}$ and $\mathbf{x} \in \mathbb{R}^n$:

$$\frac{\partial f}{\partial \mathbf{x}} = \nabla_\mathbf{x} f = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n \quad \text{(column vector)}$$

### 11.2 Derivative with respect to Matrices

For $f: \mathbb{R}^{m \times n} \rightarrow \mathbb{R}$ and $W \in \mathbb{R}^{m \times n}$:

$$\frac{\partial f}{\partial W} = \begin{bmatrix} \frac{\partial f}{\partial W_{11}} & \cdots & \frac{\partial f}{\partial W_{1n}} \\ \vdots & \ddots & \vdots \\ \frac{\partial f}{\partial W_{m1}} & \cdots & \frac{\partial f}{\partial W_{mn}} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

Same shape as W — very convenient!

### 11.3 Key Matrix Derivative Identities

| Expression | Derivative w.r.t. **x** |
|-----------|------------------------|
| $\mathbf{a}^\top \mathbf{x}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{a}$ | $\mathbf{a}$ |
| $\mathbf{x}^\top \mathbf{x}$ | $2\mathbf{x}$ |
| $\mathbf{x}^\top A \mathbf{x}$ | $(A + A^\top)\mathbf{x}$ |
| $\|A\mathbf{x} - \mathbf{b}\|^2$ | $2A^\top(A\mathbf{x} - \mathbf{b})$ |
| $\mathbf{a}^\top \mathbf{x} \mathbf{x}^\top \mathbf{b}$ | $(\mathbf{a}\mathbf{b}^\top + \mathbf{b}\mathbf{a}^\top)\mathbf{x}$ |

### 11.4 Derivative w.r.t. Matrix W

| Expression | Derivative w.r.t. W |
|-----------|---------------------|
| $\mathbf{a}^\top W \mathbf{b}$ | $\mathbf{a}\mathbf{b}^\top$ |
| $\text{tr}(W)$ | $I$ |
| $\text{tr}(AW)$ | $A^\top$ |
| $\text{tr}(W^\top A)$ | $A$ |
| $\det(W)$ | $\det(W) \cdot W^{-\top}$ |
| $\ln\det(W)$ | $W^{-\top}$ |

### 11.5 Linear Layer Gradient (Full Derivation)

Forward: $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, loss $\mathcal{L}(\mathbf{z})$

**Gradient w.r.t. input:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = W^\top \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$$

**Gradient w.r.t. weights:**

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}} \mathbf{x}^\top$$

**Gradient w.r.t. bias:**

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{\partial \mathcal{L}}{\partial \mathbf{z}}$$

### 11.6 Batch Gradient (Mini-batch of size B)

For batch input $X \in \mathbb{R}^{B \times n}$:

$$\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{B} \left(\frac{\partial \mathcal{L}}{\partial Z}\right)^\top X$$

$$\frac{\partial \mathcal{L}}{\partial \mathbf{b}} = \frac{1}{B} \sum_{i=1}^{B} \frac{\partial \mathcal{L}}{\partial \mathbf{z}_i}$$

---

## 12. Backpropagation — Full Derivation

### 12.1 Neural Network Setup

**L-layer network:**

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}$$

$$a^{(l)} = \sigma^{(l)}(z^{(l)})$$

where $a^{(0)} = \mathbf{x}$ (input), $\hat{y} = a^{(L)}$ (output).

### 12.2 Forward Pass (Computing Activations)

For $l = 1, 2, \ldots, L$:

$$z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}, \quad a^{(l)} = \sigma(z^{(l)})$$

### 12.3 Backward Pass (Computing Gradients)

Define error signal at layer $l$:

$$\delta^{(l)} = \frac{\partial \mathcal{L}}{\partial z^{(l)}}$$

**Output layer:** (using chain rule)

$$\delta^{(L)} = \frac{\partial \mathcal{L}}{\partial a^{(L)}} \odot \sigma'^{(L)}(z^{(L)})$$

**Hidden layer** (backpropagating error):

$$\delta^{(l)} = \left[(W^{(l+1)})^\top \delta^{(l+1)}\right] \odot \sigma'^{(l)}(z^{(l)})$$

**Weight gradient:**

$$\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top$$

**Bias gradient:**

$$\frac{\partial \mathcal{L}}{\partial b^{(l)}} = \delta^{(l)}$$

### 12.4 Full Backprop Algorithm

```
ALGORITHM: Backpropagation

INPUT: Network {W^(l), b^(l)}, input x, target y

FORWARD PASS:
  a^(0) = x
  for l = 1 to L:
    z^(l) = W^(l) * a^(l-1) + b^(l)
    a^(l) = σ(z^(l))

COMPUTE LOSS:
  L = Loss(a^(L), y)

BACKWARD PASS:
  δ^(L) = ∂L/∂a^(L) ⊙ σ'(z^(L))
  for l = L-1 down to 1:
    δ^(l) = (W^(l+1))ᵀ δ^(l+1) ⊙ σ'(z^(l))

GRADIENT COMPUTATION:
  for l = 1 to L:
    ∂L/∂W^(l) = δ^(l) (a^(l-1))ᵀ
    ∂L/∂b^(l) = δ^(l)

PARAMETER UPDATE:
  W^(l) ← W^(l) - η * ∂L/∂W^(l)
  b^(l) ← b^(l) - η * ∂L/∂b^(l)
```

### 12.5 Computational Complexity

| Phase | Time | Space |
|-------|------|-------|
| Forward | $O(L \cdot n^2)$ | $O(L \cdot n)$ — must store all activations |
| Backward | $O(L \cdot n^2)$ | $O(L \cdot n)$ |

The need to store all intermediate activations is why memory is the bottleneck in training large models (gradient checkpointing trades compute for memory).

---

## 13. Activation Functions & Their Derivatives

### 13.1 Sigmoid

$$\sigma(x) = \frac{1}{1+e^{-x}}, \quad \sigma'(x) = \sigma(x)(1-\sigma(x))$$

- Range: $(0, 1)$
- Problem: **Vanishing gradient** when $|x|$ is large ($\sigma' \approx 0$)
- Use: Binary classification output

### 13.2 Hyperbolic Tangent (tanh)

$$\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}, \quad \tanh'(x) = 1 - \tanh^2(x)$$

- Range: $(-1, 1)$, zero-centered
- Still suffers vanishing gradients for large $|x|$
- Relation to sigmoid: $\tanh(x) = 2\sigma(2x) - 1$

### 13.3 ReLU (Rectified Linear Unit)

$$\text{ReLU}(x) = \max(0, x) = \begin{cases} x & x > 0 \\ 0 & x \leq 0 \end{cases}$$

$$\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined (use 0)} & x = 0 \end{cases}$$

- No vanishing gradient for $x > 0$
- Problem: **Dying ReLU** — neurons stuck at 0

### 13.4 Leaky ReLU

$$\text{LReLU}(x) = \begin{cases} x & x > 0 \\ \alpha x & x \leq 0 \end{cases}, \quad \alpha \approx 0.01$$

$$\text{LReLU}'(x) = \begin{cases} 1 & x > 0 \\ \alpha & x \leq 0 \end{cases}$$

Fixes dying ReLU.

### 13.5 ELU (Exponential Linear Unit)

$$\text{ELU}(x) = \begin{cases} x & x > 0 \\ \alpha(e^x - 1) & x \leq 0 \end{cases}$$

$$\text{ELU}'(x) = \begin{cases} 1 & x > 0 \\ \text{ELU}(x) + \alpha & x \leq 0 \end{cases}$$

### 13.6 GELU (Gaussian Error Linear Unit)

Used in BERT, GPT:

$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]$$

**Approximation:**

$$\text{GELU}(x) \approx 0.5x\left[1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right]$$

$$\text{GELU}'(x) = \Phi(x) + x\phi(x)$$

where $\phi$ is the Gaussian PDF.

### 13.7 Softmax

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}$$

**Jacobian:**

$$\frac{\partial \text{softmax}(\mathbf{z})_i}{\partial z_j} = \text{softmax}(\mathbf{z})_i \left(\delta_{ij} - \text{softmax}(\mathbf{z})_j\right)$$

**Numerical stability trick:**

$$\text{softmax}(\mathbf{z})_i = \frac{e^{z_i - z_{\max}}}{\sum_j e^{z_j - z_{\max}}}$$

Subtracting $z_{\max}$ doesn't change the result but prevents overflow.

---

## 14. Loss Functions & Their Gradients

### 14.1 Mean Squared Error (MSE)

$$\mathcal{L}_{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{-2}{n}(y_i - \hat{y}_i)$$

**Use:** Regression

### 14.2 Mean Absolute Error (MAE)

$$\mathcal{L}_{MAE} = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = \frac{-1}{n}\text{sign}(y_i - \hat{y}_i)$$

**Property:** More robust to outliers than MSE but not differentiable at 0.

### 14.3 Huber Loss

Combines MSE and MAE:

$$\mathcal{L}_\delta(\hat{y}, y) = \begin{cases} \frac{1}{2}(y - \hat{y})^2 & |y - \hat{y}| \leq \delta \\ \delta\left(|y - \hat{y}| - \frac{\delta}{2}\right) & |y - \hat{y}| > \delta \end{cases}$$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}} = \begin{cases} -(y - \hat{y}) & |y - \hat{y}| \leq \delta \\ -\delta \cdot \text{sign}(y - \hat{y}) & |y - \hat{y}| > \delta \end{cases}$$

### 14.4 Binary Cross-Entropy (BCE)

$$\mathcal{L}_{BCE} = -\frac{1}{n}\sum_{i=1}^{n}\left[y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)\right]$$

$$\frac{\partial \mathcal{L}}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i} + \frac{1-y_i}{1-\hat{y}_i}$$

**With sigmoid output** $\hat{y} = \sigma(z)$, combined gradient simplifies beautifully:

$$\frac{\partial \mathcal{L}}{\partial z} = \hat{y} - y$$

### 14.5 Categorical Cross-Entropy (CCE)

$$\mathcal{L}_{CCE} = -\sum_{k=1}^{K} y_k \log \hat{y}_k = -\log \hat{y}_{c}$$

where $c$ is the true class.

**With softmax:** $\hat{\mathbf{y}} = \text{softmax}(\mathbf{z})$

Combined gradient (very elegant):

$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$

This simplicity is why softmax + cross-entropy is the standard combination.

### 14.6 KL Divergence Loss (VAE)

$$\mathcal{L}_{KL} = D_{KL}(q(z|x) \| p(z)) = -\frac{1}{2}\sum_{j=1}^{d}\left(1 + \log \sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

$$\frac{\partial \mathcal{L}_{KL}}{\partial \mu_j} = \mu_j, \quad \frac{\partial \mathcal{L}_{KL}}{\partial \sigma_j^2} = \frac{1}{2}\left(\sigma_j^2 - \frac{1}{\sigma_j^2}\right) \cdot \sigma_j^2$$

### 14.7 Contrastive Loss (SimCLR, etc.)

$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k=1}^{2N} \mathbf{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k)/\tau)}$$

where $\text{sim}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u}^\top \mathbf{v}}{\|\mathbf{u}\|\|\mathbf{v}\|}$ (cosine similarity) and $\tau$ is temperature.

### 14.8 Regularization Terms

**L2 (Ridge):** Penalizes large weights

$$\mathcal{L}_{reg} = \mathcal{L} + \frac{\lambda}{2}\|\mathbf{w}\|^2, \quad \nabla_\mathbf{w} \mathcal{L}_{reg} = \nabla_\mathbf{w} \mathcal{L} + \lambda \mathbf{w}$$

**L1 (Lasso):** Induces sparsity

$$\mathcal{L}_{reg} = \mathcal{L} + \lambda\|\mathbf{w}\|_1, \quad \nabla_\mathbf{w} \mathcal{L}_{reg} = \nabla_\mathbf{w} \mathcal{L} + \lambda \cdot \text{sign}(\mathbf{w})$$

---

## 15. Advanced Topics: Lagrange Multipliers, KKT

### 15.1 Constrained Optimization

**Equality constrained problem:**

$$\min_{\mathbf{x}} f(\mathbf{x}) \quad \text{subject to } g(\mathbf{x}) = 0$$

### 15.2 Lagrange Multipliers

Introduce multiplier $\lambda$ and form the **Lagrangian**:

$$\mathcal{L}(\mathbf{x}, \lambda) = f(\mathbf{x}) + \lambda g(\mathbf{x})$$

**Necessary conditions** (KKT for equality):

$$\nabla_\mathbf{x} \mathcal{L} = \nabla f(\mathbf{x}) + \lambda \nabla g(\mathbf{x}) = \mathbf{0}$$

$$g(\mathbf{x}) = 0$$

**Geometric interpretation:** At the optimum, $\nabla f$ and $\nabla g$ are parallel.

### 15.3 KKT Conditions (Inequality Constraints)

**Problem:**

$$\min_\mathbf{x} f(\mathbf{x}) \quad \text{s.t. } g_i(\mathbf{x}) \leq 0,\ h_j(\mathbf{x}) = 0$$

**Lagrangian:**

$$\mathcal{L}(\mathbf{x}, \boldsymbol{\mu}, \boldsymbol{\lambda}) = f(\mathbf{x}) + \sum_i \mu_i g_i(\mathbf{x}) + \sum_j \lambda_j h_j(\mathbf{x})$$

**KKT conditions:**

1. **Stationarity:** $\nabla_\mathbf{x} \mathcal{L} = \mathbf{0}$
2. **Primal feasibility:** $g_i(\mathbf{x}) \leq 0$, $h_j(\mathbf{x}) = 0$
3. **Dual feasibility:** $\mu_i \geq 0$
4. **Complementary slackness:** $\mu_i g_i(\mathbf{x}) = 0$

### 15.4 ML Application: SVM

SVM primal problem:

$$\min_{w, b} \frac{1}{2}\|w\|^2 \quad \text{s.t. } y_i(\mathbf{w}^\top \mathbf{x}_i + b) \geq 1$$

Lagrangian:

$$\mathcal{L} = \frac{1}{2}\|\mathbf{w}\|^2 - \sum_i \alpha_i [y_i(\mathbf{w}^\top \mathbf{x}_i + b) - 1]$$

KKT conditions give the **dual problem:**

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_{i,j} \alpha_i \alpha_j y_i y_j \mathbf{x}_i^\top \mathbf{x}_j$$

$$\text{s.t. } \alpha_i \geq 0,\ \sum_i \alpha_i y_i = 0$$

Solution: $\mathbf{w} = \sum_i \alpha_i y_i \mathbf{x}_i$ (only support vectors have $\alpha_i > 0$)

---

## 16. Calculus in Attention & Transformers

### 16.1 Scaled Dot-Product Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V$$

where $Q, K, V \in \mathbb{R}^{n \times d_k}$.

**Why divide by $\sqrt{d_k}$?**

For random vectors $q, k \in \mathbb{R}^{d_k}$ with components $\sim \mathcal{N}(0,1)$:

$$q \cdot k = \sum_{i=1}^{d_k} q_i k_i, \quad \text{Var}\left(\sum_i q_i k_i\right) = d_k$$

Dividing by $\sqrt{d_k}$ gives unit variance → prevents softmax saturation.

### 16.2 Gradient of Attention

Let $A = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)$, output $O = AV$.

Given $\frac{\partial \mathcal{L}}{\partial O}$:

$$\frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial O}$$

$$\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial O} V^\top$$

For softmax Jacobian, let $S = \frac{\partial \mathcal{L}}{\partial A}$:

$$\frac{\partial \mathcal{L}}{\partial \text{scores}} = A \odot \left(S - (A \odot S)\mathbf{1}\mathbf{1}^\top\right) \cdot \frac{1}{\sqrt{d_k}}$$

Then:

$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\partial \mathcal{L}}{\partial \text{scores}} \cdot K, \quad \frac{\partial \mathcal{L}}{\partial K} = \left(\frac{\partial \mathcal{L}}{\partial \text{scores}}\right)^\top Q$$

### 16.3 Layer Normalization

$$\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sigma} + \beta$$

where $\mu = \frac{1}{d}\sum_i x_i$, $\sigma^2 = \frac{1}{d}\sum_i (x_i - \mu)^2$

**Gradient** (using chain rule through normalization):

$$\frac{\partial \mathcal{L}}{\partial x_i} = \frac{\gamma_i}{\sigma}\left(\frac{\partial \mathcal{L}}{\partial \hat{x}_i} - \frac{1}{d}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} - \hat{x}_i \cdot \frac{1}{d}\sum_j \frac{\partial \mathcal{L}}{\partial \hat{x}_j} \hat{x}_j\right)$$

### 16.4 Positional Encoding

Sinusoidal PE (not learned — no gradient):

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$

$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

These are fixed — their derivatives are never computed.

---

## 17. NumPy & Calculus Connections

> **This section shows how calculus concepts map to NumPy operations — pure theory.**

### 17.1 Numerical Differentiation

**Forward difference** (approximates derivative):

$$f'(x) \approx \frac{f(x+h) - f(x)}{h}, \quad \text{error: } O(h)$$

**Central difference** (more accurate):

$$f'(x) \approx \frac{f(x+h) - f(x-h)}{2h}, \quad \text{error: } O(h^2)$$

**Second derivative (central):**

$$f''(x) \approx \frac{f(x+h) - 2f(x) + f(x-h)}{h^2}$$

**Gradient check:** Compare analytical gradient vs numerical gradient:

$$\text{relative error} = \frac{\|\nabla_{\text{analytic}} - \nabla_{\text{numeric}}\|}{\|\nabla_{\text{analytic}}\| + \|\nabla_{\text{numeric}}\|} < 10^{-5}$$

### 17.2 Dot Products → Inner Products → Projections

$$\mathbf{a} \cdot \mathbf{b} = \mathbf{a}^\top \mathbf{b} = \sum_{i} a_i b_i = \|\mathbf{a}\|\|\mathbf{b}\|\cos\theta$$

In NumPy: `np.dot(a, b)` or `a @ b`

**Projection of a onto b:**

$$\text{proj}_\mathbf{b} \mathbf{a} = \frac{\mathbf{a}^\top \mathbf{b}}{\|\mathbf{b}\|^2} \mathbf{b}$$

### 17.3 Matrix Multiplication → Linear Transformations

$$C = AB, \quad C_{ij} = \sum_k A_{ik} B_{kj}$$

This is the composition of two linear maps. In NumPy: `A @ B`

**Complexity:** $O(n^3)$ for $n \times n$ matrices.

### 17.4 Eigenvalues → Spectral Analysis of Hessian

For symmetric matrix $A$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

**Spectral decomposition:**

$$A = Q \Lambda Q^\top = \sum_i \lambda_i \mathbf{q}_i \mathbf{q}_i^\top$$

The eigenvalues of the Hessian determine:
- Condition number $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}}$ → convergence speed of gradient descent
- Positive definite iff all $\lambda_i > 0$

### 17.5 Singular Value Decomposition (SVD)

For any matrix $A \in \mathbb{R}^{m \times n}$:

$$A = U \Sigma V^\top$$

where $U \in \mathbb{R}^{m \times m}$, $\Sigma \in \mathbb{R}^{m \times n}$ (diagonal), $V \in \mathbb{R}^{n \times n}$

**ML Applications:**
- PCA: directions of maximum variance are right singular vectors of centered data matrix
- Low-rank approximation: $A_k = U_k \Sigma_k V_k^\top$ (best rank-k approximation)
- Gradient clipping: clip by singular values

### 17.6 Broadcasting & Vectorized Gradients

In NumPy, operations broadcast across dimensions. When computing gradients through broadcast operations:

**If** $y = f(x)$ where x is broadcast: gradient must be **summed** over broadcast dimensions.

**Example:** $\mathcal{L} = \sum_{ij} (X_{ij} - \mathbf{b}_j)^2$ where $\mathbf{b}$ broadcasts over rows:

$$\frac{\partial \mathcal{L}}{\partial b_j} = \sum_i \frac{\partial \mathcal{L}}{\partial X_{ij}} \cdot (-1) = -2\sum_i (X_{ij} - b_j)$$

The gradient is summed over the broadcast dimension.

### 17.7 Convolution (in CNNs)

2D discrete convolution:

$$(f * g)[m, n] = \sum_{k}\sum_{l} f[k, l] \cdot g[m-k, n-l]$$

**Gradient of convolution:**
- Gradient w.r.t. input = **full convolution** with flipped kernel
- Gradient w.r.t. kernel = **valid cross-correlation** of input with output gradient

$$\frac{\partial \mathcal{L}}{\partial W} = X * \frac{\partial \mathcal{L}}{\partial O} \quad \text{(cross-correlation)}$$

$$\frac{\partial \mathcal{L}}{\partial X} = W_{flip} * \frac{\partial \mathcal{L}}{\partial O} \quad \text{(full convolution)}$$

---

## 🔑 Quick Reference: Most Important Formulas

### Derivatives

$$\frac{d}{dx}[\sigma(x)] = \sigma(x)(1-\sigma(x))$$

$$\frac{d}{dx}[\tanh(x)] = 1 - \tanh^2(x)$$

$$\frac{d}{dx}[\text{ReLU}(x)] = \mathbf{1}[x > 0]$$

### Gradient Descent

$$\mathbf{w} \leftarrow \mathbf{w} - \eta \nabla_\mathbf{w} \mathcal{L}$$

### Adam

$$\hat{m}_t = \frac{\beta_1 m_{t-1} + (1-\beta_1) g_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{\beta_2 v_{t-1} + (1-\beta_2) g_t^2}{1-\beta_2^t}$$

$$w \leftarrow w - \frac{\eta \hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$$

### Backpropagation

$$\delta^{(L)} = \nabla_{a^{(L)}} \mathcal{L} \odot \sigma'(z^{(L)})$$

$$\delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot \sigma'(z^{(l)})$$

### Cross-Entropy + Softmax (combined gradient)

$$\frac{\partial \mathcal{L}}{\partial z_k} = \hat{y}_k - y_k$$

### Normal Equations

$$\mathbf{w}^* = (X^\top X)^{-1} X^\top \mathbf{y}$$

### KL Divergence (VAE)

$$D_{KL}(\mathcal{N}(\mu, \sigma^2) \| \mathcal{N}(0,1)) = -\frac{1}{2}(1 + \log\sigma^2 - \mu^2 - \sigma^2)$$

### Attention

$$\text{Attention}(Q,K,V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V$$

---

## 📖 Study Path Recommendations

| Stage | Topics | Goal |
|-------|--------|------|
| **Week 1–2** | §1–3: Limits, derivatives, partial derivatives | Understand what a derivative is geometrically and computationally |
| **Week 3–4** | §4–5: Chain rule, gradients | Understand why backprop works |
| **Week 5–6** | §6–7: Jacobians, Hessians, Taylor series | Understand curvature, optimization landscape |
| **Week 7–8** | §8–9: Optimization theory, gradient descent variants | Understand Adam, momentum, learning rates |
| **Week 9–10** | §10–11: Integrals, matrix calculus | Connect to probability, linear algebra |
| **Week 11–12** | §12–14: Full backprop, activations, losses | Implement from scratch |
| **Week 13–14** | §15–17: Advanced topics, attention, NumPy | Understand modern architectures |

---

## 💡 Key Intuitions Summary

1. **Derivative = sensitivity.** How much does the output wiggle when you wiggle the input?

2. **Gradient = direction of steepest ascent.** Go opposite for descent.

3. **Chain rule = backpropagation.** The entire training of neural networks is the chain rule applied recursively.

4. **Taylor expansion = why gradient descent works.** Near a point, the function looks linear. Moving against the gradient decreases it.

5. **Hessian = curvature.** Tells you if you're near a minimum, maximum, or saddle point. Too expensive to compute for large networks.

6. **Softmax + cross-entropy = $\hat{y} - y$.** This elegant cancellation is why this combination is universal.

7. **Convexity = guarantee.** Convex functions have one global minimum. Deep networks are not convex, but good initializations + good optimizers get us there in practice.

8. **Integration = expectations.** The loss you minimize is an empirical approximation of an integral (expected loss over the data distribution).

---

*These notes cover Calculus from first principles through modern transformer architectures. Master these formulas and derivations to build intuition for every component of ML/DL systems.*
