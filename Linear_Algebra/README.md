# 📐 Linear Algebra for Machine Learning & Deep Learning
### Complete Notes — From Basics to Advanced

> **"The language of data is Linear Algebra."**
> These notes cover every concept needed for ML/DL — pure theory, rich formulas, and intuition-first explanations.

---

## 📋 Table of Contents

1. [Scalars, Vectors, Matrices & Tensors](#1-scalars-vectors-matrices--tensors)
2. [Vector Operations](#2-vector-operations)
3. [Matrix Operations](#3-matrix-operations)
4. [Special Matrices](#4-special-matrices)
5. [Systems of Linear Equations](#5-systems-of-linear-equations)
6. [Vector Spaces & Subspaces](#6-vector-spaces--subspaces)
7. [Linear Independence, Basis & Rank](#7-linear-independence-basis--rank)
8. [Norms & Distance Metrics](#8-norms--distance-metrics)
9. [Projections](#9-projections)
10. [Determinants](#10-determinants)
11. [Eigenvalues & Eigenvectors](#11-eigenvalues--eigenvectors)
12. [Matrix Decompositions](#12-matrix-decompositions)
13. [Singular Value Decomposition (SVD)](#13-singular-value-decomposition-svd)
14. [Principal Component Analysis (PCA)](#14-principal-component-analysis-pca)
15. [Positive Definite Matrices](#15-positive-definite-matrices)
16. [Matrix Calculus](#16-matrix-calculus)
17. [Tensors (for Deep Learning)](#17-tensors-for-deep-learning)
18. [Linear Transformations](#18-linear-transformations)
19. [Dimensionality Reduction](#19-dimensionality-reduction)
20. [Applications in ML/DL](#20-applications-in-mldl)

---

## 1. Scalars, Vectors, Matrices & Tensors

### 1.1 Scalar

A **scalar** is a single real number.

$$x \in \mathbb{R}$$

**Example (ML context):** Learning rate $\alpha = 0.01$, regularization term $\lambda = 0.001$, a single pixel value $x = 255$.

---

### 1.2 Vector

A **vector** is an ordered list of scalars — a 1D array of numbers.

**Column vector:**
$$\mathbf{x} = \begin{bmatrix} x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix} \in \mathbb{R}^n$$

**Row vector:**
$$\mathbf{x}^T = \begin{bmatrix} x_1 & x_2 & \cdots & x_n \end{bmatrix}$$

**Example (ML context):** A data point with 3 features (height, weight, age):
$$\mathbf{x} = \begin{bmatrix} 175 \\ 70 \\ 25 \end{bmatrix}$$

---

### 1.3 Matrix

A **matrix** is a 2D array of scalars with $m$ rows and $n$ columns.

$$A \in \mathbb{R}^{m \times n}, \quad A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

Element at row $i$, column $j$: $A_{ij}$ or $a_{ij}$

**Example (ML context):** Dataset with 4 samples, 3 features each:
$$X = \begin{bmatrix} 175 & 70 & 25 \\ 160 & 55 & 30 \\ 180 & 85 & 22 \\ 165 & 60 & 28 \end{bmatrix} \in \mathbb{R}^{4 \times 3}$$

---

### 1.4 Tensor

A **tensor** is a generalization to $N$ dimensions (rank-$N$ array).

| Type | Rank | Shape Example |
|------|------|---------------|
| Scalar | 0 | `()` |
| Vector | 1 | `(n,)` |
| Matrix | 2 | `(m, n)` |
| 3D Tensor | 3 | `(batch, height, width)` |
| 4D Tensor | 4 | `(batch, channels, height, width)` |

**Example (Deep Learning):** A batch of 32 color images, 64×64 pixels:
$$\mathcal{T} \in \mathbb{R}^{32 \times 3 \times 64 \times 64}$$

---

## 2. Vector Operations

### 2.1 Addition & Subtraction

For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:

$$\mathbf{u} + \mathbf{v} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \\ \vdots \\ u_n + v_n \end{bmatrix}$$

**Properties:**
- Commutativity: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$
- Associativity: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$
- Zero vector: $\mathbf{u} + \mathbf{0} = \mathbf{u}$

---

### 2.2 Scalar Multiplication

$$c \cdot \mathbf{v} = \begin{bmatrix} c \cdot v_1 \\ c \cdot v_2 \\ \vdots \\ c \cdot v_n \end{bmatrix}$$

**Example (ML):** Gradient descent update: $\mathbf{w} \leftarrow \mathbf{w} - \alpha \cdot \nabla_\mathbf{w} L$, where $\alpha \cdot \nabla_\mathbf{w} L$ is scalar-vector multiplication.

---

### 2.3 Dot Product (Inner Product)

$$\mathbf{u} \cdot \mathbf{v} = \mathbf{u}^T \mathbf{v} = \sum_{i=1}^{n} u_i v_i \in \mathbb{R}$$

**Geometric interpretation:**
$$\mathbf{u} \cdot \mathbf{v} = \|\mathbf{u}\| \|\mathbf{v}\| \cos\theta$$

where $\theta$ is the angle between vectors.

| Dot Product Value | Meaning |
|-------------------|---------|
| $\mathbf{u} \cdot \mathbf{v} > 0$ | Vectors point in similar directions |
| $\mathbf{u} \cdot \mathbf{v} = 0$ | Vectors are **orthogonal** (perpendicular) |
| $\mathbf{u} \cdot \mathbf{v} < 0$ | Vectors point in opposite directions |

**Example (ML):** Linear prediction: $\hat{y} = \mathbf{w}^T \mathbf{x} = \sum_{i=1}^{n} w_i x_i$

---

### 2.4 Outer Product

$$\mathbf{u} \otimes \mathbf{v} = \mathbf{u} \mathbf{v}^T = \begin{bmatrix} u_1 v_1 & u_1 v_2 & \cdots \\ u_2 v_1 & u_2 v_2 & \cdots \\ \vdots & \vdots & \ddots \end{bmatrix} \in \mathbb{R}^{m \times n}$$

**Example (Deep Learning):** Gradient of weight matrix $W$ in backpropagation: $\frac{\partial L}{\partial W} = \boldsymbol{\delta} \mathbf{x}^T$, where $\boldsymbol{\delta}$ is error signal and $\mathbf{x}$ is input — this is an outer product.

---

### 2.5 Element-wise (Hadamard) Product

$$\mathbf{u} \odot \mathbf{v} = \begin{bmatrix} u_1 v_1 \\ u_2 v_2 \\ \vdots \\ u_n v_n \end{bmatrix}$$

**Example (DL):** Dropout mask: $\tilde{\mathbf{h}} = \mathbf{h} \odot \mathbf{m}$ where $\mathbf{m} \in \{0,1\}^n$ is binary mask.

---

### 2.6 Cross Product (3D only)

For $\mathbf{u}, \mathbf{v} \in \mathbb{R}^3$:

$$\mathbf{u} \times \mathbf{v} = \begin{vmatrix} \mathbf{i} & \mathbf{j} & \mathbf{k} \\ u_1 & u_2 & u_3 \\ v_1 & v_2 & v_3 \end{vmatrix} = \begin{bmatrix} u_2 v_3 - u_3 v_2 \\ u_3 v_1 - u_1 v_3 \\ u_1 v_2 - u_2 v_1 \end{bmatrix}$$

Result is a vector perpendicular to both $\mathbf{u}$ and $\mathbf{v}$.

---

## 3. Matrix Operations

### 3.1 Matrix Addition

$$C = A + B \implies C_{ij} = A_{ij} + B_{ij}$$

Requires $A, B \in \mathbb{R}^{m \times n}$ (same shape).

---

### 3.2 Scalar-Matrix Multiplication

$$C = cA \implies C_{ij} = c \cdot A_{ij}$$

---

### 3.3 Matrix-Vector Multiplication

$$A\mathbf{x} = \mathbf{b}, \quad A \in \mathbb{R}^{m \times n},\ \mathbf{x} \in \mathbb{R}^n,\ \mathbf{b} \in \mathbb{R}^m$$

$$b_i = \sum_{j=1}^{n} A_{ij} x_j \quad \text{(row-vector dot product view)}$$

**Column space view:** $A\mathbf{x}$ is a **linear combination of columns** of $A$:

$$A\mathbf{x} = x_1 \mathbf{a}_1 + x_2 \mathbf{a}_2 + \cdots + x_n \mathbf{a}_n$$

**Example (ML):** Forward pass of linear layer:
$$\mathbf{z} = W\mathbf{x} + \mathbf{b}$$

---

### 3.4 Matrix-Matrix Multiplication

$$C = AB, \quad A \in \mathbb{R}^{m \times k},\ B \in \mathbb{R}^{k \times n},\ C \in \mathbb{R}^{m \times n}$$

$$C_{ij} = \sum_{l=1}^{k} A_{il} B_{lj} = \mathbf{a}_i^T \mathbf{b}_j$$

**Properties:**
- Associativity: $(AB)C = A(BC)$
- Distributivity: $A(B+C) = AB + AC$
- **Not commutative:** $AB \neq BA$ in general
- Transpose: $(AB)^T = B^T A^T$

**Computational cost:** $O(mnk)$ — this dominates DL training time.

**Example (DL):** Multi-layer forward pass:
$$\mathbf{y} = W_2 \sigma(W_1 \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2$$

---

### 3.5 Matrix Transpose

$$(A^T)_{ij} = A_{ji}$$

**Properties:**
- $(A^T)^T = A$
- $(A + B)^T = A^T + B^T$
- $(AB)^T = B^T A^T$
- $(ABC)^T = C^T B^T A^T$

---

### 3.6 Matrix Inverse

For square $A \in \mathbb{R}^{n \times n}$, the inverse $A^{-1}$ satisfies:

$$A A^{-1} = A^{-1} A = I$$

**Conditions for invertibility:**
- $\det(A) \neq 0$
- All eigenvalues $\neq 0$
- Rows (and columns) are linearly independent
- $\text{rank}(A) = n$

**Properties:**
- $(A^{-1})^{-1} = A$
- $(AB)^{-1} = B^{-1} A^{-1}$
- $(A^T)^{-1} = (A^{-1})^T$
- $(cA)^{-1} = \frac{1}{c} A^{-1}$

**2×2 inverse formula:**
$$A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \implies A^{-1} = \frac{1}{ad - bc} \begin{bmatrix} d & -b \\ -c & a \end{bmatrix}$$

**Example (ML):** Normal equations for Linear Regression:
$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

---

### 3.7 Trace

$$\text{tr}(A) = \sum_{i=1}^{n} A_{ii} = \sum_{i=1}^{n} \lambda_i$$

**Properties:**
- $\text{tr}(A + B) = \text{tr}(A) + \text{tr}(B)$
- $\text{tr}(cA) = c\,\text{tr}(A)$
- $\text{tr}(AB) = \text{tr}(BA)$ *(cyclic property)*
- $\text{tr}(ABC) = \text{tr}(BCA) = \text{tr}(CAB)$
- $\text{tr}(A^T A) = \sum_{i,j} A_{ij}^2 = \|A\|_F^2$

**Example (ML):** The Frobenius norm squared $\|W\|_F^2 = \text{tr}(W^T W)$ appears in L2 regularization.

---

## 4. Special Matrices

### 4.1 Identity Matrix

$$I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

$$AI = IA = A, \quad I\mathbf{x} = \mathbf{x}$$

---

### 4.2 Zero Matrix

$$O_{ij} = 0 \quad \forall i, j$$

$$A + O = A, \quad AO = OA = O$$

---

### 4.3 Diagonal Matrix

$$D = \text{diag}(d_1, d_2, \ldots, d_n) = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{bmatrix}$$

$$D^{-1} = \text{diag}(d_1^{-1}, \ldots, d_n^{-1}), \quad \det(D) = \prod_{i} d_i$$

---

### 4.4 Symmetric Matrix

$$A = A^T \iff A_{ij} = A_{ji}$$

**Key property:** All eigenvalues of a real symmetric matrix are **real**, and eigenvectors are **orthogonal**.

**Example (ML):** Covariance matrix $\Sigma = \frac{1}{n} X^T X$ is symmetric. Gram matrix $K = XX^T$ is symmetric.

---

### 4.5 Orthogonal Matrix

$$Q^T Q = Q Q^T = I \implies Q^{-1} = Q^T$$

**Properties:**
- Columns are orthonormal: $\mathbf{q}_i^T \mathbf{q}_j = \delta_{ij}$
- Preserves lengths: $\|Q\mathbf{x}\| = \|\mathbf{x}\|$
- Preserves angles: $(Q\mathbf{u})^T(Q\mathbf{v}) = \mathbf{u}^T \mathbf{v}$
- $\det(Q) = \pm 1$

**Example (ML):** PCA rotation matrix $Q$ is orthogonal — it rotates the coordinate system without distortion.

---

### 4.6 Triangular Matrices

**Upper triangular:**
$$U_{ij} = 0 \quad \text{if } i > j$$

**Lower triangular:**
$$L_{ij} = 0 \quad \text{if } i < j$$

**Key use:** LU decomposition: $A = LU$, used for efficient solving of $A\mathbf{x} = \mathbf{b}$.

---

### 4.7 Positive Semidefinite (PSD) Matrix

$$A \succeq 0 \iff \mathbf{x}^T A \mathbf{x} \geq 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

**Positive Definite (PD):**
$$A \succ 0 \iff \mathbf{x}^T A \mathbf{x} > 0 \quad \forall \mathbf{x} \neq \mathbf{0}$$

**Eigenvalue characterization:**
- $A \succeq 0 \iff$ all eigenvalues $\lambda_i \geq 0$
- $A \succ 0 \iff$ all eigenvalues $\lambda_i > 0$

**Example (ML):** Covariance matrices are always PSD. Hessian of a convex loss function is PSD.

---

## 5. Systems of Linear Equations

### 5.1 General Form

$$A\mathbf{x} = \mathbf{b}$$

$$\begin{aligned} a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n &= b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n &= b_2 \\ &\vdots \\ a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n &= b_m \end{aligned}$$

### 5.2 Solution Types

| Case | Condition | Solutions |
|------|-----------|-----------|
| Unique solution | $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ | Exactly one |
| No solution | $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ | Zero (inconsistent) |
| Infinite solutions | $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ | Infinitely many |

### 5.3 Solving with Gaussian Elimination

Transform $[A|\mathbf{b}]$ to row echelon form using elementary row operations:
1. Swap two rows
2. Multiply a row by scalar $c \neq 0$
3. Add multiple of one row to another

### 5.4 Least Squares Solution

When $A\mathbf{x} = \mathbf{b}$ is overdetermined ($m > n$), the least squares solution minimizes:

$$\min_\mathbf{x} \|A\mathbf{x} - \mathbf{b}\|_2^2$$

**Normal equations:**

$$A^T A \hat{\mathbf{x}} = A^T \mathbf{b}$$

$$\hat{\mathbf{x}} = (A^T A)^{-1} A^T \mathbf{b}$$

**Example (ML):** This is exactly the closed-form solution of Linear Regression where $A = X$ (design matrix), $\mathbf{b} = \mathbf{y}$ (targets).

---

## 6. Vector Spaces & Subspaces

### 6.1 Vector Space

A set $V$ with addition and scalar multiplication satisfying 8 axioms (closure, associativity, commutativity, identity, inverses, distributivity).

**Common vector spaces in ML:**
- $\mathbb{R}^n$ — feature vectors
- $\mathbb{R}^{m \times n}$ — weight matrices
- Space of polynomials, continuous functions

### 6.2 Subspace

$S \subseteq V$ is a subspace if:
1. $\mathbf{0} \in S$
2. $\mathbf{u}, \mathbf{v} \in S \implies \mathbf{u} + \mathbf{v} \in S$ (closed under addition)
3. $\mathbf{u} \in S, c \in \mathbb{R} \implies c\mathbf{u} \in S$ (closed under scalar multiplication)

### 6.3 Four Fundamental Subspaces of Matrix $A \in \mathbb{R}^{m \times n}$

| Subspace | Definition | Dimension |
|----------|------------|-----------|
| Column space $\mathcal{C}(A)$ | All $A\mathbf{x}$ for $\mathbf{x} \in \mathbb{R}^n$ | $r = \text{rank}(A)$ |
| Null space $\mathcal{N}(A)$ | All $\mathbf{x}$ s.t. $A\mathbf{x} = \mathbf{0}$ | $n - r$ |
| Row space $\mathcal{C}(A^T)$ | All $A^T\mathbf{y}$ for $\mathbf{y} \in \mathbb{R}^m$ | $r$ |
| Left null space $\mathcal{N}(A^T)$ | All $\mathbf{y}$ s.t. $A^T\mathbf{y} = \mathbf{0}$ | $m - r$ |

**Orthogonality relations:**
$$\mathcal{C}(A) \perp \mathcal{N}(A^T), \qquad \mathcal{C}(A^T) \perp \mathcal{N}(A)$$

---

### 6.4 Span

The **span** of vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$:

$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \left\{ \sum_{i=1}^k c_i \mathbf{v}_i \;\middle|\; c_i \in \mathbb{R} \right\}$$

---

## 7. Linear Independence, Basis & Rank

### 7.1 Linear Independence

Vectors $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ are **linearly independent** if:

$$c_1 \mathbf{v}_1 + c_2 \mathbf{v}_2 + \cdots + c_k \mathbf{v}_k = \mathbf{0} \implies c_1 = c_2 = \cdots = c_k = 0$$

Otherwise **linearly dependent** (at least one vector is a linear combination of others).

**Example:**
$$\mathbf{v}_1 = \begin{bmatrix}1\\0\end{bmatrix},\ \mathbf{v}_2 = \begin{bmatrix}0\\1\end{bmatrix} \quad \text{— linearly independent}$$

$$\mathbf{v}_1 = \begin{bmatrix}1\\2\end{bmatrix},\ \mathbf{v}_2 = \begin{bmatrix}2\\4\end{bmatrix} \quad \text{— linearly dependent (} \mathbf{v}_2 = 2\mathbf{v}_1\text{)}$$

---

### 7.2 Basis

A **basis** of a vector space $V$ is a set of vectors that is:
1. Linearly independent
2. Spans $V$

**Standard basis** for $\mathbb{R}^n$: $\{\mathbf{e}_1, \mathbf{e}_2, \ldots, \mathbf{e}_n\}$

$$\mathbf{e}_1 = \begin{bmatrix}1\\0\\\vdots\\0\end{bmatrix},\quad \mathbf{e}_2 = \begin{bmatrix}0\\1\\\vdots\\0\end{bmatrix}, \quad \ldots$$

**Change of basis:** If $B = [\mathbf{b}_1, \ldots, \mathbf{b}_n]$ is a new basis matrix:
$$[\mathbf{x}]_B = B^{-1} \mathbf{x} \quad \text{(coordinates in new basis)}$$

---

### 7.3 Rank

$$\text{rank}(A) = \dim(\mathcal{C}(A)) = \dim(\mathcal{C}(A^T))$$

The number of linearly independent rows = number of linearly independent columns.

**Rank-Nullity Theorem:**
$$\text{rank}(A) + \text{nullity}(A) = n$$

where $\text{nullity}(A) = \dim(\mathcal{N}(A))$.

**Low-rank approximation (ML):** Many weight matrices in DL have approximately low rank, enabling compression via SVD.

---

## 8. Norms & Distance Metrics

### 8.1 Vector Norms

A norm $\|\cdot\|: \mathbb{R}^n \to \mathbb{R}$ satisfies:
1. $\|\mathbf{x}\| \geq 0$, and $\|\mathbf{x}\| = 0 \iff \mathbf{x} = \mathbf{0}$
2. $\|c\mathbf{x}\| = |c| \|\mathbf{x}\|$
3. $\|\mathbf{x} + \mathbf{y}\| \leq \|\mathbf{x}\| + \|\mathbf{y}\|$ (triangle inequality)

**$L^p$ Norm:**
$$\|\mathbf{x}\|_p = \left( \sum_{i=1}^{n} |x_i|^p \right)^{1/p}$$

| Norm | Formula | ML Use |
|------|---------|--------|
| $L^1$ (Manhattan) | $\|\mathbf{x}\|_1 = \sum_i |x_i|$ | Lasso regularization, sparsity |
| $L^2$ (Euclidean) | $\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$ | Ridge regularization, distances |
| $L^\infty$ (Max) | $\|\mathbf{x}\|_\infty = \max_i |x_i|$ | Robustness, clipping |
| $L^0$ (pseudo-norm) | $\|\mathbf{x}\|_0 = \text{nnz}(\mathbf{x})$ | Sparsity counting |

**Squared $L^2$ norm:**
$$\|\mathbf{x}\|_2^2 = \mathbf{x}^T \mathbf{x} = \sum_{i=1}^n x_i^2$$

---

### 8.2 Matrix Norms

**Frobenius Norm:**
$$\|A\|_F = \sqrt{\sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij}^2} = \sqrt{\text{tr}(A^T A)}$$

**Spectral Norm (induced $L^2$ norm):**
$$\|A\|_2 = \sigma_{\max}(A) = \sqrt{\lambda_{\max}(A^T A)}$$

where $\sigma_{\max}$ is the largest singular value.

**Nuclear Norm (trace norm):**
$$\|A\|_* = \sum_i \sigma_i(A) = \text{tr}(\sqrt{A^T A})$$

Used in low-rank matrix completion, collaborative filtering.

---

### 8.3 Distance Metrics

**Euclidean distance:**
$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$

**Cosine similarity:**
$$\cos\theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} \in [-1, 1]$$

Cosine **distance:** $d_{\cos}(\mathbf{u}, \mathbf{v}) = 1 - \cos\theta$

**Mahalanobis distance:**
$$d_M(\mathbf{u}, \mathbf{v}) = \sqrt{(\mathbf{u} - \mathbf{v})^T \Sigma^{-1} (\mathbf{u} - \mathbf{v})}$$

Accounts for correlations between features. Used in anomaly detection, Gaussian classifiers.

---

### 8.4 Cauchy–Schwarz Inequality

$$|\mathbf{u}^T \mathbf{v}| \leq \|\mathbf{u}\|_2 \|\mathbf{v}\|_2$$

Equality iff $\mathbf{u} = c\mathbf{v}$ for some scalar $c$.

---

## 9. Projections

### 9.1 Projection onto a Vector

Projection of $\mathbf{b}$ onto $\mathbf{a}$:

$$\text{proj}_{\mathbf{a}} \mathbf{b} = \frac{\mathbf{a}^T \mathbf{b}}{\mathbf{a}^T \mathbf{a}} \mathbf{a} = \frac{\mathbf{a} \mathbf{a}^T}{\mathbf{a}^T \mathbf{a}} \mathbf{b}$$

The **projection matrix** onto $\mathbf{a}$:
$$P = \frac{\mathbf{a}\mathbf{a}^T}{\mathbf{a}^T\mathbf{a}}$$

### 9.2 Projection onto a Subspace

Projection of $\mathbf{b}$ onto the column space of $A$:

$$\hat{\mathbf{b}} = A(A^T A)^{-1} A^T \mathbf{b} = P_A \mathbf{b}$$

**Projection matrix:**
$$P_A = A(A^T A)^{-1} A^T$$

**Properties of projection matrix $P$:**
- $P^2 = P$ (idempotent)
- $P^T = P$ (symmetric)
- Eigenvalues are 0 or 1

**Error vector (residual):**
$$\mathbf{e} = \mathbf{b} - \hat{\mathbf{b}} = (I - P_A)\mathbf{b}$$

$\mathbf{e} \perp \mathcal{C}(A)$

**Example (ML):** The hat matrix in linear regression $H = X(X^T X)^{-1} X^T$ is a projection matrix that projects $\mathbf{y}$ onto column space of $X$.

---

## 10. Determinants

### 10.1 Definition

For $A \in \mathbb{R}^{n \times n}$:

$$\det(A) = \sum_{\sigma \in S_n} \text{sgn}(\sigma) \prod_{i=1}^n a_{i,\sigma(i)}$$

**2×2:**
$$\det\begin{bmatrix} a & b \\ c & d \end{bmatrix} = ad - bc$$

**3×3 (cofactor expansion):**
$$\det(A) = a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

### 10.2 Properties

- $\det(I) = 1$
- $\det(AB) = \det(A)\det(B)$
- $\det(A^T) = \det(A)$
- $\det(A^{-1}) = \frac{1}{\det(A)}$
- $\det(cA) = c^n \det(A)$ for $A \in \mathbb{R}^{n \times n}$
- $\det(A) = \prod_{i=1}^{n} \lambda_i$ (product of eigenvalues)
- Swapping two rows: $\det \to -\det$
- If two rows identical: $\det = 0$

### 10.3 Geometric Meaning

$|\det(A)|$ = volume of the parallelotope spanned by the columns of $A$.

- $|\det(A)| = 1$: Volume-preserving transformation
- $|\det(A)| < 1$: Compression
- $|\det(A)| > 1$: Expansion
- $\det(A) = 0$: Matrix is **singular** (not invertible, collapses to lower dimension)

**Example (ML):** In Normalizing Flows (generative models), the log-determinant of the Jacobian tracks how volume changes under the transformation:

$$\log p(\mathbf{x}) = \log p_z(f(\mathbf{x})) + \log \left| \det \frac{\partial f}{\partial \mathbf{x}} \right|$$

---

## 11. Eigenvalues & Eigenvectors

### 11.1 Definition

For square $A \in \mathbb{R}^{n \times n}$, scalar $\lambda$ and nonzero vector $\mathbf{v}$:

$$A\mathbf{v} = \lambda \mathbf{v}$$

- $\lambda$ is an **eigenvalue**
- $\mathbf{v}$ is the corresponding **eigenvector**

### 11.2 Finding Eigenvalues

Solve the **characteristic equation**:

$$\det(A - \lambda I) = 0$$

This is a degree-$n$ polynomial in $\lambda$ called the **characteristic polynomial**.

**Example:** $A = \begin{bmatrix} 3 & 1 \\ 0 & 2 \end{bmatrix}$

$$\det\begin{bmatrix} 3-\lambda & 1 \\ 0 & 2-\lambda \end{bmatrix} = (3-\lambda)(2-\lambda) = 0$$

$\lambda_1 = 3,\quad \lambda_2 = 2$

### 11.3 Finding Eigenvectors

For each $\lambda_i$, solve $(A - \lambda_i I)\mathbf{v} = \mathbf{0}$.

**Example (continued):** For $\lambda_1 = 3$:

$$\begin{bmatrix} 0 & 1 \\ 0 & -1 \end{bmatrix} \mathbf{v} = \mathbf{0} \implies \mathbf{v}_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$$

### 11.4 Properties

- $\text{tr}(A) = \sum_i \lambda_i$
- $\det(A) = \prod_i \lambda_i$
- Symmetric matrix: all $\lambda_i \in \mathbb{R}$, eigenvectors are orthogonal
- Orthogonal matrix: all $|\lambda_i| = 1$
- PSD matrix: all $\lambda_i \geq 0$

### 11.5 Eigendecomposition (Spectral Decomposition)

If $A$ has $n$ linearly independent eigenvectors:

$$A = V \Lambda V^{-1}$$

where $V = [\mathbf{v}_1 | \mathbf{v}_2 | \cdots | \mathbf{v}_n]$ and $\Lambda = \text{diag}(\lambda_1, \ldots, \lambda_n)$.

For **symmetric** $A = A^T$ (eigendecomposition is orthogonal):

$$A = Q \Lambda Q^T = \sum_{i=1}^{n} \lambda_i \mathbf{q}_i \mathbf{q}_i^T$$

where $Q$ is orthogonal ($Q^T = Q^{-1}$).

**Matrix power:**
$$A^k = V \Lambda^k V^{-1}, \quad \Lambda^k = \text{diag}(\lambda_1^k, \ldots, \lambda_n^k)$$

**Matrix exponential:**
$$e^A = V e^\Lambda V^{-1} = V \,\text{diag}(e^{\lambda_1}, \ldots, e^{\lambda_n})\, V^{-1}$$

### 11.6 Rayleigh Quotient

$$R(A, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$

$$\lambda_{\min} \leq R(A, \mathbf{x}) \leq \lambda_{\max}$$

Minimized/maximized by the eigenvectors corresponding to $\lambda_{\min}$/$\lambda_{\max}$.

**Example (ML):** PCA finds the direction $\mathbf{w}$ maximizing variance, which is:
$$\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \Sigma \mathbf{w} = \lambda_{\max}(\Sigma)$$

---

## 12. Matrix Decompositions

### 12.1 LU Decomposition

$$A = LU$$

- $L$: lower triangular with 1s on diagonal
- $U$: upper triangular

**With pivoting (LUP):**
$$PA = LU$$

**Use:** Solving $A\mathbf{x} = \mathbf{b}$ efficiently in $O(n^2)$ (after $O(n^3)$ decomposition).

---

### 12.2 Cholesky Decomposition

For symmetric positive definite $A$:

$$A = LL^T$$

where $L$ is lower triangular with positive diagonal entries.

**Computation:** Twice as fast as LU for PSD matrices.

**Key formulas:**
$$L_{ii} = \sqrt{A_{ii} - \sum_{k=1}^{i-1} L_{ik}^2}$$

$$L_{ji} = \frac{1}{L_{ii}} \left( A_{ji} - \sum_{k=1}^{i-1} L_{jk} L_{ik} \right), \quad j > i$$

**Example (ML):** Multivariate Gaussian sampling: If $\Sigma = LL^T$, then $\mathbf{x} = L\mathbf{z} + \boldsymbol{\mu}$ where $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, I)$ gives $\mathbf{x} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)$.

---

### 12.3 QR Decomposition

$$A = QR$$

- $Q \in \mathbb{R}^{m \times n}$: orthonormal columns ($Q^T Q = I$)
- $R \in \mathbb{R}^{n \times n}$: upper triangular

**Gram-Schmidt process** (orthogonalization):

$$\mathbf{u}_k = \mathbf{a}_k - \sum_{j=1}^{k-1} \frac{\mathbf{u}_j^T \mathbf{a}_k}{\mathbf{u}_j^T \mathbf{u}_j} \mathbf{u}_j, \qquad \mathbf{q}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

**Use:** Numerically stable least squares; eigenvalue algorithms.

---

### 12.4 Eigendecomposition

(See Section 11.5)

---

## 13. Singular Value Decomposition (SVD)

### 13.1 Definition

For any $A \in \mathbb{R}^{m \times n}$:

$$\boxed{A = U \Sigma V^T}$$

- $U \in \mathbb{R}^{m \times m}$: left singular vectors (orthogonal, columns are eigenvectors of $AA^T$)
- $\Sigma \in \mathbb{R}^{m \times n}$: diagonal matrix of **singular values** $\sigma_1 \geq \sigma_2 \geq \cdots \geq \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$: right singular vectors (orthogonal, columns are eigenvectors of $A^T A$)

**Economy (thin) SVD:**
$$A = U_r \Sigma_r V_r^T$$

where $r = \text{rank}(A)$, $U_r \in \mathbb{R}^{m \times r}$, $\Sigma_r \in \mathbb{R}^{r \times r}$, $V_r \in \mathbb{R}^{n \times r}$.

### 13.2 Relationship to Eigendecomposition

$$A^T A = V \Sigma^T U^T U \Sigma V^T = V \Sigma^2 V^T$$

$$AA^T = U \Sigma V^T V \Sigma^T U^T = U \Sigma^2 U^T$$

So:
- Singular values: $\sigma_i = \sqrt{\lambda_i(A^T A)}$
- Right singular vectors $V$: eigenvectors of $A^T A$
- Left singular vectors $U$: eigenvectors of $AA^T$

### 13.3 Rank-$k$ Approximation (Eckart–Young Theorem)

$$A \approx A_k = \sum_{i=1}^{k} \sigma_i \mathbf{u}_i \mathbf{v}_i^T$$

This is the **best rank-$k$ approximation** in both Frobenius and spectral norms:

$$\|A - A_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

$$\|A - A_k\|_2 = \sigma_{k+1}$$

No other rank-$k$ matrix achieves a smaller error.

### 13.4 Properties

$$\|A\|_F = \sqrt{\sum_i \sigma_i^2}, \qquad \|A\|_2 = \sigma_1, \qquad \|A\|_* = \sum_i \sigma_i$$

$$\det(A) = \prod_i \sigma_i \quad \text{(for square } A\text{)}$$

$$\text{rank}(A) = \text{number of nonzero singular values}$$

### 13.5 Pseudoinverse (Moore-Penrose)

$$A^+ = V \Sigma^+ U^T$$

$$\Sigma^+_{ii} = \begin{cases} 1/\sigma_i & \text{if } \sigma_i > 0 \\ 0 & \text{otherwise} \end{cases}$$

**Least squares minimum-norm solution:**
$$\hat{\mathbf{x}} = A^+ \mathbf{b}$$

**Example (ML):**
- **Recommender systems:** SVD on user-item rating matrix for collaborative filtering
- **Image compression:** Keep top-$k$ singular values, discard the rest
- **NLP/LSA:** SVD on term-document matrix reveals latent semantic structure
- **Neural networks:** Spectral normalization uses largest singular value for Lipschitz constraint

---

## 14. Principal Component Analysis (PCA)

### 14.1 Goal

Find a low-dimensional linear subspace that captures maximum variance in the data.

### 14.2 Setup

Given data matrix $X \in \mathbb{R}^{n \times d}$ with $n$ samples, $d$ features.

**Step 1 — Center the data:**
$$\tilde{X} = X - \mathbf{1}\bar{\mathbf{x}}^T, \quad \bar{\mathbf{x}} = \frac{1}{n} X^T \mathbf{1}$$

**Step 2 — Compute covariance matrix:**
$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$

**Step 3 — Eigendecompose:**
$$\Sigma = Q \Lambda Q^T, \quad \lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_d \geq 0$$

**Step 4 — Project onto top-$k$ eigenvectors:**
$$Z = \tilde{X} Q_k \in \mathbb{R}^{n \times k}$$

where $Q_k$ contains the top-$k$ eigenvectors (principal components).

### 14.3 Variance Explained

$$\text{Explained variance ratio}_k = \frac{\lambda_k}{\sum_{i=1}^d \lambda_i}$$

$$\text{Total explained by top-}k = \frac{\sum_{i=1}^k \lambda_i}{\sum_{i=1}^d \lambda_i}$$

### 14.4 PCA via SVD

$$\tilde{X} = U S V^T$$

Then:
- Principal components: columns of $V$
- Scores: $Z = US$ (or $\tilde{X}V$)
- Singular values $S$ relate to eigenvalues: $\lambda_i = s_i^2 / (n-1)$

### 14.5 Reconstruction & Error

Reconstruction:
$$\hat{X} = Z Q_k^T + \mathbf{1}\bar{\mathbf{x}}^T$$

Reconstruction error:
$$\|X - \hat{X}\|_F^2 = \sum_{i=k+1}^d \lambda_i$$

**Example (ML):** Visualizing high-dimensional word embeddings by projecting to 2D with PCA; preprocessing face images (eigenfaces).

---

## 15. Positive Definite Matrices

### 15.1 Tests for Positive Definiteness

$A$ is positive definite (PD) iff any of these hold:

1. $\mathbf{x}^T A \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$
2. All eigenvalues $\lambda_i > 0$
3. All leading principal minors (determinants of top-left $k \times k$ submatrices) are positive — **Sylvester's criterion**
4. Cholesky decomposition $A = LL^T$ exists with positive diagonal $L$
5. $A = B^T B$ for some matrix $B$ with full column rank

### 15.2 Quadratic Forms

$$f(\mathbf{x}) = \mathbf{x}^T A \mathbf{x} = \sum_{i,j} A_{ij} x_i x_j$$

| $A$ type | Quadratic form | Shape |
|----------|---------------|-------|
| PD | $> 0$ for all $\mathbf{x} \neq 0$ | Bowl (min) |
| PSD | $\geq 0$ for all $\mathbf{x}$ | Bowl or flat |
| ND | $< 0$ for all $\mathbf{x} \neq 0$ | Inverted bowl (max) |
| Indefinite | Can be $+$ or $-$ | Saddle point |

**Example (ML):** The Hessian $H = \nabla^2 L(\boldsymbol{\theta})$ of a convex loss:
- $H \succ 0 \implies$ unique minimum
- $H \succeq 0 \implies$ convex (possibly multiple minima)
- $H$ indefinite $\implies$ saddle point (common in deep networks!)

---

## 16. Matrix Calculus

### 16.1 Gradient

For $f: \mathbb{R}^n \to \mathbb{R}$, gradient wrt column vector $\mathbf{x}$:

$$\nabla_\mathbf{x} f = \frac{\partial f}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f}{\partial x_1} \\ \frac{\partial f}{\partial x_2} \\ \vdots \\ \frac{\partial f}{\partial x_n} \end{bmatrix} \in \mathbb{R}^n$$

### 16.2 Jacobian

For $\mathbf{f}: \mathbb{R}^n \to \mathbb{R}^m$:

$$J = \frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix} \in \mathbb{R}^{m \times n}$$

### 16.3 Hessian

For $f: \mathbb{R}^n \to \mathbb{R}$:

$$H = \nabla^2 f = \frac{\partial^2 f}{\partial \mathbf{x} \partial \mathbf{x}^T} = \begin{bmatrix} \frac{\partial^2 f}{\partial x_1^2} & \cdots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial^2 f}{\partial x_n \partial x_1} & \cdots & \frac{\partial^2 f}{\partial x_n^2} \end{bmatrix} \in \mathbb{R}^{n \times n}$$

Hessian is **symmetric** for twice-differentiable $f$.

### 16.4 Essential Derivative Rules for ML

| Expression | Derivative | Notes |
|------------|-----------|-------|
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T \mathbf{x})$ | $\mathbf{a}$ | Linear |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{a})$ | $\mathbf{a}$ | Linear |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T \mathbf{x})$ | $2\mathbf{x}$ | Quadratic |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{x}^T A \mathbf{x})$ | $(A + A^T)\mathbf{x} = 2A\mathbf{x}$ | Quadratic (symmetric $A$) |
| $\frac{\partial}{\partial \mathbf{x}}(\mathbf{a}^T A \mathbf{x})$ | $A^T \mathbf{a}$ | |
| $\frac{\partial}{\partial X}(\mathbf{a}^T X \mathbf{b})$ | $\mathbf{a}\mathbf{b}^T$ | |
| $\frac{\partial}{\partial X}\text{tr}(AX)$ | $A^T$ | |
| $\frac{\partial}{\partial X}\text{tr}(X^T A)$ | $A$ | |
| $\frac{\partial}{\partial X}\text{tr}(AXB)$ | $A^T \cdot B^T$ ... wait: $B A$ | Cyclic trace |
| $\frac{\partial}{\partial X}\text{tr}(X^T A X)$ | $(A + A^T)X$ | |
| $\frac{\partial}{\partial X}\log\det(X)$ | $(X^{-1})^T = X^{-T}$ | |
| $\frac{\partial}{\partial X}\det(X)$ | $\det(X) \cdot X^{-T}$ | |

### 16.5 Chain Rule

For composition $f(g(\mathbf{x}))$:

$$\frac{\partial f}{\partial \mathbf{x}} = \frac{\partial g}{\partial \mathbf{x}}^T \frac{\partial f}{\partial \mathbf{g}}$$

**Vector chain rule (Jacobians):**
$$\frac{\partial \mathbf{f}}{\partial \mathbf{x}} = \frac{\partial \mathbf{f}}{\partial \mathbf{g}} \cdot \frac{\partial \mathbf{g}}{\partial \mathbf{x}}$$

### 16.6 Gradient of Common ML Loss Functions

**MSE loss** $L = \frac{1}{2n}\|X\boldsymbol{\theta} - \mathbf{y}\|^2$:

$$\nabla_{\boldsymbol{\theta}} L = \frac{1}{n} X^T(X\boldsymbol{\theta} - \mathbf{y})$$

**Cross-entropy loss** with sigmoid $\sigma(z) = (1+e^{-z})^{-1}$:

$$L = -\frac{1}{n}\sum_i [y_i \log \hat{y}_i + (1-y_i)\log(1-\hat{y}_i)]$$

$$\nabla_{\boldsymbol{\theta}} L = \frac{1}{n} X^T(\hat{\mathbf{y}} - \mathbf{y})$$

**Softmax cross-entropy:**

$$\hat{y}_k = \frac{e^{z_k}}{\sum_j e^{z_j}}$$

$$\frac{\partial L}{\partial \mathbf{z}} = \hat{\mathbf{y}} - \mathbf{y}$$

**L2 regularization:**

$$L_{\text{reg}} = L + \frac{\lambda}{2}\|\boldsymbol{\theta}\|_2^2 \implies \nabla_{\boldsymbol{\theta}} L_{\text{reg}} = \nabla_{\boldsymbol{\theta}} L + \lambda \boldsymbol{\theta}$$

---

## 17. Tensors (for Deep Learning)

### 17.1 Tensor Basics

A rank-$N$ tensor $\mathcal{T} \in \mathbb{R}^{d_1 \times d_2 \times \cdots \times d_N}$ generalizes matrices to $N$ dimensions.

**Element access:** $\mathcal{T}_{i_1, i_2, \ldots, i_N}$

**Fibers (1D slices):** Fix all but one index.

**Slices (2D slices):** Fix all but two indices.

### 17.2 Tensor Operations

**Tensor product (outer product):**
$$(\mathbf{u} \otimes \mathbf{v})_{ij} = u_i v_j$$

**Tensor contraction (generalized trace):**
$$C_{ij} = \sum_k A_{ikm} B_{kjm}$$

**Mode-$n$ product** of tensor $\mathcal{T} \in \mathbb{R}^{I_1 \times \cdots \times I_N}$ with matrix $M \in \mathbb{R}^{J \times I_n}$:

$$(\mathcal{T} \times_n M)_{i_1 \cdots i_{n-1} j i_{n+1} \cdots i_N} = \sum_{i_n} \mathcal{T}_{i_1 \cdots i_N} M_{j i_n}$$

### 17.3 Vectorization & Reshaping

Vectorization (flatten tensor to 1D vector, column-major):
$$\text{vec}(A) \in \mathbb{R}^{mn} \quad \text{for } A \in \mathbb{R}^{m \times n}$$

Key identity:
$$\text{vec}(AXB) = (B^T \otimes A)\text{vec}(X)$$

### 17.4 Kronecker Product

$$A \otimes B = \begin{bmatrix} a_{11}B & a_{12}B & \cdots \\ a_{21}B & a_{22}B & \cdots \\ \vdots & & \ddots \end{bmatrix}$$

For $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{p \times q}$: $A \otimes B \in \mathbb{R}^{mp \times nq}$

**Properties:**
- $(A \otimes B)^T = A^T \otimes B^T$
- $(A \otimes B)(C \otimes D) = AC \otimes BD$
- $(A \otimes B)^{-1} = A^{-1} \otimes B^{-1}$

### 17.5 Batch Matrix Multiply (BMM)

For 3D tensors $A \in \mathbb{R}^{b \times m \times k}$, $B \in \mathbb{R}^{b \times k \times n}$:

$$C_{b,i,j} = \sum_{l} A_{b,i,l} B_{b,l,j}$$

Result: $C \in \mathbb{R}^{b \times m \times n}$ — $b$ independent matrix multiplications in parallel.

**Example (DL):** Multi-head attention:
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

where $Q, K, V \in \mathbb{R}^{n \times d_k}$ — all matrix products.

---

## 18. Linear Transformations

### 18.1 Definition

$T: \mathbb{R}^n \to \mathbb{R}^m$ is linear if:
1. $T(\mathbf{u} + \mathbf{v}) = T(\mathbf{u}) + T(\mathbf{v})$
2. $T(c\mathbf{u}) = c\, T(\mathbf{u})$

Every linear transformation can be represented as a matrix: $T(\mathbf{x}) = A\mathbf{x}$

### 18.2 Fundamental Transformations

**Scaling:**
$$S = \begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}$$

**Rotation by $\theta$:**
$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

**Reflection** (about $x$-axis):
$$F = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}$$

**Shear:**
$$H = \begin{bmatrix} 1 & k \\ 0 & 1 \end{bmatrix}$$

**Projection** (onto $x$-axis):
$$P = \begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}$$

### 18.3 Composition

$$T = T_2 \circ T_1 \implies A_T = A_2 A_1$$

Note: **order matters** — right matrix is applied first.

### 18.4 Affine Transformations

$$T(\mathbf{x}) = A\mathbf{x} + \mathbf{b}$$

Used everywhere in neural networks (linear layers + bias).

**Homogeneous coordinates** (to represent affine as linear):

$$\begin{bmatrix} A & \mathbf{b} \\ \mathbf{0}^T & 1 \end{bmatrix} \begin{bmatrix} \mathbf{x} \\ 1 \end{bmatrix} = \begin{bmatrix} A\mathbf{x} + \mathbf{b} \\ 1 \end{bmatrix}$$

---

## 19. Dimensionality Reduction

### 19.1 Why Dimensionality Matters

**Curse of dimensionality:** As $d$ grows, data becomes sparse — distances become meaningless, and we need exponentially more data.

In $d$ dimensions with $n$ data points, average distance between points $\sim O(n^{-1/d})$ — vanishes fast.

### 19.2 Linear Methods

| Method | Objective | Key Formula |
|--------|-----------|-------------|
| PCA | Max variance | $\max \mathbf{w}^T \Sigma \mathbf{w}$ s.t. $\|\mathbf{w}\|=1$ |
| LDA | Max class sep. | $\max \frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$ |
| SVD | Min reconstr. error | $\min_{r} \|A - A_r\|_F$ |
| Random Proj. | Preserve distances | $JL: \frac{1}{\sqrt{k}} R$ |

**LDA (Linear Discriminant Analysis):**

$$S_B = \sum_c n_c (\boldsymbol{\mu}_c - \boldsymbol{\mu})(\boldsymbol{\mu}_c - \boldsymbol{\mu})^T \quad \text{(between-class)}$$

$$S_W = \sum_c \sum_{\mathbf{x} \in c} (\mathbf{x} - \boldsymbol{\mu}_c)(\mathbf{x} - \boldsymbol{\mu}_c)^T \quad \text{(within-class)}$$

Solve generalized eigenvalue problem: $S_B \mathbf{w} = \lambda S_W \mathbf{w}$

### 19.3 Johnson-Lindenstrauss Lemma

For $n$ points in $\mathbb{R}^d$, a random projection to $k = O(\log n / \varepsilon^2)$ dimensions preserves pairwise distances up to $(1 \pm \varepsilon)$ with high probability:

$$(1-\varepsilon)\|\mathbf{u} - \mathbf{v}\|^2 \leq \|R(\mathbf{u} - \mathbf{v})\|^2 \leq (1+\varepsilon)\|\mathbf{u} - \mathbf{v}\|^2$$

---

## 20. Applications in ML/DL

### 20.1 Linear Regression

**Model:** $\hat{\mathbf{y}} = X\boldsymbol{\theta}$

**Loss:** $L = \frac{1}{2}\|X\boldsymbol{\theta} - \mathbf{y}\|_2^2$

**Closed-form solution:**
$$\hat{\boldsymbol{\theta}} = (X^T X)^{-1} X^T \mathbf{y}$$

**Geometric interpretation:** $\hat{\mathbf{y}} = X\hat{\boldsymbol{\theta}}$ is the projection of $\mathbf{y}$ onto $\mathcal{C}(X)$.

---

### 20.2 Neural Network Forward Pass

**Single linear layer:**
$$\mathbf{z} = W\mathbf{x} + \mathbf{b}, \quad W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$$

**Multi-layer (L layers):**
$$\mathbf{h}^{(l)} = \sigma(W^{(l)} \mathbf{h}^{(l-1)} + \mathbf{b}^{(l)})$$

**Batch processing:** $Z = XW^T + \mathbf{1}\mathbf{b}^T$ for $X \in \mathbb{R}^{n \times d}$

---

### 20.3 Backpropagation via Matrix Calculus

For layer $\mathbf{z} = W\mathbf{x} + \mathbf{b}$, $\mathbf{h} = \sigma(\mathbf{z})$:

$$\frac{\partial L}{\partial W} = \frac{\partial L}{\partial \mathbf{z}} \mathbf{x}^T = \boldsymbol{\delta} \mathbf{x}^T \quad \text{(outer product!)}$$

$$\frac{\partial L}{\partial \mathbf{b}} = \frac{\partial L}{\partial \mathbf{z}} = \boldsymbol{\delta}$$

$$\frac{\partial L}{\partial \mathbf{x}} = W^T \frac{\partial L}{\partial \mathbf{z}} = W^T \boldsymbol{\delta}$$

---

### 20.4 Convolutional Layers

Convolution is a **linear operation** expressible as matrix multiplication with a **Toeplitz matrix** (circulant structure):

$$\mathbf{y} = C\mathbf{x}$$

where $C$ has the filter weights repeated with stride — this reveals why CNNs are efficient (parameter sharing = structured matrix multiplication).

---

### 20.5 Attention Mechanism

Self-attention computes:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T \in \mathbb{R}^{n \times n}$: pairwise similarity (dot-product = inner product)
- Scaling by $\frac{1}{\sqrt{d_k}}$: prevents saturation of softmax
- $V$: value matrix weighted by attention scores

All operations are pure linear algebra.

---

### 20.6 Principal Component Analysis in ML Pipeline

Feature compression pipeline:

$$X_{\text{train}} \xrightarrow{\text{center}} \tilde{X} \xrightarrow{\text{SVD}} U, \Sigma, V^T \xrightarrow{\text{project}} Z = \tilde{X} V_k$$

**Whitening transformation (useful preprocessing):**

$$X_{\text{white}} = \tilde{X} V \Lambda^{-1/2}$$

Produces decorrelated features with unit variance.

---

### 20.7 Covariance & Gram Matrices

**Covariance matrix:**
$$\Sigma = \frac{1}{n-1} \tilde{X}^T \tilde{X} \in \mathbb{R}^{d \times d}$$

Captures feature correlations. Eigenvalues = variances along principal axes.

**Gram matrix:**
$$G = \tilde{X} \tilde{X}^T \in \mathbb{R}^{n \times n}$$

$G_{ij} = \tilde{\mathbf{x}}_i^T \tilde{\mathbf{x}}_j$ — all pairwise dot products. Central to kernel methods: $G = K(X, X)$.

**Relationship via SVD:** If $\tilde{X} = U\Sigma V^T$:

$$\Sigma_{\text{cov}} = \frac{1}{n-1} V \Sigma^2 V^T, \qquad G = U \Sigma^2 U^T$$

---

### 20.8 Optimization: Gradient Descent

**Gradient Descent:**
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \nabla_{\boldsymbol{\theta}} L(\boldsymbol{\theta}^{(t)})$$

**Newton's Method (uses Hessian):**
$$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - H^{-1} \nabla L$$

where $H = \nabla^2 L$ is the Hessian. Converges in fewer steps but $H^{-1}$ is $O(n^3)$.

**Convergence condition (gradient descent):** If $L$ is $\beta$-smooth (Hessian eigenvalues $\leq \beta$):
$$\text{Converges for } \alpha \leq \frac{1}{\beta} = \frac{1}{\lambda_{\max}(H)}$$

---

### 20.9 Batch Normalization

For activations $\mathbf{z}$ in a layer:

$$\hat{\mathbf{z}} = \frac{\mathbf{z} - \boldsymbol{\mu}_B}{\sqrt{\boldsymbol{\sigma}_B^2 + \varepsilon}}, \qquad \mathbf{y} = \boldsymbol{\gamma} \odot \hat{\mathbf{z}} + \boldsymbol{\beta}$$

$$\boldsymbol{\mu}_B = \frac{1}{m}\sum_{i=1}^m \mathbf{z}_i, \qquad \boldsymbol{\sigma}_B^2 = \frac{1}{m}\sum_{i=1}^m (\mathbf{z}_i - \boldsymbol{\mu}_B)^2$$

This is an **affine normalization** — linear algebra at its core.

---

### 20.10 Singular Value Decomposition in NLP

**Latent Semantic Analysis (LSA):**

$$A = U_k \Sigma_k V_k^T$$

where $A$ is the term-document matrix. Top-$k$ singular vectors capture latent semantic topics.

**Word embeddings** can be seen as a matrix factorization:

$$M \approx WC^T$$

where $M_{ij} = \text{PMI}(w_i, c_j)$ (pointwise mutual information), $W$ and $C$ are word and context embeddings.

---

## 📌 Quick Reference — Key Formulas

$$\boxed{\text{Dot product: } \mathbf{u}^T\mathbf{v} = \|\mathbf{u}\|\|\mathbf{v}\|\cos\theta}$$

$$\boxed{\text{Matrix multiply: } C_{ij} = \sum_k A_{ik}B_{kj}}$$

$$\boxed{\text{SVD: } A = U\Sigma V^T}$$

$$\boxed{\text{Eigendecompose: } A\mathbf{v} = \lambda\mathbf{v} \iff A = V\Lambda V^{-1}}$$

$$\boxed{\text{Normal equations: } \hat{\boldsymbol{\theta}} = (X^TX)^{-1}X^T\mathbf{y}}$$

$$\boxed{\text{PCA direction: } \max_{\|\mathbf{w}\|=1} \mathbf{w}^T\Sigma\mathbf{w} = \lambda_{\max}(\Sigma)}$$

$$\boxed{\text{Gradient of quadratic: } \nabla_\mathbf{x}(\mathbf{x}^TA\mathbf{x}) = 2A\mathbf{x} \text{ (symmetric } A\text{)}}$$

$$\boxed{\text{Backprop: } \frac{\partial L}{\partial W} = \boldsymbol{\delta}\mathbf{x}^T, \quad \frac{\partial L}{\partial \mathbf{x}} = W^T\boldsymbol{\delta}}$$

$$\boxed{\text{Projection: } \hat{\mathbf{b}} = A(A^TA)^{-1}A^T\mathbf{b}}$$

$$\boxed{\text{Rank-}k\text{ approx: } A_k = \sum_{i=1}^k \sigma_i\mathbf{u}_i\mathbf{v}_i^T}$$

---

## 🔗 Connections Map

```
Scalars → Vectors → Matrices → Tensors
                        ↓
              Linear Transformations
                        ↓
           Eigenvalues/Eigenvectors
                ↓              ↓
      Eigendecomposition       SVD
           ↓                    ↓
          PCA           Low-rank Approx
           ↓                    ↓
   Dimensionality Red.    Recommender Sys.
                        ↓
              Matrix Calculus
                        ↓
           Gradient Descent / Backprop
                        ↓
              Neural Networks / ML Models
```

---

*These notes cover the complete linear algebra curriculum for ML/DL from first principles to production-level concepts. Every formula here appears somewhere in the implementation of modern machine learning systems.*
