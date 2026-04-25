# 🎲 Probability for Machine Learning & Deep Learning
### Complete Notes — From Absolute Basics to Advanced

> **"Probability is the mathematical language of uncertainty — and ML is the science of learning under uncertainty."**
> Pure theory, every formula, ML/DL context throughout.

---

## 📋 Table of Contents

1. [Foundations of Probability](#1-foundations-of-probability)
2. [Axioms & Basic Rules](#2-axioms--basic-rules)
3. [Conditional Probability](#3-conditional-probability)
4. [Bayes' Theorem](#4-bayes-theorem)
5. [Independence](#5-independence)
6. [Random Variables](#6-random-variables)
7. [Probability Distributions — Discrete](#7-probability-distributions--discrete)
8. [Probability Distributions — Continuous](#8-probability-distributions--continuous)
9. [Joint, Marginal & Conditional Distributions](#9-joint-marginal--conditional-distributions)
10. [Expectation](#10-expectation)
11. [Variance & Covariance](#11-variance--covariance)
12. [Moment Generating Functions & Characteristic Functions](#12-moment-generating-functions--characteristic-functions)
13. [Information Theory & Entropy](#13-information-theory--entropy)
14. [Common Inequalities](#14-common-inequalities)
15. [Limit Theorems](#15-limit-theorems)
16. [Bayesian Probability & Inference](#16-bayesian-probability--inference)
17. [Probabilistic Graphical Models](#17-probabilistic-graphical-models)
18. [Exponential Family](#18-exponential-family)
19. [Maximum Likelihood & MAP Estimation](#19-maximum-likelihood--map-estimation)
20. [Probabilistic Models in ML/DL](#20-probabilistic-models-in-mldl)

---

## 1. Foundations of Probability

### 1.1 Sample Space

The **sample space** $\Omega$ is the set of all possible outcomes of a random experiment.

**Examples:**
- Coin flip: $\Omega = \{H, T\}$
- Die roll: $\Omega = \{1, 2, 3, 4, 5, 6\}$
- Neural network output (regression): $\Omega = \mathbb{R}$
- Pixel intensity: $\Omega = \{0, 1, 2, \ldots, 255\}$

---

### 1.2 Event

An **event** $A$ is any subset of $\Omega$: $A \subseteq \Omega$.

- **Elementary event:** single outcome $\{\omega\}$
- **Certain event:** $\Omega$ (always occurs)
- **Impossible event:** $\emptyset$ (never occurs)

**Set operations on events:**

| Operation | Notation | Meaning |
|-----------|----------|---------|
| Union | $A \cup B$ | $A$ or $B$ (or both) |
| Intersection | $A \cap B$ | Both $A$ and $B$ |
| Complement | $A^c$ or $\bar{A}$ | Not $A$ |
| Difference | $A \setminus B$ | $A$ but not $B$ |

**De Morgan's Laws:**
$$(A \cup B)^c = A^c \cap B^c$$
$$(A \cap B)^c = A^c \cup B^c$$

---

### 1.3 Sigma-Algebra (Borel Field)

A collection $\mathcal{F}$ of subsets of $\Omega$ is a **sigma-algebra** if:
1. $\Omega \in \mathcal{F}$
2. $A \in \mathcal{F} \implies A^c \in \mathcal{F}$ (closed under complement)
3. $A_1, A_2, \ldots \in \mathcal{F} \implies \bigcup_{i=1}^\infty A_i \in \mathcal{F}$ (closed under countable union)

The triple $(\Omega, \mathcal{F}, P)$ is a **probability space**.

---

### 1.4 Interpretations of Probability

| Interpretation | Definition | ML Use |
|----------------|-----------|--------|
| **Frequentist** | $P(A) = \lim_{n\to\infty} \frac{n_A}{n}$ | MLE, confidence intervals |
| **Bayesian** | Degree of belief | Bayesian inference, priors |
| **Classical** | $P(A) = \frac{|A|}{|\Omega|}$ | Uniform distributions |

---

## 2. Axioms & Basic Rules

### 2.1 Kolmogorov Axioms

$$\boxed{P(\Omega) = 1}$$
$$\boxed{P(A) \geq 0 \quad \forall A \in \mathcal{F}}$$
$$\boxed{P\!\left(\bigcup_{i=1}^\infty A_i\right) = \sum_{i=1}^\infty P(A_i) \quad \text{for mutually exclusive } A_i}$$

### 2.2 Derived Rules

**Complement rule:**
$$P(A^c) = 1 - P(A)$$

**Probability of impossible event:**
$$P(\emptyset) = 0$$

**Monotonicity:**
$$A \subseteq B \implies P(A) \leq P(B)$$

**Inclusion-Exclusion (two events):**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Inclusion-Exclusion (three events):**
$$P(A \cup B \cup C) = P(A) + P(B) + P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

**General Inclusion-Exclusion:**
$$P\!\left(\bigcup_{i=1}^n A_i\right) = \sum_{k=1}^n (-1)^{k+1} \sum_{1 \leq i_1 < \cdots < i_k \leq n} P(A_{i_1} \cap \cdots \cap A_{i_k})$$

**Boole's Inequality (Union Bound):**
$$P\!\left(\bigcup_{i=1}^n A_i\right) \leq \sum_{i=1}^n P(A_i)$$

**Example (ML):** The probability that a model makes at least one error across $n$ independent predictions is bounded above by $\sum_i P(\text{error}_i)$ — used to bound generalization error.

---

## 3. Conditional Probability

### 3.1 Definition

$$\boxed{P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0}$$

The probability of $A$ **given** that $B$ has occurred.

**Example (ML):** $P(\text{label}=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})$ — probability of class 1 given feature vector $\mathbf{x}$ in logistic regression.

---

### 3.2 Multiplication Rule

$$P(A \cap B) = P(A \mid B) P(B) = P(B \mid A) P(A)$$

**General chain rule:**
$$P(A_1 \cap A_2 \cap \cdots \cap A_n) = P(A_1) P(A_2 \mid A_1) P(A_3 \mid A_1, A_2) \cdots P(A_n \mid A_1, \ldots, A_{n-1})$$

$$= \prod_{i=1}^n P(A_i \mid A_1, \ldots, A_{i-1})$$

**Example (DL):** Language model probability of a sentence:
$$P(w_1, w_2, \ldots, w_T) = \prod_{t=1}^T P(w_t \mid w_1, \ldots, w_{t-1})$$

---

### 3.3 Law of Total Probability

If $\{B_1, B_2, \ldots, B_n\}$ is a **partition** of $\Omega$ (mutually exclusive, exhaustive):

$$\boxed{P(A) = \sum_{i=1}^n P(A \mid B_i) P(B_i)}$$

Continuous version:
$$P(A) = \int P(A \mid B = b) \, p(b) \, db$$

**Example (ML):** Marginal likelihood in a mixture model:
$$p(\mathbf{x}) = \sum_{k=1}^K p(\mathbf{x} \mid z=k) P(z=k)$$

---

## 4. Bayes' Theorem

### 4.1 Bayes' Theorem

$$\boxed{P(B \mid A) = \frac{P(A \mid B) P(B)}{P(A)} = \frac{P(A \mid B) P(B)}{\sum_j P(A \mid B_j) P(B_j)}}$$

**Components:**
- $P(B)$: **prior** — belief before seeing data
- $P(A \mid B)$: **likelihood** — probability of data given hypothesis
- $P(B \mid A)$: **posterior** — updated belief after seeing data
- $P(A)$: **evidence** (marginal likelihood, normalizing constant)

**Posterior proportional to Likelihood times Prior:**
$$P(B \mid A) \propto P(A \mid B) \cdot P(B)$$

### 4.2 Continuous Bayes' Theorem

$$\boxed{p(\boldsymbol{\theta} \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})}{p(\mathcal{D})} = \frac{p(\mathcal{D} \mid \boldsymbol{\theta})\, p(\boldsymbol{\theta})}{\int p(\mathcal{D} \mid \boldsymbol{\theta}')\, p(\boldsymbol{\theta}')\, d\boldsymbol{\theta}'}}$$

**Example (ML):** Bayesian linear regression — given data $\mathcal{D} = (X, \mathbf{y})$:
$$p(\mathbf{w} \mid X, \mathbf{y}) = \frac{p(\mathbf{y} \mid X, \mathbf{w})\, p(\mathbf{w})}{p(\mathbf{y} \mid X)}$$

### 4.3 Naive Bayes Classifier

Assumes features are **conditionally independent** given the class label $y$:

$$P(y \mid x_1, \ldots, x_n) \propto P(y) \prod_{i=1}^n P(x_i \mid y)$$

$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^n P(x_i \mid y)$$

---

## 5. Independence

### 5.1 Independence of Events

Events $A$ and $B$ are **independent** if:

$$\boxed{P(A \cap B) = P(A) P(B)}$$

Equivalently: $P(A \mid B) = P(A)$, i.e., knowing $B$ gives no information about $A$.

### 5.2 Mutual Independence

Events $A_1, \ldots, A_n$ are **mutually independent** if for every subset $S \subseteq \{1, \ldots, n\}$:

$$P\!\left(\bigcap_{i \in S} A_i\right) = \prod_{i \in S} P(A_i)$$

> Pairwise independence does NOT imply mutual independence.

### 5.3 Conditional Independence

$A$ and $B$ are **conditionally independent given $C$**:

$$\boxed{P(A \cap B \mid C) = P(A \mid C) P(B \mid C)}$$

Notation: $A \perp\!\!\!\perp B \mid C$

**Example (ML/Graphical Models):** In a Markov chain:
$$P(X_t \mid X_{t-1}, X_{t-2}, \ldots) = P(X_t \mid X_{t-1})$$

Future is conditionally independent of the past given the present.

---

## 6. Random Variables

### 6.1 Definition

A **random variable** $X$ is a measurable function $X: \Omega \to \mathbb{R}$ mapping outcomes to real numbers.

- **Discrete RV:** takes countable values
- **Continuous RV:** takes uncountably many values

### 6.2 Cumulative Distribution Function (CDF)

$$\boxed{F_X(x) = P(X \leq x) \quad \forall x \in \mathbb{R}}$$

**Properties:**
- $0 \leq F(x) \leq 1$
- Non-decreasing: $x \leq y \implies F(x) \leq F(y)$
- Right-continuous: $\lim_{y \to x^+} F(y) = F(x)$
- $\lim_{x \to -\infty} F(x) = 0$, $\lim_{x \to +\infty} F(x) = 1$

$$P(a < X \leq b) = F(b) - F(a)$$

### 6.3 Probability Mass Function (PMF) — Discrete

$$\boxed{p_X(x) = P(X = x)}$$

$$\sum_{x} p_X(x) = 1, \qquad p_X(x) \geq 0$$

$$F(x) = \sum_{t \leq x} p_X(t)$$

### 6.4 Probability Density Function (PDF) — Continuous

$$\boxed{f_X(x) \geq 0, \qquad \int_{-\infty}^\infty f_X(x)\, dx = 1}$$

$$P(a \leq X \leq b) = \int_a^b f_X(x)\, dx$$

$$F(x) = \int_{-\infty}^x f_X(t)\, dt, \qquad f_X(x) = \frac{dF}{dx}$$

> Note: $f_X(x)$ is **not** a probability — it is a density. $f_X(x)$ can exceed 1.

### 6.5 Quantile Function (Inverse CDF)

$$Q(p) = F^{-1}(p) = \inf\{x : F(x) \geq p\}$$

- Median: $Q(0.5)$
- Quartiles: $Q(0.25)$, $Q(0.75)$

**Use (DL):** Quantile regression minimizes the pinball loss to predict quantiles directly.

### 6.6 Functions of a Random Variable

If $Y = g(X)$:

**Discrete:** $p_Y(y) = \sum_{x: g(x)=y} p_X(x)$

**Continuous (monotone $g$):**
$$f_Y(y) = f_X(g^{-1}(y)) \left|\frac{d}{dy} g^{-1}(y)\right|$$

**Multivariate change of variables:**
$$f_Y(\mathbf{y}) = f_X(g^{-1}(\mathbf{y})) \left|\det J_{g^{-1}}(\mathbf{y})\right|$$

where $J$ is the Jacobian. This formula is **critical in Normalizing Flows.**

---

## 7. Probability Distributions — Discrete

### 7.1 Bernoulli Distribution

$$X \sim \text{Bernoulli}(p), \quad X \in \{0, 1\}$$

$$\boxed{P(X=x) = p^x (1-p)^{1-x}}$$

$$\mathbb{E}[X] = p, \qquad \text{Var}(X) = p(1-p)$$

**Example (ML):** Binary classification output, coin flip, neuron firing.

---

### 7.2 Binomial Distribution

$$X \sim \text{Binomial}(n, p)$$ — number of successes in $n$ independent Bernoulli$(p)$ trials.

$$\boxed{P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n}$$

$$\mathbb{E}[X] = np, \qquad \text{Var}(X) = np(1-p)$$

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

---

### 7.3 Categorical Distribution

$$X \sim \text{Categorical}(\boldsymbol{\pi}), \quad \boldsymbol{\pi} = (\pi_1, \ldots, \pi_K),\quad \sum_k \pi_k = 1$$

$$\boxed{P(X=k) = \pi_k}$$

$$\mathbb{E}[\mathbf{1}(X=k)] = \pi_k$$

**Example (ML):** Multi-class classification output — softmax produces $\boldsymbol{\pi}$.

---

### 7.4 Multinomial Distribution

$$(\mathbf{x}) \sim \text{Multinomial}(n, \boldsymbol{\pi})$$

$$\boxed{P(X_1=x_1, \ldots, X_K=x_K) = \frac{n!}{x_1! \cdots x_K!} \prod_{k=1}^K \pi_k^{x_k}}$$

where $\sum_k x_k = n$.

$$\mathbb{E}[X_k] = n\pi_k, \qquad \text{Var}(X_k) = n\pi_k(1-\pi_k)$$

**Example (NLP):** Bag-of-words model — word counts in a document.

---

### 7.5 Poisson Distribution

$$X \sim \text{Poisson}(\lambda), \quad \lambda > 0$$

$$\boxed{P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots}$$

$$\mathbb{E}[X] = \lambda, \qquad \text{Var}(X) = \lambda$$

Models count of rare events in fixed time/space. Mean equals Variance.

---

### 7.6 Geometric Distribution

$$X \sim \text{Geometric}(p)$$ — number of trials until first success.

$$\boxed{P(X=k) = (1-p)^{k-1} p, \quad k = 1, 2, \ldots}$$

$$\mathbb{E}[X] = \frac{1}{p}, \qquad \text{Var}(X) = \frac{1-p}{p^2}$$

**Memoryless property:** $P(X > m+n \mid X > m) = P(X > n)$

---

### 7.7 Negative Binomial Distribution

$$X \sim \text{NegBin}(r, p)$$ — number of trials until $r$-th success.

$$\boxed{P(X=k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}, \quad k = r, r+1, \ldots}$$

$$\mathbb{E}[X] = \frac{r}{p}, \qquad \text{Var}(X) = \frac{r(1-p)}{p^2}$$

---

### 7.8 Hypergeometric Distribution

Drawing $n$ items without replacement from $N$ items ($K$ successes):

$$\boxed{P(X=k) = \frac{\binom{K}{k}\binom{N-K}{n-k}}{\binom{N}{n}}}$$

$$\mathbb{E}[X] = \frac{nK}{N}, \qquad \text{Var}(X) = n\frac{K}{N}\frac{N-K}{N}\frac{N-n}{N-1}$$

---

## 8. Probability Distributions — Continuous

### 8.1 Uniform Distribution

$$X \sim \text{Uniform}(a, b)$$

$$\boxed{f(x) = \frac{1}{b-a}, \quad a \leq x \leq b}$$

$$\mathbb{E}[X] = \frac{a+b}{2}, \qquad \text{Var}(X) = \frac{(b-a)^2}{12}$$

$$F(x) = \frac{x-a}{b-a}$$

**Example (ML):** Weight initialization (Glorot uniform), random seed generation.

---

### 8.2 Gaussian (Normal) Distribution

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$\boxed{f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)}$$

$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = \sigma^2$$

**Standard normal:** $Z \sim \mathcal{N}(0, 1)$

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2} \qquad \text{(PDF)}$$

$$\Phi(z) = \int_{-\infty}^z \phi(t)\, dt \qquad \text{(CDF)}$$

**Standardization:** $Z = \frac{X - \mu}{\sigma}$

**68-95-99.7 Rule:**
$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.6827$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.9545$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.9973$$

**Log-normal:** $Y = e^X$, $X \sim \mathcal{N}(\mu, \sigma^2)$:
$$\mathbb{E}[Y] = e^{\mu + \sigma^2/2}, \qquad \text{Var}(Y) = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}$$

**Example (ML):** Weight initialization (Xavier/He uses Gaussian), latent variables in VAE, noise models.

---

### 8.3 Multivariate Gaussian Distribution

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma), \quad \boldsymbol{\mu} \in \mathbb{R}^d,\ \Sigma \in \mathbb{R}^{d \times d}\ (\Sigma \succ 0)$$

$$\boxed{f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\Sigma|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)}$$

The term $(\mathbf{x}-\boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x}-\boldsymbol{\mu})$ is the squared **Mahalanobis distance**.

**Marginals:** For block structure $\mathbf{X} = (\mathbf{X}_1, \mathbf{X}_2)^T$ with $\boldsymbol{\mu} = (\boldsymbol{\mu}_1, \boldsymbol{\mu}_2)^T$ and $\Sigma = \begin{bmatrix}\Sigma_{11}&\Sigma_{12}\\\Sigma_{21}&\Sigma_{22}\end{bmatrix}$:

$$\mathbf{X}_1 \sim \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_{11})$$

**Conditional:**
$$\mathbf{X}_1 \mid \mathbf{X}_2 = \mathbf{x}_2 \sim \mathcal{N}(\boldsymbol{\mu}_{1|2},\ \Sigma_{1|2})$$

$$\boldsymbol{\mu}_{1|2} = \boldsymbol{\mu}_1 + \Sigma_{12}\Sigma_{22}^{-1}(\mathbf{x}_2 - \boldsymbol{\mu}_2)$$

$$\Sigma_{1|2} = \Sigma_{11} - \Sigma_{12}\Sigma_{22}^{-1}\Sigma_{21}$$

**Affine transformation:** If $\mathbf{Y} = A\mathbf{X} + \mathbf{b}$:
$$\mathbf{Y} \sim \mathcal{N}(A\boldsymbol{\mu} + \mathbf{b},\ A\Sigma A^T)$$

**Example (ML):** Gaussian Process, VAE latent space, Linear Discriminant Analysis.

---

### 8.4 Exponential Distribution

$$X \sim \text{Exp}(\lambda), \quad \lambda > 0$$

$$\boxed{f(x) = \lambda e^{-\lambda x}, \quad x \geq 0}$$

$$F(x) = 1 - e^{-\lambda x}$$

$$\mathbb{E}[X] = \frac{1}{\lambda}, \qquad \text{Var}(X) = \frac{1}{\lambda^2}$$

**Memoryless:** $P(X > s+t \mid X > s) = P(X > t)$

---

### 8.5 Gamma Distribution

$$X \sim \text{Gamma}(\alpha, \beta)$$

$$\boxed{f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}, \quad x > 0}$$

$$\Gamma(\alpha) = \int_0^\infty t^{\alpha-1} e^{-t}\, dt, \qquad \Gamma(n) = (n-1)!$$

$$\mathbb{E}[X] = \frac{\alpha}{\beta}, \qquad \text{Var}(X) = \frac{\alpha}{\beta^2}$$

**Special cases:**
- $\text{Gamma}(1, \lambda) = \text{Exp}(\lambda)$
- $\text{Gamma}(n/2, 1/2) = \chi^2(n)$

**Conjugate prior** for Poisson rate and Gaussian precision $\tau = 1/\sigma^2$.

---

### 8.6 Beta Distribution

$$X \sim \text{Beta}(\alpha, \beta), \quad X \in [0, 1]$$

$$\boxed{f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}}$$

$$B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

$$\mathbb{E}[X] = \frac{\alpha}{\alpha+\beta}, \qquad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**Conjugate prior** for Bernoulli/Binomial likelihood. Used in Bayesian A/B testing.

---

### 8.7 Dirichlet Distribution

$$\mathbf{X} \sim \text{Dir}(\boldsymbol{\alpha}), \quad \mathbf{X} \in \Delta^{K-1}$$

$$\boxed{f(\mathbf{x}) = \frac{\Gamma\!\left(\sum_k \alpha_k\right)}{\prod_k \Gamma(\alpha_k)} \prod_{k=1}^K x_k^{\alpha_k - 1}}$$

$$\mathbb{E}[X_k] = \frac{\alpha_k}{\alpha_0}, \quad \alpha_0 = \sum_k \alpha_k$$

$$\text{Var}(X_k) = \frac{\alpha_k(\alpha_0 - \alpha_k)}{\alpha_0^2(\alpha_0+1)}$$

**Multivariate generalization of Beta.** Conjugate prior for Categorical/Multinomial.

**Example (ML):** Topic models (LDA — Latent Dirichlet Allocation), Bayesian mixture models.

---

### 8.8 Chi-Squared Distribution

$$X \sim \chi^2(k)$$ — sum of $k$ squared standard normals: $X = \sum_{i=1}^k Z_i^2$, $Z_i \sim \mathcal{N}(0,1)$

$$\boxed{f(x) = \frac{x^{k/2-1}e^{-x/2}}{2^{k/2}\Gamma(k/2)}, \quad x > 0}$$

$$\mathbb{E}[X] = k, \qquad \text{Var}(X) = 2k$$

---

### 8.9 Student's t-Distribution

$$X \sim t(\nu)$$

$$\boxed{f(x) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1 + \frac{x^2}{\nu}\right)^{-(\nu+1)/2}}$$

$$\mathbb{E}[X] = 0\ (\nu > 1), \qquad \text{Var}(X) = \frac{\nu}{\nu-2}\ (\nu > 2)$$

Heavier tails than Gaussian. $t(\nu) \to \mathcal{N}(0,1)$ as $\nu \to \infty$.

**Example (DL):** Robust loss functions; t-SNE uses Student-$t$ in low-dimensional space.

---

### 8.10 Laplace Distribution

$$X \sim \text{Laplace}(\mu, b)$$

$$\boxed{f(x) = \frac{1}{2b} \exp\!\left(-\frac{|x - \mu|}{b}\right)}$$

$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = 2b^2$$

**Example (ML):** Lasso (L1 regularization) corresponds to a Laplace prior on weights. MAP estimation with Laplace prior equals L1 regularized MLE.

---

### 8.11 Cauchy Distribution

$$X \sim \text{Cauchy}(x_0, \gamma)$$

$$\boxed{f(x) = \frac{1}{\pi\gamma\left[1 + \left(\frac{x-x_0}{\gamma}\right)^2\right]}}$$

Mean and variance do not exist (undefined/infinite). Used in heavy-tailed modeling.

---

### 8.12 Logistic Distribution

$$X \sim \text{Logistic}(\mu, s)$$

$$\boxed{f(x) = \frac{e^{-(x-\mu)/s}}{s\left(1+e^{-(x-\mu)/s}\right)^2}}$$

$$F(x) = \frac{1}{1+e^{-(x-\mu)/s}} = \sigma\!\left(\frac{x-\mu}{s}\right)$$

CDF is the **sigmoid function** — the foundation of logistic regression.

$$\mathbb{E}[X] = \mu, \qquad \text{Var}(X) = \frac{\pi^2 s^2}{3}$$

---

## 9. Joint, Marginal & Conditional Distributions

### 9.1 Joint Distribution

**Discrete:** $p_{X,Y}(x, y) = P(X=x, Y=y)$

$$\sum_x \sum_y p_{X,Y}(x, y) = 1$$

**Continuous:** $f_{X,Y}(x, y) \geq 0$

$$\int_{-\infty}^\infty \int_{-\infty}^\infty f_{X,Y}(x, y)\, dx\, dy = 1$$

### 9.2 Marginal Distribution

**Discrete:**
$$p_X(x) = \sum_y p_{X,Y}(x, y)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^\infty f_{X,Y}(x, y)\, dy$$

### 9.3 Conditional Distribution

**Discrete:**
$$p_{Y|X}(y \mid x) = \frac{p_{X,Y}(x,y)}{p_X(x)}$$

**Continuous:**
$$f_{Y|X}(y \mid x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

### 9.4 Independence of RVs

$X$ and $Y$ are independent ($X \perp\!\!\!\perp Y$) iff:

$$\boxed{f_{X,Y}(x,y) = f_X(x)\, f_Y(y)}$$

Equivalently: $F_{X,Y}(x,y) = F_X(x) F_Y(y)$

### 9.5 Copulas

A **copula** $C: [0,1]^d \to [0,1]$ separates marginal structure from dependence structure:

$$F_{X,Y}(x,y) = C(F_X(x), F_Y(y))$$

**Sklar's Theorem:** Every multivariate distribution has a copula representation. Allows modeling complex dependencies beyond correlation.

---

## 10. Expectation

### 10.1 Definition

**Discrete:**
$$\boxed{\mathbb{E}[X] = \sum_x x\, p_X(x)}$$

**Continuous:**
$$\boxed{\mathbb{E}[X] = \int_{-\infty}^\infty x\, f_X(x)\, dx}$$

**Law of the Unconscious Statistician (LOTUS):**
$$\mathbb{E}[g(X)] = \int g(x)\, f_X(x)\, dx$$

### 10.2 Properties of Expectation

- **Linearity:** $\mathbb{E}[aX + bY + c] = a\mathbb{E}[X] + b\mathbb{E}[Y] + c$
- **Monotonicity:** $X \leq Y \implies \mathbb{E}[X] \leq \mathbb{E}[Y]$
- **Independence product:** $X \perp\!\!\!\perp Y \implies \mathbb{E}[XY] = \mathbb{E}[X]\mathbb{E}[Y]$
- **Tower property (Law of Total Expectation):**

$$\boxed{\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]}$$

More explicitly:
$$\mathbb{E}[X] = \sum_y \mathbb{E}[X \mid Y=y] P(Y=y)$$

**Example (ML):** Expected loss over data distribution:
$$\mathcal{L}(\theta) = \mathbb{E}_{(\mathbf{x},y) \sim p_{\text{data}}}[\ell(f_\theta(\mathbf{x}), y)]$$

### 10.3 Conditional Expectation

$$\mathbb{E}[X \mid Y = y] = \int x\, f_{X|Y}(x \mid y)\, dx$$

The quantity $\mathbb{E}[X \mid Y]$ is itself a **random variable** — a function of $Y$.

Properties:
- $\mathbb{E}[\mathbb{E}[X \mid Y]] = \mathbb{E}[X]$ (tower property)
- $\mathbb{E}[g(Y)X \mid Y] = g(Y)\mathbb{E}[X \mid Y]$ (taking out known factors)
- $X \perp\!\!\!\perp Y \implies \mathbb{E}[X \mid Y] = \mathbb{E}[X]$

### 10.4 Moments

**$k$-th moment:**
$$\mu_k = \mathbb{E}[X^k]$$

**$k$-th central moment:**
$$\tilde{\mu}_k = \mathbb{E}[(X - \mu)^k]$$

| Moment | Formula | Meaning |
|--------|---------|---------|
| Mean | $\mu = \mathbb{E}[X]$ | Location |
| Variance | $\sigma^2 = \mathbb{E}[(X-\mu)^2]$ | Spread |
| Skewness | $\gamma_1 = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^3\right]$ | Asymmetry |
| Kurtosis | $\gamma_2 = \mathbb{E}\!\left[\left(\frac{X-\mu}{\sigma}\right)^4\right] - 3$ | Tail heaviness |

---

## 11. Variance & Covariance

### 11.1 Variance

$$\boxed{\text{Var}(X) = \mathbb{E}[(X-\mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2}$$

$$\text{std}(X) = \sigma = \sqrt{\text{Var}(X)}$$

**Properties:**
- $\text{Var}(aX + b) = a^2 \text{Var}(X)$
- $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$
- $X \perp\!\!\!\perp Y \implies \text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$

**Law of Total Variance:**
$$\boxed{\text{Var}(X) = \mathbb{E}[\text{Var}(X \mid Y)] + \text{Var}(\mathbb{E}[X \mid Y])}$$

---

### 11.2 Covariance

$$\boxed{\text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \mathbb{E}[XY] - \mathbb{E}[X]\mathbb{E}[Y]}$$

**Properties:**
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ (symmetric)
- $\text{Cov}(aX + b, cY + d) = ac\,\text{Cov}(X, Y)$
- $X \perp\!\!\!\perp Y \implies \text{Cov}(X, Y) = 0$ (converse is NOT generally true)
- $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$

---

### 11.3 Pearson Correlation Coefficient

$$\boxed{\rho(X, Y) = \frac{\text{Cov}(X, Y)}{\text{std}(X)\,\text{std}(Y)} \in [-1, 1]}$$

| $\rho$ | Meaning |
|--------|---------|
| $+1$ | Perfect positive linear relationship |
| $0$ | No linear relationship |
| $-1$ | Perfect negative linear relationship |

> Note: $\rho = 0$ does not mean independent — only no linear correlation.

---

### 11.4 Covariance Matrix

For random vector $\mathbf{X} = (X_1, \ldots, X_d)^T$:

$$\boxed{\Sigma = \text{Cov}(\mathbf{X}) = \mathbb{E}[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T]}$$

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

**Properties:**
- Symmetric: $\Sigma = \Sigma^T$
- Positive semidefinite: $\Sigma \succeq 0$
- Diagonal entries are variances: $\Sigma_{ii} = \text{Var}(X_i)$
- For linear transform: $\text{Cov}(A\mathbf{X}) = A \Sigma A^T$

**Precision matrix:** $\Lambda = \Sigma^{-1}$

**Example (ML):** Covariance matrix is central to PCA, Gaussian discriminant analysis, Kalman filters, Gaussian Processes.

---

## 12. Moment Generating Functions & Characteristic Functions

### 12.1 Moment Generating Function (MGF)

$$\boxed{M_X(t) = \mathbb{E}[e^{tX}] = \sum_{k=0}^\infty \frac{t^k}{k!} \mathbb{E}[X^k]}$$

**Moments from MGF:**
$$\mathbb{E}[X^k] = M_X^{(k)}(0) = \left.\frac{d^k M_X}{dt^k}\right|_{t=0}$$

**Key property:** If $X \perp\!\!\!\perp Y$:
$$M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$$

**Example:** Sum of independent Gaussians is Gaussian — verified via MGFs.

### 12.2 Characteristic Function

$$\boxed{\varphi_X(t) = \mathbb{E}[e^{itX}] = \int e^{itx} f_X(x)\, dx}$$

Always exists (unlike MGF). Related to Fourier transform of PDF.

$$\mathbb{E}[X^k] = \frac{1}{i^k} \varphi_X^{(k)}(0)$$

### 12.3 Cumulants & Log-MGF

**Cumulant generating function (CGF):**
$$K_X(t) = \log M_X(t)$$

**Cumulants** $\kappa_n$:
$$K_X(t) = \sum_{n=1}^\infty \frac{\kappa_n t^n}{n!}$$

| Cumulant | Value |
|----------|-------|
| $\kappa_1$ | $\mathbb{E}[X]$ (mean) |
| $\kappa_2$ | $\text{Var}(X)$ |
| $\kappa_3$ | $\mathbb{E}[(X-\mu)^3]$ (skewness numerator) |
| $\kappa_4$ | Excess kurtosis numerator |

---

## 13. Information Theory & Entropy

### 13.1 Self-Information (Surprisal)

$$\boxed{I(x) = -\log P(X=x)}$$

In bits (base-2 log) or nats (natural log). Low probability event yields high information.

---

### 13.2 Shannon Entropy

$$\boxed{H(X) = -\sum_x p(x) \log p(x) = \mathbb{E}[-\log p(X)]}$$

**Continuous (Differential Entropy):**
$$h(X) = -\int f(x) \log f(x)\, dx$$

**Properties:**
- $H(X) \geq 0$ (for discrete; continuous can be negative)
- $H(X) \leq \log |\mathcal{X}|$ — maximized by uniform distribution
- $H(X) = 0$ iff $X$ is deterministic
- **Chain rule:** $H(X, Y) = H(X) + H(Y \mid X)$
- **Subadditivity:** $H(X, Y) \leq H(X) + H(Y)$

**Joint entropy:**
$$H(X, Y) = -\sum_{x,y} p(x,y) \log p(x,y)$$

**Conditional entropy:**
$$H(Y \mid X) = -\sum_{x,y} p(x,y) \log p(y \mid x) = \mathbb{E}_X[H(Y \mid X=x)]$$

**Entropy of Gaussian:** $h(\mathcal{N}(\mu,\sigma^2)) = \frac{1}{2}\log(2\pi e \sigma^2)$

**Example (ML):** Decision tree splitting criterion — maximize information gain (reduction in entropy).

---

### 13.3 Cross-Entropy

$$\boxed{H(p, q) = -\sum_x p(x) \log q(x) = -\mathbb{E}_p[\log q(X)]}$$

**Continuous:**
$$H(p, q) = -\int p(x) \log q(x)\, dx$$

**Relationship:** $H(p, q) = H(p) + D_{\text{KL}}(p \| q)$

**Example (ML):** The standard classification loss in DL. For one-hot target $y$ and model output $\hat{p}$:

$$L = -\sum_k y_k \log \hat{p}_k = -\log \hat{p}_{y_{\text{true}}}$$

---

### 13.4 KL Divergence (Relative Entropy)

$$\boxed{D_{\text{KL}}(p \| q) = \sum_x p(x) \log \frac{p(x)}{q(x)} = \mathbb{E}_p\!\left[\log \frac{p(X)}{q(X)}\right]}$$

**Continuous:**
$$D_{\text{KL}}(p \| q) = \int p(x) \log \frac{p(x)}{q(x)}\, dx$$

**Properties:**
- $D_{\text{KL}}(p \| q) \geq 0$ always (Gibbs' inequality)
- $D_{\text{KL}}(p \| q) = 0 \iff p = q$
- **Not symmetric:** $D_{\text{KL}}(p \| q) \neq D_{\text{KL}}(q \| p)$ in general
- Not a metric (violates triangle inequality)

**Two types of fitting:**
- $\min_q D_{\text{KL}}(p \| q)$: **mean-seeking** (zero-avoiding in $q$)
- $\min_q D_{\text{KL}}(q \| p)$: **mode-seeking** (zero-forcing in $q$, used in VI)

**KL divergence between two Gaussians:**
$$D_{\text{KL}}(\mathcal{N}_1 \| \mathcal{N}_2) = \frac{1}{2}\left[\log\frac{|\Sigma_2|}{|\Sigma_1|} - d + \text{tr}(\Sigma_2^{-1}\Sigma_1) + (\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)^T\Sigma_2^{-1}(\boldsymbol{\mu}_2-\boldsymbol{\mu}_1)\right]$$

**KL from $\mathcal{N}(\mu, \sigma^2)$ to $\mathcal{N}(0, 1)$** (used in VAE):
$$D_{\text{KL}}(\mathcal{N}(\mu,\sigma^2) \| \mathcal{N}(0,1)) = \frac{1}{2}\left(\mu^2 + \sigma^2 - \log\sigma^2 - 1\right)$$

**Example (DL — VAE):** ELBO = reconstruction loss minus KL divergence:
$$\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

---

### 13.5 Mutual Information

$$\boxed{I(X; Y) = \sum_{x,y} p(x,y) \log \frac{p(x,y)}{p(x)p(y)} = D_{\text{KL}}(p_{X,Y} \| p_X p_Y)}$$

$$I(X; Y) = H(X) - H(X \mid Y) = H(Y) - H(Y \mid X) = H(X) + H(Y) - H(X, Y)$$

**Properties:**
- $I(X; Y) \geq 0$
- $I(X; Y) = 0 \iff X \perp\!\!\!\perp Y$
- $I(X; Y) = I(Y; X)$ (symmetric)
- $I(X; Y) \leq \min(H(X), H(Y))$

**Conditional mutual information:**
$$I(X; Y \mid Z) = H(X \mid Z) - H(X \mid Y, Z)$$

**Data processing inequality:** For Markov chain $X \to Y \to Z$:
$$I(X; Z) \leq I(X; Y)$$

**Example (ML):** Feature selection — select features $X_i$ maximizing $I(X_i; Y)$.

---

### 13.6 Jensen-Shannon Divergence

$$\boxed{D_{\text{JS}}(p \| q) = \frac{1}{2} D_{\text{KL}}\!\left(p \,\Big\|\, \frac{p+q}{2}\right) + \frac{1}{2} D_{\text{KL}}\!\left(q \,\Big\|\, \frac{p+q}{2}\right)}$$

- Symmetric: $D_{\text{JS}}(p \| q) = D_{\text{JS}}(q \| p)$
- Bounded: $0 \leq D_{\text{JS}} \leq \log 2$
- $\sqrt{D_{\text{JS}}}$ is a metric

**Example (DL):** Original GAN objective minimizes JSD between real and generated distributions.

---

### 13.7 f-Divergences

General family:
$$D_f(p \| q) = \int q(x)\, f\!\left(\frac{p(x)}{q(x)}\right) dx$$

where $f$ is convex with $f(1) = 0$.

| Name | $f(t)$ |
|------|--------|
| KL: $p \| q$ | $t \log t$ |
| KL: $q \| p$ | $-\log t$ |
| Total variation | $\frac{1}{2}|t-1|$ |
| Hellinger | $(\sqrt{t}-1)^2$ |
| Chi-squared | $(t-1)^2$ |

---

## 14. Common Inequalities

### 14.1 Markov's Inequality

For non-negative $X$ and $a > 0$:

$$\boxed{P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}}$$

### 14.2 Chebyshev's Inequality

$$\boxed{P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}}$$

Equivalently: $P(|X - \mu| \geq \varepsilon) \leq \frac{\text{Var}(X)}{\varepsilon^2}$

**Example (ML):** At least $75\%$ of data lies within $2\sigma$ of the mean (regardless of distribution).

### 14.3 Jensen's Inequality

For **convex** function $\varphi$:

$$\boxed{\varphi(\mathbb{E}[X]) \leq \mathbb{E}[\varphi(X)]}$$

For **concave** $\varphi$ (e.g., $\log$): inequality reverses.

$$\log(\mathbb{E}[X]) \geq \mathbb{E}[\log X]$$

**Example (ML):** Derivation of ELBO in VAEs:

$$\log p(\mathbf{x}) = \log \mathbb{E}_{q(\mathbf{z})}\!\left[\frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right] \geq \mathbb{E}_{q(\mathbf{z})}\!\left[\log \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z})}\right] = \mathcal{L}_{\text{ELBO}}$$

### 14.4 Cauchy-Schwarz Inequality (Probabilistic)

$$(\mathbb{E}[XY])^2 \leq \mathbb{E}[X^2] \mathbb{E}[Y^2]$$

### 14.5 Hoeffding's Inequality

For independent bounded $X_i \in [a_i, b_i]$, sample mean $\bar{X}_n = \frac{1}{n}\sum_i X_i$:

$$\boxed{P(\bar{X}_n - \mathbb{E}[\bar{X}_n] \geq t) \leq \exp\!\left(-\frac{2n^2t^2}{\sum_i (b_i-a_i)^2}\right)}$$

**Example (ML):** Generalization bounds — probability training error deviates from true risk by more than $\varepsilon$ is exponentially small.

### 14.6 Gibbs' Inequality

$$-\sum_x p(x)\log p(x) \leq -\sum_x p(x)\log q(x)$$

Equivalently: $D_{\text{KL}}(p\|q) \geq 0$.

---

## 15. Limit Theorems

### 15.1 Law of Large Numbers (LLN)

Let $X_1, X_2, \ldots$ be i.i.d. with mean $\mu$.

**Weak LLN:**
$$\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

$$\forall \varepsilon > 0: \lim_{n\to\infty} P(|\bar{X}_n - \mu| > \varepsilon) = 0$$

**Strong LLN:**
$$P\!\left(\lim_{n\to\infty} \bar{X}_n = \mu\right) = 1$$

**Example (ML):** Empirical risk $\hat{\mathcal{L}}(\theta) = \frac{1}{n}\sum_i \ell_i(\theta)$ converges to true risk $\mathcal{L}(\theta)$ as $n \to \infty$. This justifies empirical risk minimization (ERM).

### 15.2 Central Limit Theorem (CLT)

For i.i.d. $X_i$ with mean $\mu$, variance $\sigma^2 < \infty$:

$$\boxed{\sqrt{n}\,\frac{\bar{X}_n - \mu}{\sigma} \xrightarrow{d} \mathcal{N}(0, 1)}$$

Equivalently: $\bar{X}_n \xrightarrow{d} \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$

**Example (ML):** Stochastic gradients are approximately Gaussian for large batch sizes. Noise in SGD has this structure.

### 15.3 Multivariate CLT

For i.i.d. random vectors $\mathbf{X}_i$ with mean $\boldsymbol{\mu}$, covariance $\Sigma$:

$$\sqrt{n}(\bar{\mathbf{X}}_n - \boldsymbol{\mu}) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \Sigma)$$

### 15.4 Delta Method

If $\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$ and $g'(\mu) \neq 0$:

$$\sqrt{n}(g(\bar{X}_n) - g(\mu)) \xrightarrow{d} \mathcal{N}(0, [g'(\mu)]^2 \sigma^2)$$

---

## 16. Bayesian Probability & Inference

### 16.1 Bayesian Framework

$$\underbrace{p(\theta \mid \mathcal{D})}_{\text{posterior}} = \frac{\underbrace{p(\mathcal{D} \mid \theta)}_{\text{likelihood}} \cdot \underbrace{p(\theta)}_{\text{prior}}}{\underbrace{p(\mathcal{D})}_{\text{evidence}}}$$

$$p(\theta \mid \mathcal{D}) \propto p(\mathcal{D} \mid \theta)\, p(\theta)$$

**Posterior predictive:**
$$p(\tilde{x} \mid \mathcal{D}) = \int p(\tilde{x} \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta$$

### 16.2 Conjugate Priors

A prior $p(\theta)$ is **conjugate** to likelihood $p(\mathcal{D} \mid \theta)$ if the posterior is in the same family as the prior.

| Likelihood | Conjugate Prior | Posterior |
|------------|-----------------|-----------|
| Bernoulli/Binomial | Beta$(\alpha, \beta)$ | Beta$(\alpha + n_1, \beta + n_0)$ |
| Poisson | Gamma$(\alpha, \beta)$ | Gamma$(\alpha + \sum x_i, \beta + n)$ |
| Gaussian ($\mu$ unknown) | Gaussian | Gaussian |
| Gaussian ($\sigma^2$ unknown) | Inverse-Gamma | Inverse-Gamma |
| Multinomial | Dirichlet | Dirichlet |
| Exponential | Gamma | Gamma |

**Example — Beta-Binomial conjugacy:**

Prior: $p \sim \text{Beta}(\alpha, \beta)$

Likelihood: $k$ successes in $n$ trials, i.e., $k \sim \text{Binomial}(n, p)$

Posterior:
$$p \mid k \sim \text{Beta}(\alpha + k,\ \beta + n - k)$$

### 16.3 Evidence Lower Bound (ELBO)

For any variational distribution $q(\mathbf{z})$:

$$\log p(\mathbf{x}) = \mathcal{L}_{\text{ELBO}} + D_{\text{KL}}(q(\mathbf{z}) \| p(\mathbf{z} \mid \mathbf{x}))$$

$$\log p(\mathbf{x}) \geq \mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q}[\log p(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q(\mathbf{z}) \| p(\mathbf{z}))$$

**Variational Inference** maximizes ELBO instead of computing the intractable posterior.

### 16.4 Bayesian Updating

Sequential Bayesian updating — each posterior becomes the new prior:

$$p(\theta \mid x_1) \propto p(x_1 \mid \theta) p(\theta)$$
$$p(\theta \mid x_1, x_2) \propto p(x_2 \mid \theta) p(\theta \mid x_1)$$
$$p(\theta \mid x_1, \ldots, x_n) \propto \prod_{i=1}^n p(x_i \mid \theta) \cdot p(\theta)$$

### 16.5 Bayesian Model Comparison

**Bayes factor** comparing models $M_1$ and $M_2$:

$$\text{BF}_{12} = \frac{p(\mathcal{D} \mid M_1)}{p(\mathcal{D} \mid M_2)}$$

$$p(\mathcal{D} \mid M) = \int p(\mathcal{D} \mid \theta, M)\, p(\theta \mid M)\, d\theta$$

Automatically penalizes complexity — Occam's Razor built in.

---

## 17. Probabilistic Graphical Models

### 17.1 Directed Graphical Models (Bayesian Networks)

A DAG where nodes are random variables, edges encode conditional dependencies.

**Joint distribution factorizes as:**
$$p(x_1, \ldots, x_n) = \prod_{i=1}^n p(x_i \mid \text{parents}(x_i))$$

**Example — Naive Bayes graph:** $Y \to X_1, Y \to X_2, \ldots, Y \to X_n$

$$p(Y, X_1, \ldots, X_n) = p(Y) \prod_{i=1}^n p(X_i \mid Y)$$

**d-separation:** $X \perp\!\!\!\perp Y \mid Z$ in the graph iff $Z$ d-separates $X$ and $Y$.

### 17.2 Undirected Graphical Models (Markov Random Fields)

**Joint distribution:**
$$p(\mathbf{x}) = \frac{1}{Z} \prod_{c \in \mathcal{C}} \psi_c(\mathbf{x}_c)$$

$$Z = \sum_\mathbf{x} \prod_{c \in \mathcal{C}} \psi_c(\mathbf{x}_c) \quad \text{(partition function)}$$

**Gibbs distribution:**
$$p(\mathbf{x}) = \frac{1}{Z} \exp\!\left(-\sum_c E_c(\mathbf{x}_c)\right) = \frac{1}{Z} e^{-E(\mathbf{x})}$$

where $E$ is the **energy function**.

**Example (DL):** Restricted Boltzmann Machine (RBM):
$$p(\mathbf{v}, \mathbf{h}) = \frac{1}{Z} e^{\mathbf{v}^T W \mathbf{h} + \mathbf{b}^T \mathbf{v} + \mathbf{c}^T \mathbf{h}}$$

### 17.3 Hidden Markov Models (HMM)

- Hidden states: $Z_1, Z_2, \ldots, Z_T$ (Markov chain)
- Observations: $X_1, X_2, \ldots, X_T$

**Markov property:** $P(Z_{t+1} \mid Z_1, \ldots, Z_t) = P(Z_{t+1} \mid Z_t)$

**Joint distribution:**
$$p(Z_{1:T}, X_{1:T}) = p(Z_1) \prod_{t=2}^T p(Z_t \mid Z_{t-1}) \prod_{t=1}^T p(X_t \mid Z_t)$$

**Transition matrix:** $A_{ij} = P(Z_{t+1}=j \mid Z_t=i)$

**Emission matrix:** $B_{ij} = P(X_t=j \mid Z_t=i)$

---

### 17.4 Latent Variable Models

Model with observed $\mathbf{x}$ and latent $\mathbf{z}$:

$$p(\mathbf{x}) = \int p(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z})\, d\mathbf{z}$$

**Gaussian Mixture Model:**
$$p(\mathbf{x}) = \sum_{k=1}^K \pi_k \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k)$$

$$p(z=k) = \pi_k, \qquad p(\mathbf{x} \mid z=k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma_k)$$

---

## 18. Exponential Family

### 18.1 Definition

A distribution belongs to the **exponential family** if its PDF/PMF can be written:

$$\boxed{p(x \mid \boldsymbol{\eta}) = h(x) \exp\!\left(\boldsymbol{\eta}^T \mathbf{T}(x) - A(\boldsymbol{\eta})\right)}$$

- $\boldsymbol{\eta}$: **natural parameters** (canonical parameters)
- $\mathbf{T}(x)$: **sufficient statistics**
- $h(x)$: base measure
- $A(\boldsymbol{\eta})$: **log-partition function** (cumulant function)

$$A(\boldsymbol{\eta}) = \log \int h(x) \exp(\boldsymbol{\eta}^T \mathbf{T}(x))\, dx$$

### 18.2 Members of Exponential Family

| Distribution | Natural param $\eta$ | Sufficient stat $T(x)$ | Log-partition $A(\eta)$ |
|-------------|--------|---------|-----------|
| Bernoulli$(p)$ | $\log\frac{p}{1-p}$ | $x$ | $\log(1+e^\eta)$ |
| Poisson$(\lambda)$ | $\log\lambda$ | $x$ | $e^\eta$ |
| Gaussian$(\mu,\sigma^2)$ | $(\mu/\sigma^2, -1/2\sigma^2)$ | $(x, x^2)$ | $-\eta_1^2/(4\eta_2) - \frac{1}{2}\log(-2\eta_2)$ |
| Gamma$(\alpha,\beta)$ | $(\alpha-1, -\beta)$ | $(\log x, x)$ | $\log\Gamma(\eta_1+1) - (\eta_1+1)\log(-\eta_2)$ |

### 18.3 Key Properties

**Moments from log-partition function:**
$$\nabla_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \mathbb{E}[\mathbf{T}(X)]$$

$$\nabla^2_{\boldsymbol{\eta}} A(\boldsymbol{\eta}) = \text{Cov}(\mathbf{T}(X))$$

**Maximum entropy:** Exponential family maximizes entropy subject to constraints $\mathbb{E}[\mathbf{T}(x)] = \boldsymbol{\mu}$.

**Sufficient statistics:** $\mathbf{T}(x_1, \ldots, x_n) = \sum_i \mathbf{T}(x_i)$ — data enters only through sufficient statistics.

**Conjugate priors always exist** for exponential family likelihoods.

---

## 19. Maximum Likelihood & MAP Estimation

### 19.1 Maximum Likelihood Estimation (MLE)

Given i.i.d. data $\mathcal{D} = \{x_1, \ldots, x_n\}$:

**Likelihood:**
$$\mathcal{L}(\theta) = p(\mathcal{D} \mid \theta) = \prod_{i=1}^n p(x_i \mid \theta)$$

**Log-likelihood:**
$$\ell(\theta) = \log \mathcal{L}(\theta) = \sum_{i=1}^n \log p(x_i \mid \theta)$$

**MLE:**
$$\boxed{\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\max_\theta \sum_{i=1}^n \log p(x_i \mid \theta)}$$

**Example — Gaussian MLE:**

$$\hat{\mu} = \frac{1}{n}\sum_i x_i, \qquad \hat{\sigma}^2 = \frac{1}{n}\sum_i (x_i - \hat{\mu})^2$$

(MLE for $\sigma^2$ is biased; unbiased estimate uses $n-1$.)

### 19.2 MLE as Minimizing KL Divergence

$$\hat{\theta}_{\text{MLE}} = \arg\min_\theta D_{\text{KL}}(\hat{p}_{\text{data}} \| p_\theta)$$

where $\hat{p}_{\text{data}}$ is the empirical distribution. MLE minimizes KL divergence from data to model.

### 19.3 Maximum A Posteriori (MAP) Estimation

$$\boxed{\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta \mid \mathcal{D}) = \arg\max_\theta \left[\sum_i \log p(x_i \mid \theta) + \log p(\theta)\right]}$$

**Relationship to regularization:**
- Gaussian prior $p(\theta) = \mathcal{N}(0, \lambda^{-1}I)$ gives $\hat{\theta}_{\text{MAP}} = $ Ridge regression (L2)
- Laplace prior $p(\theta) \propto e^{-\lambda\|\theta\|_1}$ gives $\hat{\theta}_{\text{MAP}} = $ Lasso (L1)

### 19.4 EM Algorithm (Expectation-Maximization)

For models with latent variables — maximizes $\log p(\mathbf{x} \mid \theta)$.

**E-step:** Compute expected complete-data log-likelihood:
$$Q(\theta, \theta^{\text{old}}) = \mathbb{E}_{\mathbf{z} \mid \mathbf{x}, \theta^{\text{old}}}[\log p(\mathbf{x}, \mathbf{z} \mid \theta)]$$

**M-step:** Maximize:
$$\theta^{\text{new}} = \arg\max_\theta Q(\theta, \theta^{\text{old}})$$

**Convergence guarantee:** Each iteration does not decrease $\log p(\mathbf{x} \mid \theta)$:

$$\log p(\mathbf{x} \mid \theta^{\text{new}}) \geq \log p(\mathbf{x} \mid \theta^{\text{old}})$$

**Example (ML):** EM for Gaussian Mixture Models:

E-step — compute responsibilities:
$$r_{ik} = \frac{\pi_k \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_k, \Sigma_k)}{\sum_j \pi_j \mathcal{N}(\mathbf{x}_i \mid \boldsymbol{\mu}_j, \Sigma_j)}$$

M-step — update parameters:

$$\pi_k = \frac{1}{n}\sum_i r_{ik}, \quad \boldsymbol{\mu}_k = \frac{\sum_i r_{ik} \mathbf{x}_i}{\sum_i r_{ik}}, \quad \Sigma_k = \frac{\sum_i r_{ik}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T}{\sum_i r_{ik}}$$

---

## 20. Probabilistic Models in ML/DL

### 20.1 Logistic Regression

$$P(y=1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x} + b) = \frac{1}{1+e^{-(\mathbf{w}^T\mathbf{x}+b)}}$$

$$P(y \mid \mathbf{x}) = \sigma(\mathbf{w}^T\mathbf{x})^y (1 - \sigma(\mathbf{w}^T\mathbf{x}))^{1-y}$$

**MLE** equals minimizing cross-entropy:
$$\ell(\mathbf{w}) = -\sum_i [y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)]$$

---

### 20.2 Softmax and Multi-class

$$P(y=k \mid \mathbf{x}) = \frac{e^{\mathbf{w}_k^T\mathbf{x}}}{\sum_{j=1}^K e^{\mathbf{w}_j^T\mathbf{x}}} = \text{softmax}(\mathbf{z})_k$$

**Log-sum-exp trick** (numerical stability):
$$\log\sum_j e^{z_j} = z_{\max} + \log\sum_j e^{z_j - z_{\max}}$$

**Cross-entropy loss:**
$$L = -\sum_i \sum_k y_{ik} \log \hat{p}_{ik} = -\sum_i \log \hat{p}_{i,y_i}$$

---

### 20.3 Gaussian Discriminant Analysis (GDA)

**Class conditional:**
$$p(\mathbf{x} \mid y=k) = \mathcal{N}(\mathbf{x} \mid \boldsymbol{\mu}_k, \Sigma)$$

**Prior:** $P(y=k) = \pi_k$

**LDA** (shared $\Sigma$) — log posterior ratio gives linear boundary:

$$\log\frac{P(y=1 \mid \mathbf{x})}{P(y=0 \mid \mathbf{x})} = (\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0)^T\Sigma^{-1}\mathbf{x} - \frac{1}{2}(\boldsymbol{\mu}_1^T\Sigma^{-1}\boldsymbol{\mu}_1 - \boldsymbol{\mu}_0^T\Sigma^{-1}\boldsymbol{\mu}_0) + \log\frac{\pi_1}{\pi_0}$$

**QDA** (class-specific $\Sigma_k$) gives a quadratic boundary.

---

### 20.4 Gaussian Process (GP)

A **Gaussian Process** is a distribution over functions:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}),\, k(\mathbf{x}, \mathbf{x}'))$$

- $m(\mathbf{x}) = \mathbb{E}[f(\mathbf{x})]$: mean function
- $k(\mathbf{x}, \mathbf{x}') = \text{Cov}(f(\mathbf{x}), f(\mathbf{x}'))$: covariance (kernel) function

**Any finite collection is jointly Gaussian:**
$$\mathbf{f} = [f(\mathbf{x}_1), \ldots, f(\mathbf{x}_n)]^T \sim \mathcal{N}(\mathbf{m}, K)$$

**GP Regression posterior** (noise model $y = f(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0,\sigma_n^2)$):

$$\bar{f}_* = \mathbf{k}_*^T (K + \sigma_n^2 I)^{-1} \mathbf{y}$$

$$\text{Var}(f_*) = k(\mathbf{x}_*, \mathbf{x}_*) - \mathbf{k}_*^T(K + \sigma_n^2 I)^{-1}\mathbf{k}_*$$

**Common kernels:**

| Kernel | Formula |
|--------|---------|
| RBF / Squared Exp. | $k(\mathbf{x},\mathbf{x}') = \sigma^2\exp\!\left(-\frac{\|\mathbf{x}-\mathbf{x}'\|^2}{2\ell^2}\right)$ |
| Matern | $k(\mathbf{x},\mathbf{x}') = \frac{2^{1-\nu}}{\Gamma(\nu)}\left(\frac{\sqrt{2\nu}r}{\ell}\right)^\nu K_\nu\!\left(\frac{\sqrt{2\nu}r}{\ell}\right)$ |
| Periodic | $k(\mathbf{x},\mathbf{x}') = \exp\!\left(-\frac{2\sin^2(\pi|\mathbf{x}-\mathbf{x}'|/p)}{\ell^2}\right)$ |
| Linear | $k(\mathbf{x},\mathbf{x}') = \sigma_b^2 + \sigma_v^2(\mathbf{x}-c)(\mathbf{x}'-c)$ |

---

### 20.5 Variational Autoencoder (VAE)

**Generative model:**
$$p(\mathbf{x}, \mathbf{z}) = p(\mathbf{x} \mid \mathbf{z})\, p(\mathbf{z}), \quad p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, I)$$

**Inference model (encoder):**
$$q_\phi(\mathbf{z} \mid \mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}),\, \text{diag}(\boldsymbol{\sigma}^2_\phi(\mathbf{x})))$$

**ELBO (objective to maximize):**
$$\mathcal{L}_{\text{VAE}}(\theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x} \mid \mathbf{z})] - D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \| p(\mathbf{z}))$$

**Reparameterization trick** (enables backprop through sampling):
$$\mathbf{z} = \boldsymbol{\mu}_\phi(\mathbf{x}) + \boldsymbol{\sigma}_\phi(\mathbf{x}) \odot \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, I)$$

**KL term closed form for Gaussian encoder:**
$$D_{\text{KL}}(q_\phi \| p) = \frac{1}{2}\sum_j \left(\mu_j^2 + \sigma_j^2 - \log\sigma_j^2 - 1\right)$$

---

### 20.6 Generative Adversarial Networks (GAN)

**Minimax game:**
$$\min_G \max_D\, \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))]$$

**Optimal discriminator:** $D^*(\mathbf{x}) = \frac{p_{\text{data}}(\mathbf{x})}{p_{\text{data}}(\mathbf{x}) + p_g(\mathbf{x})}$

**GAN objective equals JSD:**
$$V(D^*, G) = 2\, D_{\text{JS}}(p_{\text{data}} \| p_g) - \log 4$$

---

### 20.7 Normalizing Flows

**Change of variables formula:**
$$\log p_X(\mathbf{x}) = \log p_Z(f(\mathbf{x})) + \log\left|\det\!\left(\frac{\partial f}{\partial \mathbf{x}}\right)\right|$$

A sequence of invertible transformations $f = f_K \circ \cdots \circ f_1$:

$$\log p(\mathbf{x}) = \log p(\mathbf{z}_0) - \sum_{k=1}^K \log\left|\det J_k\right|$$

where $\mathbf{z}_k = f_k(\mathbf{z}_{k-1})$ and $J_k$ is the Jacobian of $f_k$.

**Example:** RealNVP, Glow, Neural ODEs all use this log-det-Jacobian formula.

---

### 20.8 Probabilistic Interpretation of Regularization

| Regularization | Probabilistic Interpretation |
|----------------|------------------------------|
| L2 (Ridge) $\lambda\|\mathbf{w}\|_2^2$ | MAP with Gaussian prior $\mathcal{N}(0, \frac{1}{2\lambda}I)$ |
| L1 (Lasso) $\lambda\|\mathbf{w}\|_1$ | MAP with Laplace prior $\text{Lap}(0, \frac{1}{2\lambda})$ |
| Dropout | Approximate Bayesian inference |
| Weight decay | Gaussian prior on weights |

---

### 20.9 Score Function & Fisher Information

**Score function:**
$$s(\theta) = \nabla_\theta \log p(x \mid \theta)$$

**Fisher information:**
$$\mathcal{I}(\theta) = \mathbb{E}_{x \sim p(\cdot|\theta)}\!\left[s(\theta) s(\theta)^T\right] = -\mathbb{E}\!\left[\nabla^2_\theta \log p(x \mid \theta)\right]$$

**Cramer-Rao lower bound:** Variance of any unbiased estimator $\hat{\theta}$:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{\mathcal{I}(\theta)}$$

**Example (DL):** Natural gradient descent uses Fisher information as preconditioner:
$$\theta \leftarrow \theta + \alpha\, \mathcal{I}(\theta)^{-1} \nabla_\theta \mathcal{L}$$

---

### 20.10 Monte Carlo Methods

**Monte Carlo estimation:**
$$\mathbb{E}_{p}[f(X)] \approx \frac{1}{N}\sum_{i=1}^N f(x_i), \quad x_i \sim p$$

**Error:** $O(N^{-1/2})$ — independent of dimension (key advantage over grid methods).

**Importance sampling:**
$$\mathbb{E}_p[f(X)] = \mathbb{E}_q\!\left[f(X)\frac{p(X)}{q(X)}\right] \approx \frac{1}{N}\sum_{i=1}^N f(x_i)\frac{p(x_i)}{q(x_i)}, \quad x_i \sim q$$

Importance weights: $w_i = \frac{p(x_i)}{q(x_i)}$

**Metropolis-Hastings acceptance ratio:**
$$\alpha = \min\!\left(1,\, \frac{p(x') q(x \mid x')}{p(x) q(x' \mid x)}\right)$$

**Example (DL):** MC Dropout — run network $T$ times with dropout at test time:
$$p(y \mid \mathbf{x}) \approx \frac{1}{T}\sum_{t=1}^T p(y \mid \mathbf{x}, \hat{\mathbf{W}}_t)$$

Predictive uncertainty:
$$\text{Var}(y) \approx \frac{1}{T}\sum_t \hat{y}_t^2 - \bar{y}^2 + \frac{1}{T}\sum_t \hat{\sigma}_t^2$$

---

## Quick Reference — Master Formula Sheet

$$\boxed{P(B \mid A) = \frac{P(A \mid B) P(B)}{P(A)}} \quad \text{Bayes' Theorem}$$

$$\boxed{P(A) = \sum_i P(A \mid B_i) P(B_i)} \quad \text{Total Probability}$$

$$\boxed{\mathbb{E}[X] = \int x\, f(x)\, dx} \quad \text{Expectation}$$

$$\boxed{\text{Var}(X) = \mathbb{E}[X^2] - (\mathbb{E}[X])^2} \quad \text{Variance shortcut}$$

$$\boxed{H(X) = -\sum_x p(x)\log p(x)} \quad \text{Entropy}$$

$$\boxed{D_{\text{KL}}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \geq 0} \quad \text{KL Divergence}$$

$$\boxed{H(p,q) = H(p) + D_{\text{KL}}(p\|q)} \quad \text{Cross-Entropy}$$

$$\boxed{I(X;Y) = H(X) - H(X \mid Y)} \quad \text{Mutual Information}$$

$$\boxed{\mathbb{E}[X] = \mathbb{E}[\mathbb{E}[X \mid Y]]} \quad \text{Tower Property}$$

$$\boxed{\text{Var}(X) = \mathbb{E}[\text{Var}(X|Y)] + \text{Var}(\mathbb{E}[X|Y])} \quad \text{Total Variance}$$

$$\boxed{\hat{\theta}_{\text{MLE}} = \arg\max_\theta \sum_i \log p(x_i \mid \theta)} \quad \text{MLE}$$

$$\boxed{f_Y(y) = f_X(g^{-1}(y))\left|\frac{d}{dy}g^{-1}(y)\right|} \quad \text{Change of Variables}$$

$$\boxed{\log p(\mathbf{x}) \geq \mathbb{E}_q[\log p(\mathbf{x},\mathbf{z})] - \mathbb{E}_q[\log q(\mathbf{z})]} \quad \text{ELBO}$$

---

## Connections Map

```
Sample Space Omega
      |
  Events --> Probability Axioms --> Basic Rules
      |
Conditional Probability --> Bayes' Theorem --> Bayesian Inference
      |                          |
  Independence              Prior x Likelihood ~ Posterior
      |                          |
Random Variables          Conjugate Priors / ELBO / VI
   /        \                    |
Discrete  Continuous        EM Algorithm
   |          |
PMF/CDF     PDF/CDF
   |          |
Expectation / Variance / Covariance / Moments
      |
  MGF / Characteristic Functions / Cumulants
      |
Information Theory: Entropy --> KL --> Cross-Entropy --> Mutual Info
      |
Inequalities: Markov, Chebyshev, Jensen, Hoeffding, Cauchy-Schwarz
      |
Limit Theorems: LLN --> CLT --> Delta Method
      |
Exponential Family --> MLE / MAP --> Regularization <--> Priors
      |
Probabilistic Models:
  Logistic Regression --> GDA --> GP --> VAE --> GAN --> Flows
```

---

*Every formula in these notes maps directly to a concrete concept used in training, evaluating, or designing ML/DL models. Probability is not just background math — it IS the framework of machine learning.*
