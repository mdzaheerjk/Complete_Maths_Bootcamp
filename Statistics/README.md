# 📊 Statistics for Machine Learning & Deep Learning
### Complete Theory & Formulas — From Absolute Beginner to Advanced

> **Philosophy of this document:**  
> Every formula has a name. Every name has intuition. Every concept connects to ML/DL.  
> No code. Pure mathematics, theory, and worked examples.

---

## 📚 Table of Contents

1. [Foundations — Types of Data & Measurement](#1-foundations--types-of-data--measurement)
2. [Descriptive Statistics — Measures of Central Tendency](#2-descriptive-statistics--measures-of-central-tendency)
3. [Descriptive Statistics — Measures of Spread](#3-descriptive-statistics--measures-of-spread)
4. [Shape of Distributions](#4-shape-of-distributions)
5. [Probability Theory Fundamentals](#5-probability-theory-fundamentals)
6. [Conditional Probability & Bayes' Theorem](#6-conditional-probability--bayes-theorem)
7. [Random Variables](#7-random-variables)
8. [Probability Distributions — Discrete](#8-probability-distributions--discrete)
9. [Probability Distributions — Continuous](#9-probability-distributions--continuous)
10. [Joint, Marginal & Conditional Distributions](#10-joint-marginal--conditional-distributions)
11. [Expectation, Variance & Covariance — Deep Dive](#11-expectation-variance--covariance--deep-dive)
12. [The Normal Distribution — Master Reference](#12-the-normal-distribution--master-reference)
13. [Sampling Theory & the Central Limit Theorem](#13-sampling-theory--the-central-limit-theorem)
14. [Estimation Theory](#14-estimation-theory)
15. [Hypothesis Testing](#15-hypothesis-testing)
16. [Statistical Tests Reference](#16-statistical-tests-reference)
17. [Correlation & Dependence](#17-correlation--dependence)
18. [Linear Regression — Statistical Foundation](#18-linear-regression--statistical-foundation)
19. [Information Theory](#19-information-theory)
20. [Bayesian Statistics](#20-bayesian-statistics)
21. [Matrix Statistics & Multivariate Analysis](#21-matrix-statistics--multivariate-analysis)
22. [Resampling Methods](#22-resampling-methods)
23. [Statistics in Deep Learning — Direct Connections](#23-statistics-in-deep-learning--direct-connections)
24. [Statistical Learning Theory](#24-statistical-learning-theory)
25. [Master Formula Reference Sheet](#25-master-formula-reference-sheet)

---

## 1. Foundations — Types of Data & Measurement

### Scales of Measurement

| Scale | Description | Operations | ML Example |
|---|---|---|---|
| **Nominal** | Categories, no order | =, ≠ | Dog/Cat/Bird labels |
| **Ordinal** | Ordered categories | =, ≠, <, > | Star ratings 1–5 |
| **Interval** | Equal spacing, no true zero | +, −, =, <, > | Temperature °C |
| **Ratio** | Equal spacing + true zero | +, −, ×, ÷ | Age, height, income |

> **ML Implication:** Nominal data → one-hot encode. Ordinal → label encode with care. Interval/Ratio → normalize/standardize.

### Population vs Sample

| Symbol | Meaning | Population | Sample |
|---|---|---|---|
| Size | Number of observations | $N$ | $n$ |
| Mean | Average | $\mu$ | $\bar{x}$ |
| Variance | Spread | $\sigma^2$ | $s^2$ |
| Std Dev | Square root of variance | $\sigma$ | $s$ |
| Proportion | Fraction with property | $p$ | $\hat{p}$ |

> **Key Insight:** In ML, your training dataset is a **sample** from the true data-generating distribution (the **population**). Generalization error arises because $\bar{x} \neq \mu$.

---

## 2. Descriptive Statistics — Measures of Central Tendency

### Arithmetic Mean

$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

**Population mean:**

$$\mu = \frac{1}{N} \sum_{i=1}^{N} x_i$$

**Example:** Dataset = {2, 4, 4, 4, 5, 5, 7, 9}

$$\bar{x} = \frac{2+4+4+4+5+5+7+9}{8} = \frac{40}{8} = 5$$

**ML Use:** Feature means used in standardization. The mean of a loss function guides gradient descent.

### Weighted Mean

$$\bar{x}_w = \frac{\sum_{i=1}^{n} w_i x_i}{\sum_{i=1}^{n} w_i}$$

**ML Use:** Class-weighted loss functions. Weighted accuracy in imbalanced datasets.

### Geometric Mean

$$G = \left(\prod_{i=1}^{n} x_i\right)^{1/n} = \exp\!\left(\frac{1}{n}\sum_{i=1}^{n} \ln x_i\right)$$

**ML Use:** Averaging growth rates, log-scale metrics. Used in computing average BLEU scores.

### Harmonic Mean

$$H = \frac{n}{\sum_{i=1}^{n} \frac{1}{x_i}}$$

**ML Use:** The **F1 score** is the harmonic mean of Precision and Recall:

$$F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} = H(\text{Precision}, \text{Recall})$$

### Median

The **middle value** when data is sorted. For $n$ observations:

$$\text{Median} = \begin{cases} x_{(n+1)/2} & \text{if } n \text{ is odd} \\ \dfrac{x_{n/2} + x_{n/2+1}}{2} & \text{if } n \text{ is even} \end{cases}$$

**Example:** {2, 4, 4, **4**, 5, 5, 7, 9} → Median = (4+5)/2 = **4.5**

**ML Use:** Robust to outliers. Used in Huber loss, median absolute deviation.

### Mode

The value that appears **most frequently**.

**Example:** {2, 4, **4**, **4**, 5, 5, 7, 9} → Mode = **4**

**ML Use:** Categorical target in classification. Imputing missing categorical values.

### Relationship Between Mean, Median, Mode

$$\text{Mean} - \text{Mode} \approx 3(\text{Mean} - \text{Median}) \quad \text{(Pearson's empirical rule)}$$

| Distribution Shape | Relationship |
|---|---|
| Symmetric | Mean = Median = Mode |
| Right-skewed (positive) | Mode < Median < Mean |
| Left-skewed (negative) | Mean < Median < Mode |

---

## 3. Descriptive Statistics — Measures of Spread

### Range

$$\text{Range} = x_{\max} - x_{\min}$$

### Interquartile Range (IQR)

$$\text{IQR} = Q_3 - Q_1$$

where $Q_1$ = 25th percentile, $Q_3$ = 75th percentile.

**Outlier Detection Rule:**
$$\text{Lower fence} = Q_1 - 1.5 \times \text{IQR}$$
$$\text{Upper fence} = Q_3 + 1.5 \times \text{IQR}$$

Any value outside these fences is a **potential outlier**.

### Variance

**Population variance:**
$$\sigma^2 = \frac{1}{N} \sum_{i=1}^{N} (x_i - \mu)^2$$

**Sample variance (Bessel's correction):**
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$

> **Why $n-1$?** Using $n$ underestimates the true variance (biased estimator). Dividing by $n-1$ makes $s^2$ an **unbiased estimator** of $\sigma^2$. The $-1$ accounts for one degree of freedom lost by estimating $\mu$ with $\bar{x}$.

**Computational formula (faster, numerically equivalent):**
$$\sigma^2 = \frac{\sum x_i^2}{N} - \mu^2 = E[X^2] - (E[X])^2$$

**Example:** Dataset = {2, 4, 4, 4, 5, 5, 7, 9}, $\bar{x} = 5$

$$s^2 = \frac{(2-5)^2+(4-5)^2+(4-5)^2+(4-5)^2+(5-5)^2+(5-5)^2+(7-5)^2+(9-5)^2}{8-1}$$
$$= \frac{9+1+1+1+0+0+4+16}{7} = \frac{32}{7} \approx 4.57$$

### Standard Deviation

$$\sigma = \sqrt{\sigma^2} \qquad s = \sqrt{s^2}$$

**Same units as the data** (unlike variance). In example above: $s = \sqrt{4.57} \approx 2.14$

### Coefficient of Variation (CV)

$$CV = \frac{\sigma}{\mu} \times 100\%$$

**ML Use:** Compare spread across features with different units. High CV → feature may need scaling.

### Mean Absolute Deviation (MAD)

$$\text{MAD} = \frac{1}{n} \sum_{i=1}^{n} |x_i - \bar{x}|$$

**Robust MAD (using median):**
$$\text{MAD}_{\text{robust}} = \text{median}\left(|x_i - \text{median}(x)|\right)$$

**ML Use:** Huber loss is inspired by MAD. Robust to outliers unlike variance.

### Standard Error of the Mean (SEM)

$$SE = \frac{s}{\sqrt{n}} = \frac{\sigma}{\sqrt{n}}$$

**Interpretation:** How much the sample mean $\bar{x}$ varies from sample to sample.

**ML Use:** Error bars on validation metrics. As dataset size $n$ grows, SEM shrinks → more reliable estimates.

---

## 4. Shape of Distributions

### Percentiles and Quantiles

The $p$-th percentile is the value below which $p\%$ of observations fall.

$$x_p = x_{\lfloor np/100 \rfloor + 1}$$

**Special Quantiles:**
| Name | Percentile |
|---|---|
| Median | 50th |
| Quartiles Q1, Q2, Q3 | 25th, 50th, 75th |
| Deciles | 10th, 20th, ..., 90th |

### Skewness

Measures the **asymmetry** of the distribution.

**Population skewness:**
$$\gamma_1 = \frac{\mu_3}{\sigma^3} = \frac{E[(X-\mu)^3]}{\sigma^3}$$

**Sample skewness (Fisher's):**
$$g_1 = \frac{n}{(n-1)(n-2)} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{s}\right)^3$$

| Value | Interpretation |
|---|---|
| $\gamma_1 = 0$ | Symmetric |
| $\gamma_1 > 0$ | Right-skewed (long right tail) |
| $\gamma_1 < 0$ | Left-skewed (long left tail) |
| $|\gamma_1| > 1$ | Highly skewed |

**ML Use:** Skewed features → apply log transform or Box-Cox before training. Loss landscapes can be skewed.

### Kurtosis

Measures the **tailedness** of the distribution (how extreme the outliers are).

$$\gamma_2 = \frac{\mu_4}{\sigma^4} = \frac{E[(X-\mu)^4]}{\sigma^4}$$

**Excess kurtosis (relative to Normal):**
$$\kappa = \gamma_2 - 3$$

| Value | Distribution Type | Interpretation |
|---|---|---|
| $\kappa = 0$ | Mesokurtic | Normal distribution |
| $\kappa > 0$ | Leptokurtic | Heavy tails, sharp peak |
| $\kappa < 0$ | Platykurtic | Light tails, flat peak |

**ML Use:** Heavy-tailed loss distributions → use robust loss functions. Weight distributions in neural networks should be ~mesokurtic.

### Central Moments

The $k$-th central moment:

$$\mu_k = E[(X - \mu)^k] = \frac{1}{N}\sum_{i=1}^N (x_i - \mu)^k$$

| Moment | Formula | Name |
|---|---|---|
| 1st raw | $E[X]$ | Mean |
| 2nd central | $E[(X-\mu)^2]$ | Variance |
| 3rd central | $E[(X-\mu)^3]$ | Related to Skewness |
| 4th central | $E[(X-\mu)^4]$ | Related to Kurtosis |

---

## 5. Probability Theory Fundamentals

### Sample Space & Events

- **Sample space** $\Omega$: set of all possible outcomes
- **Event** $A \subseteq \Omega$: subset of outcomes
- **Complement** $A^c$ or $\bar{A}$: all outcomes NOT in $A$

### Axioms of Probability (Kolmogorov)

$$P(\Omega) = 1 \qquad \text{(certainty)}$$
$$P(A) \geq 0 \quad \forall A \qquad \text{(non-negativity)}$$
$$P(A \cup B) = P(A) + P(B) \quad \text{if } A \cap B = \emptyset \quad \text{(additivity)}$$

### Derived Rules

**Complement rule:**
$$P(A^c) = 1 - P(A)$$

**Addition rule (general):**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$

**Inclusion-exclusion (three events):**
$$P(A \cup B \cup C) = P(A)+P(B)+P(C) - P(A \cap B) - P(A \cap C) - P(B \cap C) + P(A \cap B \cap C)$$

**Multiplication rule:**
$$P(A \cap B) = P(A) \cdot P(B|A) = P(B) \cdot P(A|B)$$

**Independence:** Events $A$ and $B$ are independent if:
$$P(A \cap B) = P(A) \cdot P(B) \iff P(A|B) = P(A)$$

**Example:** In a spam classifier, $P(\text{spam}) = 0.3$, $P(\text{contains "free"} | \text{spam}) = 0.8$, $P(\text{contains "free"} | \text{ham}) = 0.1$.

$$P(\text{contains "free"}) = 0.8 \times 0.3 + 0.1 \times 0.7 = 0.24 + 0.07 = 0.31$$

### Law of Total Probability

If $\{B_1, B_2, \ldots, B_n\}$ is a partition of $\Omega$:

$$P(A) = \sum_{i=1}^{n} P(A | B_i) \cdot P(B_i)$$

**Example:** $P(\text{error}) = P(\text{error}|\text{train}) P(\text{train}) + P(\text{error}|\text{val}) P(\text{val}) + P(\text{error}|\text{test}) P(\text{test})$

---

## 6. Conditional Probability & Bayes' Theorem

### Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

**Read:** "Probability of A **given** B has occurred."

**Example:** $P(\text{spam} \cap \text{"free"}) = 0.24$, $P(\text{"free"}) = 0.31$

$$P(\text{spam} | \text{"free"}) = \frac{0.24}{0.31} \approx 0.774$$

### Bayes' Theorem

$$\boxed{P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)}}$$

| Term | Name | ML Meaning |
|---|---|---|
| $P(H)$ | **Prior** | Belief before seeing data |
| $P(E\|H)$ | **Likelihood** | Probability of evidence given hypothesis |
| $P(E)$ | **Marginal likelihood** (evidence) | Normalizing constant |
| $P(H\|E)$ | **Posterior** | Updated belief after seeing data |

**Expanded form using total probability:**
$$P(H|E) = \frac{P(E|H) \cdot P(H)}{\sum_j P(E|H_j) \cdot P(H_j)}$$

**Posterior ∝ Likelihood × Prior:**
$$P(H|E) \propto P(E|H) \cdot P(H)$$

**Example — Medical Test:**
- Disease prevalence: $P(D) = 0.01$
- Test sensitivity: $P(+|D) = 0.99$
- Test specificity: $P(-|\bar{D}) = 0.95 \Rightarrow P(+|\bar{D}) = 0.05$

$$P(D|+) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = \frac{0.0099}{0.0099 + 0.0495} = \frac{0.0099}{0.0594} \approx 0.167$$

> **Insight:** Even with 99% accuracy test, only 16.7% chance of disease if positive. This is why base rates (priors) matter enormously in ML — class imbalance is a base rate problem.

### Naive Bayes Classifier

For features $x_1, x_2, \ldots, x_n$ and class $C_k$:

$$P(C_k | x_1, \ldots, x_n) \propto P(C_k) \prod_{i=1}^{n} P(x_i | C_k)$$

The "naive" assumption: features are **conditionally independent** given the class.

$$\hat{y} = \arg\max_{k} P(C_k) \prod_{i=1}^{n} P(x_i | C_k)$$

---

## 7. Random Variables

### Discrete vs Continuous

| Property | Discrete RV | Continuous RV |
|---|---|---|
| Values | Countable set | Uncountable interval |
| Probability function | PMF: $P(X=x)$ | PDF: $f(x)$, where $P(X=x)=0$ |
| Sum/Integral | $\sum_x P(X=x) = 1$ | $\int_{-\infty}^{\infty} f(x)\,dx = 1$ |
| CDF | $F(x) = \sum_{t \leq x} P(X=t)$ | $F(x) = \int_{-\infty}^{x} f(t)\,dt$ |

### Cumulative Distribution Function (CDF)

$$F(x) = P(X \leq x)$$

**Properties:**
- $F(-\infty) = 0$, $F(+\infty) = 1$
- $F$ is non-decreasing
- $P(a < X \leq b) = F(b) - F(a)$

**Relationship PDF ↔ CDF:**
$$f(x) = \frac{d}{dx} F(x) \qquad F(x) = \int_{-\infty}^{x} f(t)\,dt$$

### Expectation (Expected Value)

**Discrete:**
$$E[X] = \sum_x x \cdot P(X = x)$$

**Continuous:**
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\,dx$$

**Linearity of expectation (always true, even for dependent variables):**
$$E[aX + bY + c] = aE[X] + bE[Y] + c$$

### Variance of a Random Variable

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

**Properties:**
$$\text{Var}(aX + b) = a^2 \text{Var}(X)$$
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \quad \text{if } X \perp Y$$

---

## 8. Probability Distributions — Discrete

### Bernoulli Distribution

Models a **single trial** with success probability $p$.

$$X \sim \text{Bernoulli}(p)$$

$$P(X = x) = p^x (1-p)^{1-x}, \quad x \in \{0, 1\}$$

$$E[X] = p \qquad \text{Var}(X) = p(1-p)$$

**ML Use:** Binary classification output. Single neuron with sigmoid activation.

### Binomial Distribution

$n$ **independent** Bernoulli trials, each with probability $p$.

$$X \sim \text{Binomial}(n, p)$$

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

$$E[X] = np \qquad \text{Var}(X) = np(1-p)$$

**Example:** 10 test samples, model accuracy 0.8. P(exactly 8 correct)?

$$P(X=8) = \binom{10}{8}(0.8)^8(0.2)^2 = 45 \times 0.1678 \times 0.04 \approx 0.302$$

### Categorical Distribution

Generalization of Bernoulli to $K$ classes.

$$P(X = k) = p_k, \quad \sum_{k=1}^{K} p_k = 1$$

**ML Use:** Output layer of multiclass classifier. Softmax outputs are parameters of Categorical distribution.

### Multinomial Distribution

$n$ trials, $K$ possible outcomes with probabilities $p_1, \ldots, p_K$.

$$P(X_1=k_1, \ldots, X_K=k_K) = \frac{n!}{k_1! \cdots k_K!} \prod_{j=1}^K p_j^{k_j}$$

$$E[X_j] = np_j \qquad \text{Var}(X_j) = np_j(1-p_j)$$

**ML Use:** Cross-entropy loss derivation. Text generation token probabilities.

### Poisson Distribution

Number of events in a fixed interval, at average rate $\lambda$.

$$X \sim \text{Poisson}(\lambda)$$

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

$$E[X] = \lambda \qquad \text{Var}(X) = \lambda$$

> **Key property:** Mean = Variance = $\lambda$. This is the **only** distribution with this property.

**ML Use:** Count data modeling. Number of words in a document. Anomaly detection in event streams.

### Geometric Distribution

Number of trials until **first success**.

$$P(X = k) = (1-p)^{k-1} p, \quad k = 1, 2, 3, \ldots$$

$$E[X] = \frac{1}{p} \qquad \text{Var}(X) = \frac{1-p}{p^2}$$

### Negative Binomial Distribution

Number of trials until the $r$-th success.

$$P(X = k) = \binom{k-1}{r-1} p^r (1-p)^{k-r}$$

$$E[X] = \frac{r}{p} \qquad \text{Var}(X) = \frac{r(1-p)}{p^2}$$

**ML Use:** Overdispersed count data (where Var > Mean). NLP word frequency modeling.

---

## 9. Probability Distributions — Continuous

### Uniform Distribution

Every value in $[a, b]$ equally likely.

$$X \sim \text{Uniform}(a, b)$$

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

$$E[X] = \frac{a+b}{2} \qquad \text{Var}(X) = \frac{(b-a)^2}{12}$$

**ML Use:** Weight initialization (LeCun, Xavier uniform). Random number generation for sampling.

### Normal (Gaussian) Distribution

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

$$E[X] = \mu \qquad \text{Var}(X) = \sigma^2$$

**68-95-99.7 Rule:**
$$P(\mu - \sigma \leq X \leq \mu + \sigma) \approx 0.6827$$
$$P(\mu - 2\sigma \leq X \leq \mu + 2\sigma) \approx 0.9545$$
$$P(\mu - 3\sigma \leq X \leq \mu + 3\sigma) \approx 0.9973$$

**ML Use:** Weight initialization (He, Xavier normal). Gaussian noise in VAEs. Assumption in linear regression residuals.

### Standard Normal Distribution

$$Z \sim \mathcal{N}(0, 1)$$

$$\phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}$$

**Standardization:**
$$Z = \frac{X - \mu}{\sigma}$$

**CDF of Standard Normal:**
$$\Phi(z) = P(Z \leq z) = \int_{-\infty}^{z} \phi(t)\,dt$$

**Key values:** $\Phi(1.96) \approx 0.975$, $\Phi(2.576) \approx 0.995$

### Exponential Distribution

Time between Poisson events.

$$X \sim \text{Exp}(\lambda)$$

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

$$E[X] = \frac{1}{\lambda} \qquad \text{Var}(X) = \frac{1}{\lambda^2}$$

**Memoryless property:**
$$P(X > s + t \mid X > s) = P(X > t)$$

**ML Use:** Time-to-failure models. Dropout as Bernoulli, generalized by exponential family.

### Gamma Distribution

Sum of $k$ independent Exponential($\lambda$) random variables.

$$X \sim \text{Gamma}(k, \theta)$$

$$f(x) = \frac{x^{k-1} e^{-x/\theta}}{\theta^k \Gamma(k)}, \quad x > 0$$

$$E[X] = k\theta \qquad \text{Var}(X) = k\theta^2$$

where $\Gamma(k) = (k-1)!$ for integer $k$.

### Beta Distribution

Models probabilities (values between 0 and 1).

$$X \sim \text{Beta}(\alpha, \beta)$$

$$f(x) = \frac{x^{\alpha-1}(1-x)^{\beta-1}}{B(\alpha, \beta)}, \quad 0 \leq x \leq 1$$

$$B(\alpha, \beta) = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha+\beta)}$$

$$E[X] = \frac{\alpha}{\alpha+\beta} \qquad \text{Var}(X) = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$

**ML Use:** Bayesian prior for probabilities. Thompson sampling in reinforcement learning. Hyperparameter search.

### Chi-Squared Distribution

Sum of squares of $k$ standard normal variables.

$$X = Z_1^2 + Z_2^2 + \cdots + Z_k^2 \sim \chi^2(k)$$

$$E[X] = k \qquad \text{Var}(X) = 2k$$

**ML Use:** Goodness-of-fit tests. Feature selection ($\chi^2$ test). Hypothesis testing.

### Student's t-Distribution

$$T = \frac{Z}{\sqrt{V/\nu}} \sim t(\nu)$$

where $Z \sim \mathcal{N}(0,1)$ and $V \sim \chi^2(\nu)$ are independent.

$$f(t) = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}{\sqrt{\nu\pi}\,\Gamma\!\left(\frac{\nu}{2}\right)} \left(1+\frac{t^2}{\nu}\right)^{-(\nu+1)/2}$$

$$E[T] = 0 \quad (\nu > 1) \qquad \text{Var}(T) = \frac{\nu}{\nu-2} \quad (\nu > 2)$$

**As $\nu \to \infty$:** $t \to \mathcal{N}(0,1)$. Heavier tails than Normal for small $\nu$.

**ML Use:** Small sample inference. t-SNE visualization. Robust regression with heavy-tailed noise.

### F-Distribution

Ratio of two chi-squared distributions.

$$F = \frac{V_1/d_1}{V_2/d_2} \sim F(d_1, d_2)$$

**ML Use:** ANOVA. Comparing variances. F-test in linear regression significance.

### Laplace Distribution

$$X \sim \text{Laplace}(\mu, b)$$

$$f(x) = \frac{1}{2b} \exp\!\left(-\frac{|x-\mu|}{b}\right)$$

$$E[X] = \mu \qquad \text{Var}(X) = 2b^2$$

**ML Use:** L1 regularization (Lasso) corresponds to a Laplace prior on weights. Sparse representations.

### Dirichlet Distribution

Multivariate generalization of Beta. Parameters $\boldsymbol{\alpha} = (\alpha_1, \ldots, \alpha_K)$.

$$f(\mathbf{x}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K x_k^{\alpha_k - 1}$$

$$E[X_k] = \frac{\alpha_k}{\sum_j \alpha_j}$$

**ML Use:** Prior over categorical distributions in LDA (topic models). Bayesian neural networks.

---

## 10. Joint, Marginal & Conditional Distributions

### Joint Distribution

For two random variables $X$ and $Y$:

**Discrete:** $P(X=x, Y=y)$ — probability both equal specific values

**Continuous:** $f_{X,Y}(x,y)$ — joint PDF such that:
$$\int\!\!\int f_{X,Y}(x,y)\,dx\,dy = 1$$

### Marginal Distribution

Obtained by **integrating/summing out** the other variable.

**Discrete:**
$$P(X = x) = \sum_y P(X=x, Y=y)$$

**Continuous:**
$$f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y)\,dy$$

### Conditional Distribution

$$f_{X|Y}(x|y) = \frac{f_{X,Y}(x,y)}{f_Y(y)}$$

**Independence condition:**
$$X \perp Y \iff f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y)$$

### Covariance

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)] = E[XY] - E[X]E[Y]$$

**Properties:**
- $\text{Cov}(X, X) = \text{Var}(X)$
- $\text{Cov}(X, Y) = \text{Cov}(Y, X)$
- $\text{Cov}(aX, bY) = ab\,\text{Cov}(X, Y)$
- If $X \perp Y$: $\text{Cov}(X,Y) = 0$ (converse not always true!)

### Covariance Matrix

For a random vector $\mathbf{X} = (X_1, X_2, \ldots, X_p)^T$:

$$\boldsymbol{\Sigma} = \text{Cov}(\mathbf{X}) = E[(\mathbf{X}-\boldsymbol{\mu})(\mathbf{X}-\boldsymbol{\mu})^T]$$

$$\Sigma_{ij} = \text{Cov}(X_i, X_j)$$

The diagonal entries are variances: $\Sigma_{ii} = \text{Var}(X_i)$

**Properties:**
- Symmetric: $\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$
- Positive semi-definite: $\mathbf{v}^T\boldsymbol{\Sigma}\mathbf{v} \geq 0$ for all $\mathbf{v}$

**ML Use:** PCA finds the eigenvectors of the covariance matrix. Gaussian discriminant analysis. Multivariate normal distribution parameterization.

---

## 11. Expectation, Variance & Covariance — Deep Dive

### Law of Total Expectation (Tower Property)

$$E[X] = E[E[X|Y]]$$

**Example:** $E[\text{model loss}] = E_{\text{dataset}}\left[E[\text{loss}|\text{dataset}]\right]$

### Law of Total Variance

$$\text{Var}(X) = E[\text{Var}(X|Y)] + \text{Var}(E[X|Y])$$

This decomposes total variance into:
- $E[\text{Var}(X|Y)]$: **within-group variance** (irreducible noise)
- $\text{Var}(E[X|Y])$: **between-group variance** (variance due to grouping)

**ML Connection — Bias-Variance Tradeoff:**

$$E[(\hat{f}(x) - f(x))^2] = \text{Bias}^2 + \text{Variance} + \text{Irreducible Error}$$

$$\text{Bias}^2 = \left(E[\hat{f}(x)] - f(x)\right)^2$$
$$\text{Variance} = E\left[\left(\hat{f}(x) - E[\hat{f}(x)]\right)^2\right]$$

### Moment Generating Function (MGF)

$$M_X(t) = E[e^{tX}]$$

**Why useful:**
$$E[X^k] = M_X^{(k)}(0) = \left.\frac{d^k M_X}{dt^k}\right|_{t=0}$$

The MGF uniquely determines the distribution (if it exists).

### Characteristic Function

$$\varphi_X(t) = E[e^{itX}] = E[\cos(tX)] + i\,E[\sin(tX)]$$

Always exists (unlike MGF). Used to prove the CLT.

### Jensen's Inequality

For a **convex** function $g$:
$$g(E[X]) \leq E[g(X)]$$

For a **concave** function $g$:
$$g(E[X]) \geq E[g(X)]$$

**Examples:**
- $g(x) = x^2$ (convex): $E[X]^2 \leq E[X^2]$ → confirms Var$(X) \geq 0$
- $g(x) = \log(x)$ (concave): $\log(E[X]) \geq E[\log(X)]$

**ML Use:** Deriving the ELBO in variational autoencoders. Proving log-sum-exp inequality. KL divergence is non-negative.

---

## 12. The Normal Distribution — Master Reference

### Why the Normal Distribution Dominates ML

1. **Central Limit Theorem** — sums of any i.i.d. variables → Normal
2. **Maximum entropy** — given mean and variance, Normal has maximum entropy
3. **Closed under linear operations** — linear combinations of Normals are Normal
4. **Conjugate prior** for its own mean

### Key Formulas

$$X \sim \mathcal{N}(\mu, \sigma^2)$$

**PDF:**
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} \exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**Log-PDF (used in log-likelihood):**
$$\log f(x) = -\frac{1}{2}\log(2\pi\sigma^2) - \frac{(x-\mu)^2}{2\sigma^2}$$

**CDF:**
$$F(x) = \Phi\!\left(\frac{x-\mu}{\sigma}\right) = \frac{1}{2}\left[1 + \text{erf}\!\left(\frac{x-\mu}{\sigma\sqrt{2}}\right)\right]$$

**Q-function (tail probability):**
$$Q(z) = P(Z > z) = 1 - \Phi(z) = \frac{1}{2}\text{erfc}\!\left(\frac{z}{\sqrt{2}}\right)$$

### Multivariate Normal Distribution

$$\mathbf{X} \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$$

$$f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2}|\boldsymbol{\Sigma}|^{1/2}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)$$

The term $(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$ is the **Mahalanobis distance** squared.

**Properties:**
- Any linear combination $\mathbf{a}^T\mathbf{X} \sim \mathcal{N}(\mathbf{a}^T\boldsymbol{\mu}, \mathbf{a}^T\boldsymbol{\Sigma}\mathbf{a})$
- Marginals are Normal: $X_i \sim \mathcal{N}(\mu_i, \Sigma_{ii})$
- Conditionals are Normal

**ML Use:** Gaussian discriminant analysis. Gaussian processes. VAE latent space. Kalman filters.

### Log-Normal Distribution

If $\log X \sim \mathcal{N}(\mu, \sigma^2)$, then $X$ is Log-Normal.

$$E[X] = e^{\mu + \sigma^2/2} \qquad \text{Var}(X) = (e^{\sigma^2}-1)e^{2\mu+\sigma^2}$$

**ML Use:** Income, price data. Learning rates (often sampled in log space). Any positive-valued skewed data.

### Mahalanobis Distance

$$D_M(\mathbf{x}) = \sqrt{(\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})}$$

**ML Use:** Anomaly detection. Distance metric that accounts for feature correlations and scales.

---

## 13. Sampling Theory & the Central Limit Theorem

### Sampling Distributions

The **sampling distribution** is the distribution of a statistic computed from repeated samples.

**For sample mean $\bar{X}$ from population with mean $\mu$ and variance $\sigma^2$:**

$$E[\bar{X}] = \mu \qquad \text{Var}(\bar{X}) = \frac{\sigma^2}{n} \qquad \text{SE}(\bar{X}) = \frac{\sigma}{\sqrt{n}}$$

### Central Limit Theorem (CLT)

> **The most important theorem in statistics.**

Let $X_1, X_2, \ldots, X_n$ be i.i.d. with mean $\mu$ and finite variance $\sigma^2$. Then:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} \mathcal{N}(0, 1) \quad \text{as } n \to \infty$$

Equivalently:
$$\bar{X}_n \xrightarrow{d} \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$$

**Practical rule:** CLT applies well for $n \geq 30$ (for roughly symmetric distributions).

**ML Applications:**
1. Mini-batch gradient is an average → approximately Normal by CLT → justifies using Normal-based learning rate schedules
2. Ensemble predictions converge to Normal
3. Test set accuracy averages are approximately Normal → confidence intervals are valid

### Berry-Esseen Theorem

The CLT convergence rate:
$$\sup_x \left| P\!\left(\frac{\bar{X}-\mu}{\sigma/\sqrt{n}} \leq x\right) - \Phi(x) \right| \leq \frac{C \cdot E[|X-\mu|^3]}{\sigma^3 \sqrt{n}}$$

Higher skewness → slower convergence to Normal.

### Law of Large Numbers (LLN)

**Weak LLN:**
$$\bar{X}_n \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

**Strong LLN:**
$$P\!\left(\lim_{n\to\infty} \bar{X}_n = \mu\right) = 1$$

**ML Use:** As training set size grows, empirical risk → true risk. More data → better generalization.

---

## 14. Estimation Theory

### Point Estimation

A **point estimator** $\hat{\theta}$ is a function of the sample that estimates parameter $\theta$.

**Desirable properties:**

**1. Unbiasedness:**
$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta = 0$$

**2. Consistency:**
$$\hat{\theta}_n \xrightarrow{P} \theta \quad \text{as } n \to \infty$$

**3. Efficiency (CRLB):**
The **Cramér-Rao Lower Bound** gives the minimum variance any unbiased estimator can achieve:

$$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$

where $I(\theta)$ is the **Fisher information**:

$$I(\theta) = E\!\left[\left(\frac{\partial \log f(X;\theta)}{\partial \theta}\right)^2\right] = -E\!\left[\frac{\partial^2 \log f(X;\theta)}{\partial \theta^2}\right]$$

**4. Sufficiency:** $T(\mathbf{X})$ is sufficient for $\theta$ if the distribution of $\mathbf{X}|T$ doesn't depend on $\theta$.

### Maximum Likelihood Estimation (MLE)

**Log-likelihood:**
$$\ell(\theta) = \log L(\theta; \mathbf{x}) = \sum_{i=1}^n \log f(x_i; \theta)$$

**MLE:**
$$\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta)$$

**Found by solving:**
$$\frac{\partial \ell(\theta)}{\partial \theta} = 0$$

**Example — MLE for Normal distribution:**

$$\ell(\mu, \sigma^2) = -\frac{n}{2}\log(2\pi) - \frac{n}{2}\log\sigma^2 - \frac{1}{2\sigma^2}\sum_{i=1}^n(x_i-\mu)^2$$

Setting derivatives to zero:
$$\hat{\mu}_{\text{MLE}} = \bar{x} = \frac{1}{n}\sum x_i$$
$$\hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum(x_i - \bar{x})^2 \quad \text{(biased — divides by } n \text{, not } n-1\text{)}$$

**ML Connection:**
- **MSE loss** = negative log-likelihood under Gaussian noise assumption
- **Cross-entropy loss** = negative log-likelihood under Bernoulli/Categorical assumption
- **Training a neural network = performing MLE**

$$\text{MSE} = \frac{1}{n}\sum(y_i - \hat{y}_i)^2 \propto -\ell(\theta)_{\text{Gaussian}}$$
$$\text{Cross-entropy} = -\frac{1}{n}\sum y_i\log\hat{p}_i + (1-y_i)\log(1-\hat{p}_i) \propto -\ell(\theta)_{\text{Bernoulli}}$$

### Maximum A Posteriori (MAP) Estimation

$$\hat{\theta}_{\text{MAP}} = \arg\max_\theta \underbrace{\log P(\theta|\mathbf{x})}_{\text{posterior}} = \arg\max_\theta \left[\underbrace{\log P(\mathbf{x}|\theta)}_{\text{log-likelihood}} + \underbrace{\log P(\theta)}_{\text{log-prior}}\right]$$

**Regularization as MAP:**
- L2 regularization ↔ Gaussian prior: $P(\theta) = \mathcal{N}(0, \lambda^{-1})$

$$\hat{\theta}_{\text{Ridge}} = \arg\min_\theta \left[\text{MSE} + \lambda\|\theta\|_2^2\right] = \hat{\theta}_{\text{MAP with Gaussian prior}}$$

- L1 regularization ↔ Laplace prior: $P(\theta) = \text{Laplace}(0, \lambda^{-1})$

$$\hat{\theta}_{\text{Lasso}} = \arg\min_\theta \left[\text{MSE} + \lambda\|\theta\|_1\right] = \hat{\theta}_{\text{MAP with Laplace prior}}$$

### Confidence Intervals

A $(1-\alpha) \times 100\%$ confidence interval for $\mu$:

**Known $\sigma$:**
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**Unknown $\sigma$ (use $t$ distribution with $\nu=n-1$ df):**
$$\bar{x} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

**Common critical values:**

| Confidence Level | $\alpha$ | $z_{\alpha/2}$ |
|---|---|---|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

**Example:** n=100 test samples, $\bar{\text{accuracy}}=0.85$, $s=0.12$

$$95\% \text{ CI} = 0.85 \pm 1.96 \times \frac{0.12}{\sqrt{100}} = 0.85 \pm 0.0235 = (0.826, \, 0.874)$$

**Interpretation:** If we repeated the test procedure 100 times, approximately 95 of those intervals would contain the true accuracy.

---

## 15. Hypothesis Testing

### Framework

| Step | Action |
|---|---|
| 1 | State $H_0$ (null hypothesis) and $H_1$ (alternative) |
| 2 | Choose significance level $\alpha$ (usually 0.05) |
| 3 | Compute test statistic |
| 4 | Find p-value or critical region |
| 5 | Reject $H_0$ if p-value < $\alpha$ |

### Error Types

| Decision \ Truth | $H_0$ True | $H_0$ False |
|---|---|---|
| Fail to reject $H_0$ | ✅ Correct ($1-\alpha$) | ❌ **Type II error** ($\beta$) |
| Reject $H_0$ | ❌ **Type I error** ($\alpha$) | ✅ Correct (Power = $1-\beta$) |

**Type I error (False Positive):** Rejecting $H_0$ when it's true → probability = $\alpha$ (significance level)

**Type II error (False Negative):** Failing to reject $H_0$ when $H_1$ is true → probability = $\beta$

**Power of a test:**
$$\text{Power} = 1 - \beta = P(\text{reject } H_0 \mid H_0 \text{ false})$$

**Precision-Recall analogy in ML:**
- Type I error ~ False Positive Rate (1 - Specificity)
- Type II error ~ False Negative Rate (1 - Recall/Sensitivity)

### The p-value

$$p\text{-value} = P(\text{observing test statistic as extreme or more extreme} \mid H_0 \text{ true})$$

**NOT the probability that $H_0$ is true.** It's the probability of the data given $H_0$.

**Decision rule:**
- $p < \alpha$: Reject $H_0$ (result is "statistically significant")
- $p \geq \alpha$: Fail to reject $H_0$

### One-Sample z-test

Test if population mean equals hypothesized value $\mu_0$, known $\sigma$:

$$z = \frac{\bar{x} - \mu_0}{\sigma / \sqrt{n}}$$

**Reject $H_0$: $\mu = \mu_0$ when:**
- Two-tailed: $|z| > z_{\alpha/2}$
- Right-tailed: $z > z_\alpha$
- Left-tailed: $z < -z_\alpha$

### One-Sample t-test

Test if population mean equals $\mu_0$, unknown $\sigma$:

$$t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}, \quad \text{df} = n - 1$$

### Two-Sample t-test

Compare means of two groups (equal variances assumed):

$$t = \frac{\bar{x}_1 - \bar{x}_2}{s_p\sqrt{1/n_1 + 1/n_2}}, \quad \text{df} = n_1 + n_2 - 2$$

where the **pooled variance:**
$$s_p^2 = \frac{(n_1-1)s_1^2 + (n_2-1)s_2^2}{n_1+n_2-2}$$

**Welch's t-test (unequal variances):**
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_1^2/n_1 + s_2^2/n_2}}$$

**Satterthwaite degrees of freedom:**
$$\nu = \frac{\left(\frac{s_1^2}{n_1} + \frac{s_2^2}{n_2}\right)^2}{\frac{(s_1^2/n_1)^2}{n_1-1} + \frac{(s_2^2/n_2)^2}{n_2-1}}$$

**ML Use:** Comparing two models' performance. A/B testing ML systems.

### Effect Size

Statistical significance ≠ practical significance. Measure **effect size**:

**Cohen's d:**
$$d = \frac{\bar{x}_1 - \bar{x}_2}{s_p}$$

| $|d|$ | Interpretation |
|---|---|
| 0.2 | Small |
| 0.5 | Medium |
| 0.8 | Large |

**ML Use:** A 0.1% accuracy improvement with $p < 0.001$ may have tiny effect size with large $n$.

### Multiple Testing Problem

If you run $m$ tests at $\alpha = 0.05$:
$$P(\text{at least one false positive}) = 1 - (1-0.05)^m$$

For $m=20$: $P \approx 0.64$ — 64% chance of a false positive!

**Bonferroni Correction:**
$$\alpha_{\text{corrected}} = \frac{\alpha}{m}$$

**Benjamini-Hochberg (FDR control):**
Sort p-values $p_{(1)} \leq p_{(2)} \leq \cdots \leq p_{(m)}$. Reject all $H_{(i)}$ where:
$$p_{(i)} \leq \frac{i}{m} \cdot \alpha$$

**ML Use:** Feature selection tests. Hyperparameter search over many configurations.

---

## 16. Statistical Tests Reference

### Normality Tests

**Shapiro-Wilk test** (best for $n < 50$):
$$W = \frac{\left(\sum_{i=1}^n a_i x_{(i)}\right)^2}{\sum_{i=1}^n (x_i - \bar{x})^2}$$

Close to 1 → more normal.

**Kolmogorov-Smirnov test:**
$$D_n = \sup_x |F_n(x) - F(x)|$$

where $F_n$ is the empirical CDF.

### Chi-Squared Test for Independence

For a contingency table with $r$ rows and $c$ columns:

$$\chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}, \quad \text{df} = (r-1)(c-1)$$

where $E_{ij} = \frac{\text{(row total}_i) \times \text{(col total}_j)}{\text{grand total}}$

**ML Use:** Feature selection — testing if a categorical feature is independent of the target.

### ANOVA (Analysis of Variance)

Tests if $k$ group means are equal.

**F-statistic:**
$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}} = \frac{MS_B}{MS_W}$$

$$MS_B = \frac{\sum_{j=1}^k n_j(\bar{x}_j - \bar{x})^2}{k-1}, \quad MS_W = \frac{\sum_{j=1}^k \sum_{i=1}^{n_j}(x_{ij}-\bar{x}_j)^2}{N-k}$$

**Total sum of squares decomposition:**
$$SS_T = SS_B + SS_W$$

**ML Use:** Compare performance across $k > 2$ models simultaneously.

### Non-Parametric Tests

| Test | Purpose | Parametric Equivalent |
|---|---|---|
| Mann-Whitney U | Compare 2 independent groups | Two-sample t-test |
| Wilcoxon signed-rank | Compare 2 paired groups | Paired t-test |
| Kruskal-Wallis | Compare $k$ groups | One-way ANOVA |
| Spearman rank | Monotone association | Pearson correlation |
| Friedman | $k$ treatments, repeated measures | Repeated-measures ANOVA |

**When to use non-parametric tests:** Non-normal data, small samples, ordinal data.

---

## 17. Correlation & Dependence

### Pearson Correlation Coefficient

Measures **linear** association between two variables.

**Population:**
$$\rho_{X,Y} = \frac{\text{Cov}(X,Y)}{\sigma_X \sigma_Y} = \frac{E[(X-\mu_X)(Y-\mu_Y)]}{\sigma_X\sigma_Y}$$

**Sample:**
$$r = \frac{\sum_{i=1}^n (x_i-\bar{x})(y_i-\bar{y})}{\sqrt{\sum(x_i-\bar{x})^2 \cdot \sum(y_i-\bar{y})^2}}$$

$$-1 \leq r \leq 1$$

| $r$ | Interpretation |
|---|---|
| +1 | Perfect positive linear |
| +0.7 to +1 | Strong positive |
| +0.3 to +0.7 | Moderate positive |
| 0 | No linear association |
| −1 | Perfect negative linear |

> **Warning:** $r = 0$ does NOT mean no association. Only no *linear* association.

### Spearman Rank Correlation

Replace values with their ranks, then compute Pearson:

$$r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i = \text{rank}(x_i) - \text{rank}(y_i)$.

**Measures monotone (not just linear) association.** Robust to outliers.

### Kendall's Tau

$$\tau = \frac{n_c - n_d}{\binom{n}{2}}$$

where $n_c$ = concordant pairs, $n_d$ = discordant pairs.

### Point-Biserial Correlation

Correlation between a binary and continuous variable:

$$r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{s_Y}\sqrt{\frac{n_1 n_0}{n^2}}$$

**ML Use:** Correlation between binary target and continuous features.

### Correlation Matrix

$$\mathbf{R} = \begin{pmatrix} 1 & r_{12} & \cdots & r_{1p} \\ r_{21} & 1 & \cdots & r_{2p} \\ \vdots & & \ddots & \vdots \\ r_{p1} & r_{p2} & \cdots & 1 \end{pmatrix}$$

Symmetric, all diagonals = 1.

**ML Use:** Feature selection (drop highly correlated features). Multicollinearity detection in regression.

### Partial Correlation

Correlation between $X$ and $Y$ after removing effect of $Z$:

$$r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ}r_{YZ}}{\sqrt{(1-r_{XZ}^2)(1-r_{YZ}^2)}}$$

---

## 18. Linear Regression — Statistical Foundation

### Simple Linear Regression

$$Y = \beta_0 + \beta_1 X + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2)$$

**OLS (Ordinary Least Squares) estimates:**

$$\hat{\beta}_1 = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sum(x_i-\bar{x})^2} = \frac{\text{Cov}(X,Y)}{\text{Var}(X)} = r\frac{s_Y}{s_X}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$$

### Multiple Linear Regression

$$\mathbf{Y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}, \quad \boldsymbol{\varepsilon} \sim \mathcal{N}(\mathbf{0}, \sigma^2\mathbf{I})$$

**OLS solution (normal equations):**

$$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}$$

**Gauss-Markov Theorem:** OLS is the **Best Linear Unbiased Estimator (BLUE)** under classical assumptions.

### Coefficient of Determination

$$R^2 = 1 - \frac{SS_{\text{res}}}{SS_{\text{tot}}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

$$0 \leq R^2 \leq 1$$

**Adjusted $R^2$ (penalizes extra predictors):**

$$\bar{R}^2 = 1 - (1-R^2)\frac{n-1}{n-p-1}$$

### Residuals & Assumptions (LIME)

**L**inearity: $E[\varepsilon] = 0$
**I**ndependence: residuals are uncorrelated
**M**ean zero: $E[\varepsilon_i] = 0$
**E**qual variance: $\text{Var}(\varepsilon_i) = \sigma^2$ (homoscedasticity)

### Variance of OLS Estimator

$$\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^T\mathbf{X})^{-1}$$

**Standard error of $\hat{\beta}_j$:**
$$SE(\hat{\beta}_j) = \hat{\sigma}\sqrt{[(\mathbf{X}^T\mathbf{X})^{-1}]_{jj}}$$

**t-statistic for testing $H_0: \beta_j = 0$:**
$$t_j = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} \sim t(n-p-1)$$

### Ridge Regression (L2 Regularization)

$$\hat{\boldsymbol{\beta}}_{\text{Ridge}} = (\mathbf{X}^T\mathbf{X} + \lambda\mathbf{I})^{-1}\mathbf{X}^T\mathbf{Y}$$

Minimizes: $\|\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2$

Ridge always has a solution even when $\mathbf{X}^T\mathbf{X}$ is singular.

---

## 19. Information Theory

Information theory is **fundamental to modern ML** — it underlies loss functions, model evaluation, and compression.

### Self-Information (Surprisal)

The information content of event $x$ with probability $P(x)$:

$$I(x) = -\log_2 P(x) \text{ bits} = -\log P(x) \text{ nats}$$

**Intuition:** Rare events ($P$ near 0) carry more information. Certain events ($P=1$) carry zero information.

**Example:** A fair coin flip: $I(\text{heads}) = -\log_2(0.5) = 1$ bit.

### Shannon Entropy

Expected self-information — measures **uncertainty** of a distribution.

**Discrete:**
$$H(X) = -\sum_{x} P(x) \log P(x) = E[-\log P(X)]$$

**Continuous (Differential entropy):**
$$h(X) = -\int f(x) \log f(x)\,dx$$

**Properties:**
- $H(X) \geq 0$
- Maximum entropy when distribution is uniform: $H = \log K$ for $K$ outcomes
- $H(X,Y) = H(X) + H(Y)$ if $X \perp Y$
- $H(X|Y) \leq H(X)$ — conditioning reduces entropy

**Example:** Binary entropy function:
$$H(p) = -p\log p - (1-p)\log(1-p)$$
Maximum at $p=0.5$: $H(0.5) = 1$ bit.

**ML Use:** Decision tree splitting criterion (Information Gain). Measures impurity in nodes.

### Joint Entropy

$$H(X, Y) = -\sum_x \sum_y P(x,y) \log P(x,y)$$

### Conditional Entropy

$$H(Y|X) = -\sum_x \sum_y P(x,y) \log P(y|x) = H(X,Y) - H(X)$$

### Information Gain (Mutual Information)

How much knowing $X$ reduces uncertainty about $Y$:

$$I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$$

$$I(X;Y) = \sum_x \sum_y P(x,y) \log \frac{P(x,y)}{P(x)P(y)}$$

**Properties:**
- $I(X;Y) \geq 0$
- $I(X;Y) = 0$ iff $X \perp Y$
- $I(X;Y) = I(Y;X)$ (symmetric)

**ML Use:** Feature selection — select features with highest mutual information with target.

### KL Divergence (Relative Entropy)

Measures how much distribution $Q$ differs from reference distribution $P$:

$$D_{KL}(P \| Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)} = E_P\!\left[\log\frac{P(X)}{Q(X)}\right]$$

**Continuous:**
$$D_{KL}(P \| Q) = \int p(x) \log \frac{p(x)}{q(x)}\,dx$$

**Properties:**
- $D_{KL}(P\|Q) \geq 0$ (Gibbs' inequality, follows from Jensen)
- $D_{KL}(P\|Q) = 0$ iff $P = Q$ a.e.
- **Asymmetric:** $D_{KL}(P\|Q) \neq D_{KL}(Q\|P)$ in general

**Forward KL** ($P\|Q$): zero-avoiding (mass-covering)
**Reverse KL** ($Q\|P$): zero-forcing (mode-seeking)

**ML Use:** VAE loss. Policy gradient objectives. Model comparison. Any probabilistic model training.

### Cross-Entropy

$$H(P, Q) = H(P) + D_{KL}(P\|Q) = -\sum_x P(x) \log Q(x)$$

When $P$ is the true distribution and $Q$ is the model's distribution.

**Binary Cross-Entropy Loss:**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right]$$

**Categorical Cross-Entropy Loss:**
$$\mathcal{L} = -\frac{1}{n}\sum_{i=1}^n \sum_{k=1}^K y_{ik} \log \hat{p}_{ik}$$

> **Key insight:** Minimizing cross-entropy loss = minimizing KL divergence between true and predicted distributions = MLE.

### Jensen-Shannon Divergence

Symmetric version of KL divergence:

$$JSD(P\|Q) = \frac{1}{2}D_{KL}(P\|M) + \frac{1}{2}D_{KL}(Q\|M), \quad M = \frac{P+Q}{2}$$

$$0 \leq JSD(P\|Q) \leq 1 \quad (\text{using } \log_2)$$

**ML Use:** Generative Adversarial Networks (GANs) — original GAN minimizes JSD between real and generated distributions.

### Wasserstein Distance

$$W_1(P, Q) = \inf_{\gamma \in \Pi(P,Q)} E_{(x,y)\sim\gamma}\left[\|x-y\|\right]$$

Also known as the **Earth Mover's Distance** — the minimum cost of transporting one distribution to another.

**ML Use:** Wasserstein GAN (WGAN) — more stable training than original GAN.

---

## 20. Bayesian Statistics

### The Bayesian Paradigm

| Aspect | Frequentist | Bayesian |
|---|---|---|
| Probability | Long-run frequency | Degree of belief |
| Parameters | Fixed unknown constants | Random variables |
| Inference | Confidence intervals | Credible intervals |
| Result | Point estimate | Full posterior distribution |
| Prior knowledge | Not incorporated | Incorporated via prior |

### Bayesian Inference Pipeline

$$\underbrace{P(\theta|\mathbf{X})}_{\text{posterior}} = \frac{\underbrace{P(\mathbf{X}|\theta)}_{\text{likelihood}} \cdot \underbrace{P(\theta)}_{\text{prior}}}{\underbrace{P(\mathbf{X})}_{\text{evidence}}}$$

**Evidence (marginal likelihood):**
$$P(\mathbf{X}) = \int P(\mathbf{X}|\theta) P(\theta)\,d\theta$$

### Conjugate Priors

When prior and posterior are in the same family — tractable closed-form updates.

| Likelihood | Conjugate Prior | Posterior |
|---|---|---|
| Binomial($n$, $p$) | Beta($\alpha$, $\beta$) | Beta($\alpha+k$, $\beta+n-k$) |
| Normal($\mu$, $\sigma^2$) | Normal($\mu_0$, $\tau^2$) | Normal (updated) |
| Poisson($\lambda$) | Gamma($\alpha$, $\beta$) | Gamma($\alpha+\sum x_i$, $\beta+n$) |
| Categorical($\mathbf{p}$) | Dirichlet($\boldsymbol{\alpha}$) | Dirichlet($\boldsymbol{\alpha}+\mathbf{n}$) |

**Example — Beta-Binomial:**

Prior: $P(p) = \text{Beta}(\alpha, \beta)$ — our prior belief about coin bias.

Observe $k$ heads in $n$ flips.

Posterior: $P(p|\text{data}) = \text{Beta}(\alpha + k, \, \beta + n - k)$

As $n \to \infty$: posterior concentrates around MLE $\hat{p} = k/n$.

### Bayesian Credible Interval

A 95% **credible interval** $[a, b]$ satisfies:
$$P(a \leq \theta \leq b \mid \mathbf{X}) = 0.95$$

**Direct probability statement** about the parameter (unlike frequentist CI).

**Highest Posterior Density (HPD) interval:** The shortest interval containing 95% of the posterior mass.

### Variational Inference

When posterior $P(\theta|\mathbf{X})$ is intractable, approximate with $Q(\theta)$:

$$Q^*(\theta) = \arg\min_{Q \in \mathcal{Q}} D_{KL}(Q(\theta) \| P(\theta|\mathbf{X}))$$

**Evidence Lower Bound (ELBO):**

$$\log P(\mathbf{X}) = \mathcal{L}(Q) + D_{KL}(Q\|P)$$

$$\mathcal{L}(Q) = E_Q[\log P(\mathbf{X}|\theta)] - D_{KL}(Q(\theta)\|P(\theta))$$

Maximizing ELBO = Minimizing KL divergence to posterior.

**ML Use:** Variational Autoencoders (VAE). Bayesian neural networks. Topic models (LDA).

### Markov Chain Monte Carlo (MCMC)

Sample from intractable posteriors using a Markov chain whose stationary distribution is $P(\theta|\mathbf{X})$.

**Metropolis-Hastings acceptance ratio:**

$$\alpha = \min\!\left(1, \frac{P(\theta^*|\mathbf{X}) \, Q(\theta_t|\theta^*)}{P(\theta_t|\mathbf{X}) \, Q(\theta^*|\theta_t)}\right)$$

**Hamiltonian Monte Carlo (HMC):** Uses gradient information for efficient sampling — used in Stan, PyMC3.

### Gaussian Processes

A Gaussian process (GP) is a distribution over functions:

$$f(\mathbf{x}) \sim \mathcal{GP}(m(\mathbf{x}), k(\mathbf{x}, \mathbf{x}'))$$

where $m(\mathbf{x})$ is the mean function and $k(\mathbf{x}, \mathbf{x}')$ is the covariance (kernel) function.

**Posterior predictive:**
$$P(f_* | \mathbf{X}, \mathbf{y}, \mathbf{x}_*) = \mathcal{N}(\mu_*, \sigma^2_*)$$

$$\mu_* = \mathbf{k}_*^T(\mathbf{K}+\sigma^2\mathbf{I})^{-1}\mathbf{y}$$
$$\sigma^2_* = k_{**} - \mathbf{k}_*^T(\mathbf{K}+\sigma^2\mathbf{I})^{-1}\mathbf{k}_*$$

**ML Use:** Bayesian optimization (hyperparameter tuning). Uncertainty quantification. Small data regression.

**Common Kernels:**

| Kernel | Formula | Property |
|---|---|---|
| RBF / Squared Exponential | $k(x,x') = \exp(-\|x-x'\|^2/2l^2)$ | Infinitely differentiable |
| Matérn | Complex, parameterized by $\nu$ | Controls smoothness |
| Linear | $k(x,x') = x^T x'$ | GP equivalent of linear regression |
| Periodic | $\exp(-2\sin^2(\pi\|x-x'\|/p)/l^2)$ | Periodic functions |

---

## 21. Matrix Statistics & Multivariate Analysis

### Principal Component Analysis (PCA)

**Goal:** Find directions of maximum variance in $p$-dimensional data.

**Covariance matrix:**
$$\mathbf{S} = \frac{1}{n-1}(\mathbf{X}-\bar{\mathbf{X}})^T(\mathbf{X}-\bar{\mathbf{X}})$$

**Eigendecomposition:**
$$\mathbf{S} = \mathbf{V}\boldsymbol{\Lambda}\mathbf{V}^T$$

where $\mathbf{V} = [\mathbf{v}_1, \ldots, \mathbf{v}_p]$ are eigenvectors (principal components) and $\boldsymbol{\Lambda} = \text{diag}(\lambda_1, \ldots, \lambda_p)$ are eigenvalues ($\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p$).

**Proportion of variance explained by component $j$:**
$$\text{PVE}_j = \frac{\lambda_j}{\sum_{k=1}^p \lambda_k}$$

**Projection onto first $k$ PCs:**
$$\mathbf{Z} = \mathbf{X}\mathbf{V}_k \quad \in \mathbb{R}^{n \times k}$$

**Reconstruction error:**
$$\|\mathbf{X} - \mathbf{X}\mathbf{V}_k\mathbf{V}_k^T\|_F^2 = \sum_{j=k+1}^p \lambda_j$$

### Singular Value Decomposition (SVD)

$$\mathbf{X} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^T$$

where $\mathbf{U} \in \mathbb{R}^{n \times n}$, $\boldsymbol{\Sigma} \in \mathbb{R}^{n \times p}$, $\mathbf{V} \in \mathbb{R}^{p \times p}$.

**Connection to PCA:** The right singular vectors of $\mathbf{X}$ are the eigenvectors of $\mathbf{X}^T\mathbf{X}$.

$$\sigma_j^2 / (n-1) = \lambda_j \quad \text{(eigenvalues of covariance matrix)}$$

**Low-rank approximation (Eckart-Young theorem):**

The best rank-$k$ approximation in Frobenius norm:
$$\mathbf{X}_k = \mathbf{U}_k\boldsymbol{\Sigma}_k\mathbf{V}_k^T$$

### Mahalanobis Distance

$$D^2(\mathbf{x}) = (\mathbf{x}-\boldsymbol{\mu})^T\boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})$$

Under $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$: $D^2 \sim \chi^2(p)$

**ML Use:** Anomaly detection. Gaussian discriminant analysis. k-NN with Mahalanobis metric.

### Linear Discriminant Analysis (LDA)

Find the projection $\mathbf{w}$ that maximizes class separation:

$$\mathbf{w}^* = \arg\max_{\mathbf{w}} \frac{\mathbf{w}^T\mathbf{S}_B\mathbf{w}}{\mathbf{w}^T\mathbf{S}_W\mathbf{w}}$$

**Between-class scatter matrix:**
$$\mathbf{S}_B = \sum_{k=1}^K n_k(\boldsymbol{\mu}_k - \boldsymbol{\mu})(\boldsymbol{\mu}_k - \boldsymbol{\mu})^T$$

**Within-class scatter matrix:**
$$\mathbf{S}_W = \sum_{k=1}^K \sum_{i \in C_k}(\mathbf{x}_i - \boldsymbol{\mu}_k)(\mathbf{x}_i - \boldsymbol{\mu}_k)^T$$

**Solution:** Eigenvectors of $\mathbf{S}_W^{-1}\mathbf{S}_B$.

---

## 22. Resampling Methods

### Cross-Validation

**k-Fold CV:**
$$CV_{(k)} = \frac{1}{k}\sum_{j=1}^k \text{MSE}_j$$

**Leave-One-Out CV (LOOCV):**
$$CV_{(n)} = \frac{1}{n}\sum_{i=1}^n (y_i - \hat{y}_{-i})^2$$

For linear regression, LOOCV has a shortcut:
$$CV_{(n)} = \frac{1}{n}\sum_{i=1}^n \left(\frac{y_i - \hat{y}_i}{1-h_{ii}}\right)^2$$

where $h_{ii} = [\mathbf{H}]_{ii} = [\mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T]_{ii}$ is the leverage.

### Bootstrap

Sample **with replacement** $B$ times from the dataset. Each bootstrap sample is size $n$.

**Probability a given observation is NOT in a bootstrap sample:**
$$P(\text{not selected}) = \left(1-\frac{1}{n}\right)^n \to e^{-1} \approx 0.368 \quad \text{as } n \to \infty$$

So about **36.8% of observations** are out-of-bag (OOB) in each bootstrap sample.

**Bootstrap standard error:**
$$\widehat{SE}_B(\hat{\theta}) = \sqrt{\frac{1}{B-1}\sum_{b=1}^B (\hat{\theta}^{*b} - \bar{\hat{\theta}}^*)^2}$$

**Bootstrap confidence interval (percentile method):**
$$\text{CI}_{1-\alpha} = \left[\hat{\theta}^*_{(\alpha/2)}, \, \hat{\theta}^*_{(1-\alpha/2)}\right]$$

**ML Use:** Random Forest uses bagging (bootstrap aggregation). OOB error as validation estimate.

### Jackknife

Leave one observation out at a time:

$$\hat{\theta}_{(-i)} = \hat{\theta}(\mathbf{x}_1, \ldots, \mathbf{x}_{i-1}, \mathbf{x}_{i+1}, \ldots, \mathbf{x}_n)$$

**Jackknife estimate of bias:**
$$\widehat{\text{Bias}}_J = (n-1)(\bar{\hat{\theta}}_{(\cdot)} - \hat{\theta})$$

**Jackknife SE:**
$$\widehat{SE}_J = \sqrt{\frac{n-1}{n}\sum_{i=1}^n (\hat{\theta}_{(-i)} - \bar{\hat{\theta}})^2}$$

---

## 23. Statistics in Deep Learning — Direct Connections

### Batch Normalization

For a mini-batch $\mathcal{B} = \{x_1, \ldots, x_m\}$:

**Mini-batch mean:**
$$\mu_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m x_i$$

**Mini-batch variance:**
$$\sigma^2_\mathcal{B} = \frac{1}{m}\sum_{i=1}^m (x_i - \mu_\mathcal{B})^2$$

**Normalize:**
$$\hat{x}_i = \frac{x_i - \mu_\mathcal{B}}{\sqrt{\sigma^2_\mathcal{B} + \varepsilon}}$$

**Scale and shift (learnable parameters $\gamma$, $\beta$):**
$$y_i = \gamma\hat{x}_i + \beta$$

**Why it works statistically:**
- Reduces internal covariate shift
- Smooths loss landscape (reduces condition number of Hessian)
- Acts as regularizer — adds noise via mini-batch statistics

### Weight Initialization

**Xavier/Glorot initialization** (for tanh/sigmoid):
$$W \sim \mathcal{U}\!\left[-\frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}, \frac{\sqrt{6}}{\sqrt{n_{in}+n_{out}}}\right]$$
$$\text{Var}(W) = \frac{2}{n_{in}+n_{out}}$$

**He initialization** (for ReLU):
$$W \sim \mathcal{N}\!\left(0, \frac{2}{n_{in}}\right)$$
$$\text{Var}(W) = \frac{2}{n_{in}}$$

**Why these specific variances?** Keep variance of activations constant through layers:

If $y = Wx$ and $x \sim \mathcal{N}(0,1)$, then $\text{Var}(y_j) = n_{in} \cdot \text{Var}(W_{ij})$.

Setting $\text{Var}(W) = 1/n_{in}$ ensures $\text{Var}(y) = 1$ (for linear activations).

He doubles this to account for ReLU killing half the variance.

### Dropout as Bayesian Approximation

Applying dropout at test time with rate $p$:

$$\mathbf{y} = \frac{1}{1-p} \cdot (m \odot \mathbf{h}), \quad m_i \sim \text{Bernoulli}(1-p)$$

**Statistical view:** Dropout is an ensemble of $2^n$ weight-sharing networks. Test-time prediction approximates the ensemble average.

**Gal & Ghahramani (2016):** Dropout is equivalent to variational inference in a Bayesian neural network.

**MC Dropout for uncertainty:**

Run $T$ forward passes with dropout enabled. The variance across passes estimates **epistemic uncertainty**:

$$\text{Var}(\hat{y}) \approx \frac{1}{T}\sum_{t=1}^T \hat{y}_t^2 - \left(\frac{1}{T}\sum_{t=1}^T \hat{y}_t\right)^2$$

### Gradient Descent — Statistical View

**SGD (Stochastic Gradient Descent):**

$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta \nabla_\boldsymbol{\theta} \mathcal{L}(\boldsymbol{\theta}_t; \mathbf{x}^{(i)}, y^{(i)})$$

The SGD gradient is an **unbiased estimator** of the full gradient:

$$E[\nabla_\boldsymbol{\theta}\mathcal{L}(\boldsymbol{\theta}; \mathbf{x}^{(i)})] = \nabla_\boldsymbol{\theta}\mathcal{L}(\boldsymbol{\theta})$$

**Mini-batch gradient variance:**

$$\text{Var}\!\left(\frac{1}{B}\sum_{i \in \mathcal{B}} \nabla \ell_i\right) = \frac{\sigma^2_{\nabla}}{B}$$

Larger batch → lower gradient variance → more stable but less regularization.

### Adam Optimizer — Statistical Moments

**Adam** maintains exponentially weighted estimates of:
- **1st moment** (mean): $m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$
- **2nd moment** (uncentered variance): $v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$

**Bias-corrected estimates:**
$$\hat{m}_t = \frac{m_t}{1-\beta_1^t} \qquad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$$

**Update rule:**
$$\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \frac{\eta}{\sqrt{\hat{v}_t}+\varepsilon}\hat{m}_t$$

Intuition: $\hat{m}_t/\sqrt{\hat{v}_t}$ ≈ signal-to-noise ratio of the gradient.

### Loss Functions — Statistical Derivations

| Loss | Formula | Statistical Assumption | Distribution |
|---|---|---|---|
| MSE | $\frac{1}{n}\sum(y_i-\hat{y}_i)^2$ | Gaussian noise | $\mathcal{N}(\hat{y},\sigma^2)$ |
| MAE | $\frac{1}{n}\sum|y_i-\hat{y}_i|$ | Laplace noise | $\text{Laplace}(\hat{y},b)$ |
| Cross-entropy (binary) | $-[y\log\hat{p}+(1-y)\log(1-\hat{p})]$ | Bernoulli | $\text{Bernoulli}(\hat{p})$ |
| Cross-entropy (multi) | $-\sum y_k\log\hat{p}_k$ | Categorical | $\text{Cat}(\hat{\mathbf{p}})$ |
| Huber | $\begin{cases}\frac{1}{2}r^2 & |r|\leq\delta\\\delta|r|-\frac{\delta^2}{2} & |r|>\delta\end{cases}$ | Mixture Gaussian-Laplace | Robust |
| KL divergence | $\sum p\log(p/q)$ | Information-theoretic | — |

---

## 24. Statistical Learning Theory

### PAC Learning

**Probably Approximately Correct (PAC) Learning:**

A concept class $\mathcal{C}$ is PAC-learnable if there exists an algorithm $A$ such that for any distribution $\mathcal{D}$ and any $\varepsilon, \delta > 0$:

$$P\!\left(\text{error}(h) \leq \varepsilon\right) \geq 1-\delta$$

with sample complexity:
$$n \geq \frac{1}{\varepsilon}\left(\ln|\mathcal{H}| + \ln\frac{1}{\delta}\right)$$

### VC Dimension

The **Vapnik-Chervonenkis (VC) dimension** is the largest set of points that can be **shattered** by $\mathcal{H}$.

**Shatter:** $\mathcal{H}$ can correctly classify every possible labeling.

**Examples:**
- Linear classifiers in $\mathbb{R}^d$: $\text{VCdim} = d+1$
- Convex polygons in $\mathbb{R}^2$: $\text{VCdim} = \infty$

**Generalization bound (VC bound):**

$$\text{error}(h) \leq \hat{\text{error}}(h) + \sqrt{\frac{d\log(2n/d)+\log(4/\delta)}{n}}$$

where $d$ = VC dimension with probability $\geq 1-\delta$.

### Bias-Variance-Noise Decomposition

For squared loss, expected test error decomposes as:

$$E\!\left[(y-\hat{f}(x))^2\right] = \underbrace{\left(f(x) - E[\hat{f}(x)]\right)^2}_{\text{Bias}^2} + \underbrace{E\!\left[\left(\hat{f}(x)-E[\hat{f}(x)]\right)^2\right]}_{\text{Variance}} + \underbrace{\sigma^2_\varepsilon}_{\text{Irreducible noise}}$$

**Tradeoffs:**

| Model | Bias | Variance | Typical Issue |
|---|---|---|---|
| Too simple (underfitting) | High | Low | High training AND test error |
| Too complex (overfitting) | Low | High | Low training, high test error |
| Just right | Low | Low | Good generalization |

### Regularization and Effective Degrees of Freedom

Ridge regression effective degrees of freedom:
$$\text{df}(\lambda) = \text{tr}\!\left[\mathbf{X}(\mathbf{X}^T\mathbf{X}+\lambda\mathbf{I})^{-1}\mathbf{X}^T\right] = \sum_{j=1}^p \frac{d_j^2}{d_j^2+\lambda}$$

where $d_j$ are singular values of $\mathbf{X}$.

- $\lambda \to 0$: $\text{df} \to p$ (OLS, full complexity)
- $\lambda \to \infty$: $\text{df} \to 0$ (null model)

### AIC / BIC — Model Selection

**Akaike Information Criterion:**
$$\text{AIC} = 2k - 2\ln\hat{L}$$

**Bayesian Information Criterion:**
$$\text{BIC} = k\ln n - 2\ln\hat{L}$$

where $k$ = number of parameters, $\hat{L}$ = maximum likelihood.

**Lower is better.** BIC penalizes complexity more than AIC for $n \geq 8$.

**ML Use:** Comparing models with different numbers of parameters. Choosing regularization strength. Gaussian mixture model order selection.

### Rademacher Complexity

Measures the richness of a hypothesis class:

$$\hat{\mathfrak{R}}_n(\mathcal{F}) = E_\sigma\!\left[\sup_{f \in \mathcal{F}} \frac{1}{n}\sum_{i=1}^n \sigma_i f(x_i)\right]$$

where $\sigma_i \sim \text{Uniform}\{-1, +1\}$ (Rademacher variables).

**Generalization bound:**
$$E[\text{error}(h)] \leq \hat{\text{error}}(h) + 2\mathfrak{R}_n(\mathcal{H}) + \sqrt{\frac{\ln(1/\delta)}{2n}}$$

---

## 25. Master Formula Reference Sheet

### Central Tendency
$$\bar{x} = \frac{1}{n}\sum x_i \qquad \text{Median}(X) = x_{(n+1)/2} \qquad G = \left(\prod x_i\right)^{1/n} \qquad H = \frac{n}{\sum 1/x_i}$$

### Spread
$$s^2 = \frac{\sum(x_i-\bar{x})^2}{n-1} \qquad s = \sqrt{s^2} \qquad \text{IQR} = Q_3-Q_1 \qquad \text{CV}=\frac{\sigma}{\mu}$$

### Shape
$$\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3} \qquad \gamma_2 = \frac{E[(X-\mu)^4]}{\sigma^4} \qquad \kappa = \gamma_2-3$$

### Probability
$$P(A|B) = \frac{P(A\cap B)}{P(B)} \qquad P(A\cap B)=P(A)P(B|A) \qquad P(H|E)=\frac{P(E|H)P(H)}{P(E)}$$

### Distributions
$$\mathcal{N}:\; f(x)=\frac{1}{\sigma\sqrt{2\pi}}e^{-(x-\mu)^2/2\sigma^2} \qquad Z=\frac{X-\mu}{\sigma}$$
$$\text{Bin}: P(k)=\binom{n}{k}p^k(1-p)^{n-k},\; \mu=np,\; \sigma^2=np(1-p)$$
$$\text{Poisson}: P(k)=\frac{\lambda^k e^{-\lambda}}{k!},\; \mu=\sigma^2=\lambda$$

### Expectation & Variance
$$E[aX+b]=aE[X]+b \qquad \text{Var}(aX+b)=a^2\text{Var}(X) \qquad \text{Var}(X)=E[X^2]-(E[X])^2$$

### Covariance & Correlation
$$\text{Cov}(X,Y)=E[XY]-E[X]E[Y] \qquad \rho=\frac{\text{Cov}(X,Y)}{\sigma_X\sigma_Y} \qquad -1\leq\rho\leq 1$$

### Statistical Tests
$$z=\frac{\bar{x}-\mu_0}{\sigma/\sqrt{n}} \qquad t=\frac{\bar{x}-\mu_0}{s/\sqrt{n}} \qquad \chi^2=\sum\frac{(O-E)^2}{E} \qquad F=\frac{MS_B}{MS_W}$$

### Confidence Interval
$$\bar{x}\pm z_{\alpha/2}\frac{\sigma}{\sqrt{n}} \qquad \bar{x}\pm t_{\alpha/2,n-1}\frac{s}{\sqrt{n}}$$

### MLE / MAP
$$\hat{\theta}_{\text{MLE}}=\arg\max_\theta \sum\log f(x_i;\theta) \qquad \hat{\theta}_{\text{MAP}}=\arg\max_\theta\left[\ell(\theta)+\log P(\theta)\right]$$

### Regularization (MAP interpretation)
$$\|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|_2^2+\lambda\|\boldsymbol{\beta}\|_2^2 \leftrightarrow \text{Gaussian prior} \qquad \|\mathbf{y}-\mathbf{X}\boldsymbol{\beta}\|_2^2+\lambda\|\boldsymbol{\beta}\|_1 \leftrightarrow \text{Laplace prior}$$

### Information Theory
$$H(X)=-\sum P(x)\log P(x) \qquad D_{KL}(P\|Q)=\sum P\log\frac{P}{Q} \qquad H(P,Q)=H(P)+D_{KL}(P\|Q)$$

### Bias-Variance
$$E[(y-\hat{f})^2]=\text{Bias}^2(\hat{f})+\text{Var}(\hat{f})+\sigma^2_\varepsilon$$

### OLS
$$\hat{\boldsymbol{\beta}}=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y} \qquad R^2=1-\frac{SS_{\text{res}}}{SS_{\text{tot}}} \qquad \text{Var}(\hat{\boldsymbol{\beta}})=\sigma^2(\mathbf{X}^T\mathbf{X})^{-1}$$

### Bayes Deep Learning
$$\text{ELBO}=E_Q[\log P(\mathbf{X}|\theta)]-D_{KL}(Q(\theta)\|P(\theta)) \qquad \text{MC-Dropout Var}=\frac{1}{T}\sum\hat{y}_t^2-\bar{\hat{y}}^2$$

---

## 🗺️ Learning Roadmap

```
Week 1:  Sections 1–4    → Descriptive statistics, distributions, shape
Week 2:  Sections 5–7    → Probability theory, Bayes, random variables
Week 3:  Sections 8–9    → Discrete & continuous distributions
Week 4:  Sections 10–13  → Joint distributions, CLT, sampling theory
Week 5:  Sections 14–16  → Estimation, hypothesis testing, statistical tests
Week 6:  Sections 17–19  → Correlation, regression, information theory
Week 7:  Sections 20–21  → Bayesian statistics, multivariate analysis
Week 8:  Sections 22–25  → Resampling, DL connections, learning theory
```

---

## 📎 Key Concepts — Quick Mental Model

| Concept | One-line Intuition | ML Formula |
|---|---|---|
| Mean | Center of gravity | Loss gradient direction |
| Variance | Average squared surprise | Overfitting measure |
| CLT | Averages become Normal | Mini-batch gradients ≈ Normal |
| MLE | Most likely parameters | Training = maximize log-likelihood |
| MAP | MLE + prior belief | Training with regularization |
| KL Divergence | Cost of using wrong distribution | VAE + GAN objective |
| Cross-entropy | Encoding cost of true dist. using model | Classification loss |
| Bayes' Theorem | Update beliefs with evidence | Posterior inference |
| Bias-Variance | Underfitting vs Overfitting | Generalization tradeoff |
| Confidence Interval | Plausible range for parameter | Model uncertainty bound |
| p-value | Probability of data under $H_0$ | A/B test significance |

---

*Complete statistics reference for ML/DL — every formula, every derivation, every connection.*  
*From `$\bar{x}$` to ELBO, from Bernoulli to Bayesian Neural Networks.*
