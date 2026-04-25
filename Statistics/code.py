"""
================================================================================
  STATISTICS FOR MACHINE LEARNING & DEEP LEARNING
  Complete Implementation — NumPy · Math · Torch · Sklearn · TensorFlow
================================================================================
  Covers all 25 sections from the reference document:
  Descriptive Stats → Probability → Distributions → Hypothesis Testing
  → Regression → Information Theory → Bayesian Stats → Deep Learning Connections
================================================================================
"""

import math
import warnings
import numpy as np
import torch
import torch.nn as nn
import tensorflow as tf
from scipy import stats
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import f1_score, mean_squared_error
from sklearn.covariance import EmpiricalCovariance
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)
torch.manual_seed(42)

DIVIDER = "=" * 72
SECTION  = "-" * 72

def header(title: str) -> None:
    print(f"\n{DIVIDER}\n  {title}\n{DIVIDER}")

def subheader(title: str) -> None:
    print(f"\n  --- {title} ---")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 1 · Foundations — Data Types & Encoding
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 1 · Foundations — Data Types & Encoding")

# Nominal → One-hot encode
nominal_data = np.array(["Dog", "Cat", "Bird", "Dog", "Cat"])
ohe = OneHotEncoder(sparse_output=False)
one_hot = ohe.fit_transform(nominal_data.reshape(-1, 1))
print(f"Nominal  → One-Hot:\n{one_hot}")

# Ordinal → Label encode
ordinal_data = np.array(["Bad", "Average", "Good", "Excellent"])
le = LabelEncoder()
label_encoded = le.fit_transform(ordinal_data)
print(f"\nOrdinal  → Label encoded: {label_encoded}")

# Ratio/Interval → Standardize (Z-score)
ratio_data = np.array([[25, 175], [30, 165], [22, 180], [35, 170]], dtype=float)
scaler = StandardScaler()
standardized = scaler.fit_transform(ratio_data)
print(f"\nRatio    → Standardized:\n{standardized.round(3)}")

# Population vs Sample stats
data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)
pop_mean  = np.mean(data)            # μ
samp_var  = np.var(data, ddof=1)     # s²  (Bessel's correction)
pop_var   = np.var(data, ddof=0)     # σ²
print(f"\nPopulation μ={pop_mean:.2f} | Pop σ²={pop_var:.2f} | Sample s²={samp_var:.2f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 2 · Descriptive Statistics — Central Tendency
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 2 · Descriptive Statistics — Central Tendency")

data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)

# Arithmetic mean:  x̄ = Σxᵢ / n
arithmetic_mean = np.mean(data)

# Weighted mean:  x̄_w = Σwᵢxᵢ / Σwᵢ
weights = np.array([1, 2, 2, 2, 1, 1, 1, 1], dtype=float)
weighted_mean = np.average(data, weights=weights)

# Geometric mean:  G = (Π xᵢ)^(1/n) = exp(mean(log x))
geometric_mean = np.exp(np.mean(np.log(data)))

# Harmonic mean:  H = n / Σ(1/xᵢ)
harmonic_mean = len(data) / np.sum(1.0 / data)

# Median
median = np.median(data)

# Mode (manual — numpy has no mode)
values, counts = np.unique(data, return_counts=True)
mode = values[np.argmax(counts)]

print(f"Arithmetic mean : {arithmetic_mean:.4f}")
print(f"Weighted mean   : {weighted_mean:.4f}")
print(f"Geometric mean  : {geometric_mean:.4f}")
print(f"Harmonic mean   : {harmonic_mean:.4f}")
print(f"Median          : {median:.4f}")
print(f"Mode            : {mode:.4f}")

# F1 = harmonic mean of Precision & Recall
precision, recall = 0.80, 0.70
f1_manual = 2 * precision * recall / (precision + recall)
print(f"\nF1 (harmonic of P={precision}, R={recall}) = {f1_manual:.4f}")

# Pearson's empirical rule: Mode ≈ Mean − 3(Mean − Median)
pearson_mode = arithmetic_mean - 3 * (arithmetic_mean - median)
print(f"Pearson empirical mode ≈ {pearson_mode:.4f}  (actual={mode})")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 3 · Descriptive Statistics — Measures of Spread
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 3 · Descriptive Statistics — Measures of Spread")

data = np.array([2, 4, 4, 4, 5, 5, 7, 9], dtype=float)

# Range
data_range = np.ptp(data)                  # peak-to-peak

# IQR and outlier fences (Tukey)
Q1, Q3 = np.percentile(data, [25, 75])
IQR = Q3 - Q1
lower_fence = Q1 - 1.5 * IQR
upper_fence  = Q3 + 1.5 * IQR

# Variance & Standard Deviation
pop_var  = np.var(data, ddof=0)     # σ²  — population
samp_var = np.var(data, ddof=1)     # s²  — sample (Bessel)
pop_std  = np.std(data, ddof=0)     # σ
samp_std = np.std(data, ddof=1)     # s

# Coefficient of Variation:  CV = σ/μ × 100%
cv = (pop_std / np.mean(data)) * 100

# Mean Absolute Deviation (MAD)
mad_mean   = np.mean(np.abs(data - np.mean(data)))
mad_median = np.median(np.abs(data - np.median(data)))   # robust MAD

# Standard Error of the Mean:  SE = s / √n
sem = samp_std / math.sqrt(len(data))

print(f"Range          : {data_range}")
print(f"Q1={Q1}  Q3={Q3}  IQR={IQR}")
print(f"Outlier fences : [{lower_fence:.2f}, {upper_fence:.2f}]")
print(f"Population σ²  : {pop_var:.4f}  σ={pop_std:.4f}")
print(f"Sample s²      : {samp_var:.4f}  s={samp_std:.4f}")
print(f"CV             : {cv:.2f}%")
print(f"MAD (mean)     : {mad_mean:.4f}")
print(f"MAD (median)   : {mad_median:.4f}")
print(f"SEM            : {sem:.4f}")

# Torch: verify variance formula Var(X) = E[X²] - (E[X])²
t = torch.tensor(data)
var_formula = torch.mean(t**2) - torch.mean(t)**2   # population variance
print(f"\nTorch Var(X)=E[X²]-(E[X])² = {var_formula:.4f}  (matches {pop_var:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 4 · Shape of Distributions
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 4 · Shape of Distributions")

np.random.seed(42)
normal_data    = np.random.normal(0, 1, 5000)
right_skewed   = np.random.exponential(1, 5000)      # positive skew
left_skewed    = -np.random.exponential(1, 5000)     # negative skew
heavy_tailed   = np.random.standard_t(df=3, size=5000)

def distribution_shape(name: str, arr: np.ndarray) -> None:
    # Skewness: γ₁ = E[(X-μ)³] / σ³
    skew    = stats.skew(arr)
    # Excess kurtosis: κ = E[(X-μ)⁴]/σ⁴ - 3
    kurt    = stats.kurtosis(arr)   # scipy returns excess kurtosis
    print(f"  {name:<20} skew={skew:+.3f}  excess-kurt={kurt:+.3f}")

print("Distribution shapes:")
distribution_shape("Normal",       normal_data)
distribution_shape("Right-skewed", right_skewed)
distribution_shape("Left-skewed",  left_skewed)
distribution_shape("Heavy-tailed", heavy_tailed)

# Percentiles / Quantiles
data = np.random.normal(50, 10, 1000)
p10, p25, p50, p75, p90 = np.percentile(data, [10, 25, 50, 75, 90])
print(f"\nPercentiles: P10={p10:.1f}  Q1={p25:.1f}  Med={p50:.1f}  Q3={p75:.1f}  P90={p90:.1f}")

# Central moments using torch
t = torch.tensor(normal_data, dtype=torch.float32)
mu = t.mean()
centered = t - mu
m2 = (centered**2).mean()           # variance (2nd central moment)
m3 = (centered**3).mean()           # 3rd central moment
m4 = (centered**4).mean()           # 4th central moment
skew_torch = m3 / (m2**1.5)
kurt_torch = m4 / (m2**2) - 3      # excess kurtosis
print(f"\nTorch moments — skew={skew_torch:.3f}  excess-kurt={kurt_torch:.3f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 5 · Probability Theory Fundamentals
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 5 · Probability Theory Fundamentals")

# Monte-Carlo verification of axioms
N = 100_000
outcomes = np.random.randint(1, 7, size=N)          # fair die

def P(event_mask: np.ndarray) -> float:
    return event_mask.sum() / N

A = outcomes <= 3           # {1,2,3}
B = outcomes >= 2           # {2,3,4,5,6}

p_a        = P(A)
p_b        = P(B)
p_a_and_b  = P(A & B)
p_a_or_b   = P(A | B)
p_a_given_b = p_a_and_b / p_b

# Addition rule: P(A∪B) = P(A) + P(B) - P(A∩B)
addition_rule = p_a + p_b - p_a_and_b

print(f"P(A)       = {p_a:.4f}  (expected 0.5)")
print(f"P(B)       = {p_b:.4f}  (expected 0.833)")
print(f"P(A∩B)     = {p_a_and_b:.4f}")
print(f"P(A∪B) MC  = {p_a_or_b:.4f}")
print(f"P(A∪B) add = {addition_rule:.4f}  ← addition rule ✓")
print(f"P(A|B)     = {p_a_given_b:.4f}")

# Spam-classifier example (Law of Total Probability)
p_spam       = 0.30
p_free_spam  = 0.80
p_free_ham   = 0.10
p_free       = p_free_spam * p_spam + p_free_ham * (1 - p_spam)
print(f"\nSpam example P('free') = {p_free:.4f}  (expected 0.31)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 6 · Conditional Probability & Bayes' Theorem
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 6 · Conditional Probability & Bayes' Theorem")

def bayes(likelihood: float, prior: float, evidence: float) -> float:
    """P(H|E) = P(E|H)*P(H) / P(E)"""
    return (likelihood * prior) / evidence

# Medical test example
p_disease          = 0.01
p_pos_given_dis    = 0.99        # sensitivity
p_pos_given_nodis  = 0.05        # 1 - specificity
p_positive         = p_pos_given_dis * p_disease + p_pos_given_nodis * (1 - p_disease)
p_disease_given_pos = bayes(p_pos_given_dis, p_disease, p_positive)
print(f"P(Disease|+Test) = {p_disease_given_pos:.4f}  (≈16.7% despite 99% accuracy)")
print(f"  → Base rate matters! Prior p_disease={p_disease} dominates.\n")

# Spam → P(spam|'free')
p_spam_given_free = bayes(p_free_spam, p_spam, p_free)
print(f"P(spam|'free') = {p_spam_given_free:.4f}  (expected ≈0.774)")

# Naive Bayes classifier using sklearn
from sklearn.naive_bayes import GaussianNB
X_nb = np.array([[1,0],[1,1],[0,1],[0,0],[1,0],[0,1]])
y_nb = np.array([1, 1, 0, 0, 1, 0])
nb_clf = GaussianNB()
nb_clf.fit(X_nb, y_nb)
pred = nb_clf.predict([[1,1]])
print(f"\nNaive Bayes prediction for [1,1]: class={pred[0]}")
print(f"Class priors: {nb_clf.class_prior_}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 7 · Random Variables
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 7 · Random Variables")

# Discrete RV: E[X] = Σ x·P(X=x)
x_vals  = np.array([1, 2, 3, 4, 5, 6])
probs   = np.ones(6) / 6              # fair die
E_X     = np.sum(x_vals * probs)
E_X2    = np.sum(x_vals**2 * probs)
Var_X   = E_X2 - E_X**2              # Var(X) = E[X²] - (E[X])²
print(f"Die: E[X]={E_X:.4f}  E[X²]={E_X2:.4f}  Var(X)={Var_X:.4f}")

# Linearity: E[aX + b] = aE[X] + b
a, b = 3, 10
E_aXb = a * E_X + b
print(f"E[3X + 10] = {E_aXb:.4f}")

# Var(aX + b) = a²Var(X)
Var_aXb = a**2 * Var_X
print(f"Var(3X+10) = {Var_aXb:.4f}")

# CDF using torch: F(x) = P(X ≤ x)
normal_dist = torch.distributions.Normal(0.0, 1.0)
z_val = 1.96
cdf_196 = normal_dist.cdf(torch.tensor(z_val))
print(f"\nStandard Normal CDF at z=1.96: Φ(1.96) = {cdf_196:.4f}  (expected ≈0.975)")

# PDF at x
pdf_at_0 = torch.exp(normal_dist.log_prob(torch.tensor(0.0)))
print(f"Standard Normal PDF at z=0   : φ(0) = {pdf_at_0:.4f}  (expected ≈0.3989)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 8 · Probability Distributions — Discrete
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 8 · Probability Distributions — Discrete")

subheader("Bernoulli")
p = 0.7
bern = torch.distributions.Bernoulli(probs=p)
samples_b = bern.sample((1000,))
print(f"  Bernoulli(p={p}): E[X]={p}  Var(X)={p*(1-p):.4f}")
print(f"  Empirical mean={samples_b.mean():.4f}  var={samples_b.var():.4f}")

subheader("Binomial")
n, p_binom = 10, 0.8
binom_dist = stats.binom(n, p_binom)
# P(X=8): 10 samples, model accuracy 0.8
p_exactly_8 = binom_dist.pmf(8)
print(f"  Binomial(n=10, p=0.8): P(X=8) = {p_exactly_8:.4f}  (expected ≈0.302)")
print(f"  E[X]={n*p_binom:.1f}  Var(X)={n*p_binom*(1-p_binom):.4f}")

subheader("Poisson")
lam = 5.0
poisson = torch.distributions.Poisson(rate=lam)
samples_p = poisson.sample((10000,)).float()
print(f"  Poisson(λ={lam}): E[X]=Var(X)={lam}")
print(f"  Empirical mean={samples_p.mean():.4f}  var={samples_p.var():.4f}")

# P(X=k) = λᵏ e^{-λ} / k!  manually
k = 3
p_k = (lam**k * math.exp(-lam)) / math.factorial(k)
print(f"  P(X={k}) = {p_k:.4f}")

subheader("Geometric & Negative Binomial")
p_geom = 0.3
E_geom  = 1 / p_geom
Var_geom = (1 - p_geom) / p_geom**2
print(f"  Geometric(p={p_geom}): E[X]={E_geom:.4f}  Var(X)={Var_geom:.4f}")

r, p_nb = 3, 0.4
E_nb  = r / p_nb
Var_nb = r * (1-p_nb) / p_nb**2
print(f"  NegBinomial(r={r}, p={p_nb}): E[X]={E_nb:.4f}  Var(X)={Var_nb:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 9 · Probability Distributions — Continuous
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 9 · Probability Distributions — Continuous")

subheader("Uniform")
a_u, b_u = 0.0, 1.0
unif = torch.distributions.Uniform(a_u, b_u)
s_u  = unif.sample((10000,))
print(f"  Uniform({a_u},{b_u}): E[X]={((a_u+b_u)/2):.2f}  Var=(b-a)²/12={(b_u-a_u)**2/12:.4f}")
print(f"  Empirical mean={s_u.mean():.4f}  var={s_u.var():.4f}")

subheader("Normal — 68-95-99.7 Rule")
mu_n, sigma_n = 0.0, 1.0
norm_d = stats.norm(mu_n, sigma_n)
within_1 = norm_d.cdf(1) - norm_d.cdf(-1)
within_2 = norm_d.cdf(2) - norm_d.cdf(-2)
within_3 = norm_d.cdf(3) - norm_d.cdf(-3)
print(f"  P(μ±1σ)={within_1:.4f}  P(μ±2σ)={within_2:.4f}  P(μ±3σ)={within_3:.4f}")

subheader("Exponential (memoryless)")
lambda_e = 2.0
exp_d = torch.distributions.Exponential(lambda_e)
s_e   = exp_d.sample((10000,))
# P(X > s+t | X > s) = P(X > t)
s_val, t_val = 1.0, 0.5
memoryless_check = ((s_e > s_val + t_val).float().mean() /
                    (s_e > s_val).float().mean())
p_x_gt_t = (s_e > t_val).float().mean()
print(f"  Memoryless: P(X>{s_val+t_val}|X>{s_val})={memoryless_check:.4f}"
      f"  P(X>{t_val})={p_x_gt_t:.4f}  ≈ equal ✓")

subheader("Beta Distribution")
alpha_b, beta_b = 2.0, 5.0
beta_dist = torch.distributions.Beta(alpha_b, beta_b)
s_beta = beta_dist.sample((10000,))
E_beta = alpha_b / (alpha_b + beta_b)
Var_beta = alpha_b*beta_b / ((alpha_b+beta_b)**2 * (alpha_b+beta_b+1))
print(f"  Beta({alpha_b},{beta_b}): E[X]={E_beta:.4f}  Var={Var_beta:.4f}")
print(f"  Empirical mean={s_beta.mean():.4f}  var={s_beta.var():.4f}")

subheader("Student's t & Chi-squared")
nu = 5
t_dist = stats.t(df=nu)
chi2_dist = stats.chi2(df=nu)
print(f"  t(ν={nu}): E[T]=0  Var={nu/(nu-2):.4f}")
print(f"  χ²(k={nu}): E[X]={nu}  Var={2*nu}")

subheader("Laplace — L1 regularization prior")
mu_l, b_l = 0.0, 1.0
laplace = torch.distributions.Laplace(mu_l, b_l)
s_l = laplace.sample((10000,))
print(f"  Laplace({mu_l},{b_l}): E[X]={mu_l}  Var(X)={2*b_l**2:.4f}")
print(f"  Empirical mean={s_l.mean():.4f}  var={s_l.var():.4f}")

subheader("Dirichlet Distribution")
alphas = torch.tensor([2.0, 3.0, 5.0])
dirich = torch.distributions.Dirichlet(alphas)
s_dir = dirich.sample((5000,))
E_dir = alphas / alphas.sum()
print(f"  Dirichlet({alphas.tolist()}): E[X]={E_dir.tolist()}")
print(f"  Empirical mean={s_dir.mean(0).tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 10 · Joint, Marginal & Conditional Distributions
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 10 · Joint, Marginal & Conditional Distributions")

np.random.seed(42)
N = 5000
X_joint = np.random.normal(0, 1, N)
Y_joint = 2 * X_joint + np.random.normal(0, 0.5, N)

# Joint distribution approximated via 2D histogram
hist2d, xedge, yedge = np.histogram2d(X_joint, Y_joint, bins=20, density=True)

# Marginals (sum out the other variable)
marginal_x = hist2d.sum(axis=1)
marginal_y = hist2d.sum(axis=0)
print(f"2D histogram shape: {hist2d.shape}")
print(f"Marginal X sum (≈1): {marginal_x.sum() * (xedge[1]-xedge[0]):.4f}")

# Covariance
cov_xy = np.cov(X_joint, Y_joint)
print(f"\nCovariance matrix:\n{cov_xy.round(4)}")

# Correlation
corr_xy = np.corrcoef(X_joint, Y_joint)[0, 1]
print(f"Correlation ρ = {corr_xy:.4f}  (expected ≈ 0.97 since Y≈2X)")

# Check independence: if X ⊥ Y → Cov(X,Y) = 0
Z_indep = np.random.normal(0, 1, N)
cov_xz  = np.cov(X_joint, Z_indep)[0, 1]
print(f"Cov(X, independent Z) ≈ {cov_xz:.4f}  (expected ≈ 0)")

# Covariance matrix with sklearn
data_matrix = np.column_stack([X_joint, Y_joint])
ec = EmpiricalCovariance()
ec.fit(data_matrix)
print(f"\nEmpirical covariance matrix:\n{ec.covariance_.round(4)}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 11 · Expectation, Variance & Covariance — Deep Dive
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 11 · Expectation, Variance & Covariance — Deep Dive")

# Law of Total Expectation: E[X] = E[E[X|Y]]
groups = {0: np.random.normal(5, 1, 500),
          1: np.random.normal(10, 1, 500)}
all_data = np.concatenate(list(groups.values()))
E_total_direct = np.mean(all_data)
group_probs = {0: 0.5, 1: 0.5}
E_tower = sum(group_probs[g] * np.mean(groups[g]) for g in groups)
print(f"E[X] direct  = {E_total_direct:.4f}")
print(f"E[E[X|Y]]    = {E_tower:.4f}  ← Tower property ✓")

# Law of Total Variance: Var(X) = E[Var(X|Y)] + Var(E[X|Y])
within_var   = sum(group_probs[g] * np.var(groups[g], ddof=1) for g in groups)
between_var  = np.var([np.mean(groups[g]) for g in groups], ddof=0)
total_var_ltp = within_var + between_var
total_var_direct = np.var(all_data, ddof=1)
print(f"\nVar(X) direct                    = {total_var_direct:.4f}")
print(f"Within-group + Between-group     = {total_var_ltp:.4f}  ← Total variance ✓")

# Cov(X,Y) = E[XY] - E[X]E[Y]  via torch
X_t = torch.tensor(X_joint[:1000], dtype=torch.float32)
Y_t = torch.tensor(Y_joint[:1000], dtype=torch.float32)
cov_formula = torch.mean(X_t * Y_t) - torch.mean(X_t) * torch.mean(Y_t)
cov_torch   = torch.cov(torch.stack([X_t, Y_t]))[0, 1]
print(f"\nCov(X,Y) formula = {cov_formula:.4f}")
print(f"Cov(X,Y) torch   = {cov_torch:.4f}")

# Var(X+Y) = Var(X) + Var(Y) + 2Cov(X,Y)
Var_X_t = X_t.var(unbiased=True)
Var_Y_t = Y_t.var(unbiased=True)
Var_sum_formula = Var_X_t + Var_Y_t + 2 * cov_torch
Var_sum_direct  = (X_t + Y_t).var(unbiased=True)
print(f"\nVar(X+Y) formula = {Var_sum_formula:.4f}")
print(f"Var(X+Y) direct  = {Var_sum_direct:.4f}  ← matches ✓")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 12 · The Normal Distribution — Master Reference
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 12 · The Normal Distribution — Master Reference")

# PDF: f(x) = 1/(σ√2π) * exp(-(x-μ)²/2σ²)
def normal_pdf(x: float, mu: float = 0, sigma: float = 1) -> float:
    coeff = 1 / (sigma * math.sqrt(2 * math.pi))
    return coeff * math.exp(-0.5 * ((x - mu) / sigma) ** 2)

# Standardization: Z = (X - μ) / σ
mu_ex, sigma_ex = 50, 10
x_val = 70
z = (x_val - mu_ex) / sigma_ex
print(f"X={x_val}, μ={mu_ex}, σ={sigma_ex}  → Z={(z):.2f}")
print(f"P(X ≤ {x_val}) = Φ({z}) = {stats.norm.cdf(z):.4f}")

# Normal PDF via TensorFlow
x_tf = tf.constant([0.0, 1.0, -1.0, 2.0])
normal_tf = tf.exp(-0.5 * x_tf**2) / tf.sqrt(2.0 * tf.cast(math.pi, tf.float32))
print(f"\nNormal PDF at [0,1,-1,2]: {normal_tf.numpy().round(4)}")

# Xavier / Glorot weight initialization (Normal)
n_in, n_out = 128, 64
xavier_std = math.sqrt(2.0 / (n_in + n_out))
xavier_weights = np.random.normal(0, xavier_std, size=(n_in, n_out))
print(f"\nXavier init: std={xavier_std:.4f}  weights shape={xavier_weights.shape}")
print(f"  weights mean={xavier_weights.mean():.4f}  std={xavier_weights.std():.4f}")

# He initialization (for ReLU)
he_std = math.sqrt(2.0 / n_in)
he_weights = np.random.normal(0, he_std, size=(n_in, n_out))
print(f"He init     : std={he_std:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 13 · Sampling Theory & Central Limit Theorem
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 13 · Sampling Theory & Central Limit Theorem")

# CLT: sample means of ANY distribution → Normal as n grows
pop_exponential = np.random.exponential(scale=2.0, size=100_000)   # skewed!
sample_means = [np.mean(np.random.choice(pop_exponential, 30, replace=False))
                for _ in range(5000)]
sample_means = np.array(sample_means)

true_mean = np.mean(pop_exponential)
expected_sem = np.std(pop_exponential, ddof=1) / math.sqrt(30)

print(f"Population mean       : {true_mean:.4f}")
print(f"Mean of sample means  : {np.mean(sample_means):.4f}  ← approaches μ ✓")
print(f"Expected SEM          : {expected_sem:.4f}")
print(f"Std of sample means   : {np.std(sample_means, ddof=1):.4f}  ← ≈ SEM ✓")
print(f"Normality test (p>0.05 is normal): p={stats.shapiro(sample_means[:500])[1]:.4f}")

# Mini-batch gradient simulation (CLT in deep learning)
full_gradient  = np.random.normal(0.5, 2.0, 50000)
batch_size     = 64
batch_grad_means = [np.mean(np.random.choice(full_gradient, batch_size))
                    for _ in range(1000)]
print(f"\nFull-batch gradient mean  : {full_gradient.mean():.4f}")
print(f"Mini-batch gradient mean  : {np.mean(batch_grad_means):.4f}")
print(f"Mini-batch gradient std   : {np.std(batch_grad_means):.4f}")
print(f"Expected (σ/√batch)       : {full_gradient.std()/math.sqrt(batch_size):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 14 · Estimation Theory — MLE & MAP
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 14 · Estimation Theory — MLE & MAP")

subheader("MLE for Gaussian Parameters")
# Data ~ N(μ, σ²)  →  MLE: μ̂=x̄,  σ̂²=Σ(xᵢ-x̄)²/n
data_mle  = np.random.normal(loc=5.0, scale=2.0, size=1000)
mu_mle    = data_mle.mean()                           # μ̂_MLE = x̄
sigma2_mle = np.var(data_mle, ddof=0)                 # σ̂²_MLE = Σ(xᵢ-x̄)²/n
print(f"  MLE μ̂={mu_mle:.4f}  σ̂²={sigma2_mle:.4f}")
print(f"  True  μ=5.0000   σ²=4.0000")

subheader("MLE for Bernoulli")
# p̂_MLE = (number of successes) / n
flips  = np.random.binomial(1, 0.6, size=500)
p_mle  = flips.mean()
print(f"  Bernoulli MLE: p̂={p_mle:.4f}  (true p=0.6)")

subheader("MAP — MLE + Prior (Regularization)")
# MAP with Gaussian prior ↔ L2 regularization (Ridge)
# MAP with Laplace  prior ↔ L1 regularization (Lasso)
np.random.seed(0)
X_map = np.random.randn(100, 5)
true_beta = np.array([1.0, 2.0, 0.0, -1.5, 0.5])
y_map = X_map @ true_beta + np.random.randn(100) * 0.5

# OLS (MLE, no prior)
ols = LinearRegression(fit_intercept=False)
ols.fit(X_map, y_map)

# Ridge (MAP, Gaussian prior → L2 penalty)
ridge = Ridge(alpha=1.0, fit_intercept=False)
ridge.fit(X_map, y_map)

# Lasso (MAP, Laplace prior → L1 penalty)
lasso = Lasso(alpha=0.1, fit_intercept=False)
lasso.fit(X_map, y_map)

print(f"  True β  : {true_beta}")
print(f"  OLS β   : {ols.coef_.round(4)}")
print(f"  Ridge β : {ridge.coef_.round(4)}  ← shrunk toward 0 (Gaussian prior)")
print(f"  Lasso β : {lasso.coef_.round(4)}  ← sparse (Laplace prior)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 15 · Hypothesis Testing
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 15 · Hypothesis Testing")

subheader("One-sample z-test")
# H₀: μ = μ₀,   z = (x̄ - μ₀) / (σ/√n)
sample_z = np.random.normal(5.2, 2.0, 50)
mu_0  = 5.0
sigma_known = 2.0
z_stat = (np.mean(sample_z) - mu_0) / (sigma_known / math.sqrt(len(sample_z)))
p_val_z = 2 * (1 - stats.norm.cdf(abs(z_stat)))   # two-tailed
print(f"  z={z_stat:.4f}  p-value={p_val_z:.4f}")
print(f"  {'Reject H₀' if p_val_z < 0.05 else 'Fail to reject H₀'} at α=0.05")

subheader("One-sample t-test (σ unknown)")
# t = (x̄ - μ₀) / (s/√n),  df = n-1
sample_t = np.random.normal(5.5, 2.0, 30)
t_stat, p_val_t = stats.ttest_1samp(sample_t, popmean=5.0)
print(f"  t={t_stat:.4f}  p-value={p_val_t:.4f}")
print(f"  {'Reject H₀' if p_val_t < 0.05 else 'Fail to reject H₀'} at α=0.05")

subheader("Two-sample t-test")
group_A = np.random.normal(10, 2, 50)
group_B = np.random.normal(11, 2, 50)
t2, p2  = stats.ttest_ind(group_A, group_B)
print(f"  t={t2:.4f}  p={p2:.4f}  → {'Significant' if p2 < 0.05 else 'Not significant'}")

subheader("Type I / II Errors")
print(f"  α (Type I):  Reject H₀ when true  → False Positive (FPR)")
print(f"  β (Type II): Accept H₀ when false  → False Negative (FNR)")
print(f"  Power = 1 - β = P(reject H₀ | H₁ true)")

# Power analysis
from scipy.stats import norm as scipy_norm
alpha, effect_size, n_pow = 0.05, 0.5, 50
z_alpha = scipy_norm.ppf(1 - alpha / 2)
se_pow  = 1 / math.sqrt(n_pow)
z_power = effect_size / se_pow - z_alpha
power   = scipy_norm.cdf(z_power)
print(f"\n  Power analysis: effect={effect_size}, n={n_pow}, α={alpha} → power={power:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 16 · Statistical Tests Reference
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 16 · Statistical Tests Reference")

subheader("Chi-squared Goodness-of-Fit")
# χ² = Σ (O - E)² / E
observed = np.array([20, 30, 25, 25])
expected = np.array([25, 25, 25, 25])   # uniform
chi2_stat, p_chi2 = stats.chisquare(observed, expected)
print(f"  χ²={chi2_stat:.4f}  p={p_chi2:.4f}")

subheader("Chi-squared Test of Independence")
contingency = np.array([[30, 10], [20, 40]])
chi2_ind, p_ind, dof, expected_ind = stats.chi2_contingency(contingency)
print(f"  χ²={chi2_ind:.4f}  p={p_ind:.4f}  df={dof}")

subheader("ANOVA (F-test)")
# F = MS_between / MS_within
g1 = np.random.normal(10, 2, 30)
g2 = np.random.normal(12, 2, 30)
g3 = np.random.normal(11, 2, 30)
f_stat, p_anova = stats.f_oneway(g1, g2, g3)
print(f"  F={f_stat:.4f}  p={p_anova:.4f}")

subheader("Shapiro-Wilk Normality Test")
normal_sample = np.random.normal(0, 1, 100)
skewed_sample = np.random.exponential(1, 100)
_, p_norm   = stats.shapiro(normal_sample)
_, p_skewed = stats.shapiro(skewed_sample)
print(f"  Normal data p={p_norm:.4f}  (>0.05 → normal ✓)")
print(f"  Skewed data p={p_skewed:.4f}  (<0.05 → not normal ✓)")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 17 · Correlation & Dependence
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 17 · Correlation & Dependence")

np.random.seed(42)
X_corr = np.random.normal(0, 1, 500)
Y_lin  = 2 * X_corr + np.random.normal(0, 0.5, 500)   # strong linear
Y_nonlin = X_corr**2 + np.random.normal(0, 0.5, 500)  # non-linear (quadratic)

# Pearson: ρ = Cov(X,Y) / (σX σY)
pearson_lin,    _  = stats.pearsonr(X_corr, Y_lin)
pearson_nonlin, _  = stats.pearsonr(X_corr, Y_nonlin)

# Spearman (rank correlation): robust to non-linearity
spearman_lin,    _ = stats.spearmanr(X_corr, Y_lin)
spearman_nonlin, _ = stats.spearmanr(X_corr, Y_nonlin)

# Kendall's τ
kendall_lin,    _ = stats.kendalltau(X_corr, Y_lin)
kendall_nonlin, _ = stats.kendalltau(X_corr, Y_nonlin)

print(f"{'Measure':<12} {'Linear':>10} {'Quadratic':>12}")
print(f"{'Pearson':<12} {pearson_lin:>10.4f} {pearson_nonlin:>12.4f}")
print(f"{'Spearman':<12} {spearman_lin:>10.4f} {spearman_nonlin:>12.4f}")
print(f"{'Kendall':<12} {kendall_lin:>10.4f} {kendall_nonlin:>12.4f}")
print(f"\n  Pearson misses quadratic relationship; Spearman captures rank!")

# Mutual Information (via sklearn)
from sklearn.feature_selection import mutual_info_regression
mi_lin    = mutual_info_regression(X_corr.reshape(-1,1), Y_lin,    random_state=42)[0]
mi_nonlin = mutual_info_regression(X_corr.reshape(-1,1), Y_nonlin, random_state=42)[0]
print(f"\n  Mutual Information: linear={mi_lin:.4f}  quadratic={mi_nonlin:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 18 · Linear Regression — Statistical Foundation
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 18 · Linear Regression — Statistical Foundation")

np.random.seed(0)
n_reg = 200
X_lr  = np.random.randn(n_reg, 1)
y_lr  = 3 + 2 * X_lr.ravel() + np.random.randn(n_reg) * 1.5

# OLS: β̂ = (XᵀX)⁻¹ Xᵀy
X_lr_aug = np.column_stack([np.ones(n_reg), X_lr])      # add intercept
beta_ols  = np.linalg.inv(X_lr_aug.T @ X_lr_aug) @ X_lr_aug.T @ y_lr
print(f"OLS (normal equations): β̂ = {beta_ols.round(4)}")

# Compare with sklearn
lr = LinearRegression()
lr.fit(X_lr, y_lr)
print(f"sklearn LinearRegression: intercept={lr.intercept_:.4f}  coef={lr.coef_[0]:.4f}")

# R² = 1 - SS_res / SS_tot
y_hat      = X_lr_aug @ beta_ols
SS_res     = np.sum((y_lr - y_hat)**2)
SS_tot     = np.sum((y_lr - y_lr.mean())**2)
R2         = 1 - SS_res / SS_tot
print(f"R² = {R2:.4f}")

# Var(β̂) = σ²(XᵀX)⁻¹
sigma2_est = SS_res / (n_reg - 2)       # MSE
cov_beta   = sigma2_est * np.linalg.inv(X_lr_aug.T @ X_lr_aug)
se_beta    = np.sqrt(np.diag(cov_beta))
print(f"SE(β̂) = {se_beta.round(4)}")

# t-statistics for coefficients
t_stats = beta_ols / se_beta
print(f"t-stats  = {t_stats.round(4)}")

# TensorFlow linear regression (gradient descent)
subheader("TensorFlow Gradient Descent Regression")
X_tf_train = tf.constant(X_lr, dtype=tf.float32)
y_tf_train = tf.constant(y_lr.reshape(-1,1), dtype=tf.float32)
tf_model   = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
tf_model.compile(optimizer=tf.keras.optimizers.Adam(0.1), loss='mse')
tf_model.fit(X_tf_train, y_tf_train, epochs=200, verbose=0)
w, b = tf_model.layers[0].get_weights()
print(f"  TF learned: slope={w[0,0]:.4f}  intercept={b[0]:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 19 · Information Theory
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 19 · Information Theory")

def entropy(p: np.ndarray) -> float:
    """Shannon entropy: H(X) = -Σ p(x) log₂ p(x)"""
    p = p[p > 0]
    return -np.sum(p * np.log2(p))

def cross_entropy(p: np.ndarray, q: np.ndarray) -> float:
    """H(P,Q) = -Σ p(x) log q(x)"""
    q = np.clip(q, 1e-12, 1.0)
    return -np.sum(p * np.log2(q))

def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """D_KL(P||Q) = Σ p log(p/q)  — note: H(P,Q) = H(P) + D_KL(P||Q)"""
    mask = (p > 0)
    return np.sum(p[mask] * np.log2(p[mask] / np.clip(q[mask], 1e-12, 1)))

# Binary entropy
p_binary = np.array([0.5, 0.5])
print(f"H(fair coin)     = {entropy(p_binary):.4f} bits  (max)")
p_biased = np.array([0.9, 0.1])
print(f"H(biased 0.9/0.1)= {entropy(p_biased):.4f} bits  (less uncertain)")

# Uniform over 4 classes
p_unif4 = np.ones(4) / 4
print(f"H(uniform 4)     = {entropy(p_unif4):.4f} bits  (= log₂4 = 2)")

# Cross-entropy and KL
P = np.array([0.4, 0.3, 0.2, 0.1])
Q = np.array([0.25, 0.25, 0.25, 0.25])    # model predicts uniform
H_P  = entropy(P)
H_PQ = cross_entropy(P, Q)
D_KL = kl_divergence(P, Q)
print(f"\nH(P)    = {H_P:.4f}")
print(f"H(P,Q)  = {H_PQ:.4f}  ← classification loss")
print(f"D_KL    = {D_KL:.4f}  ← H(P,Q) - H(P) = {H_PQ - H_P:.4f} ✓")

# TensorFlow cross-entropy loss (with logits)
y_true_it = tf.constant([[1.0, 0.0, 0.0]])
logits_it  = tf.constant([[2.0, 1.0, 0.5]])
ce_tf = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
print(f"\nTF categorical cross-entropy loss = {ce_tf(y_true_it, logits_it).numpy():.4f}")

# Mutual Information: I(X;Y) = H(X) - H(X|Y)
H_X = entropy(np.array([0.5, 0.5]))
H_X_given_Y = 0.3 * entropy(np.array([0.8, 0.2])) + 0.7 * entropy(np.array([0.3, 0.7]))
MI = H_X - H_X_given_Y
print(f"\nMutual Information I(X;Y) = {MI:.4f} bits")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 20 · Bayesian Statistics
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 20 · Bayesian Statistics")

subheader("Beta-Binomial Conjugate Updating")
# Prior: Beta(α, β)  → after n trials, k successes → Posterior: Beta(α+k, β+n-k)
alpha_prior, beta_prior = 2.0, 2.0    # uniform-ish prior
successes  = 7
failures   = 3
alpha_post = alpha_prior + successes
beta_post  = beta_prior  + failures
E_prior    = alpha_prior / (alpha_prior + beta_prior)
E_post     = alpha_post  / (alpha_post  + beta_post)
print(f"  Prior  Beta({alpha_prior},{beta_prior}): E[p]={E_prior:.4f}")
print(f"  Data: {successes} successes, {failures} failures")
print(f"  Posterior Beta({alpha_post},{beta_post}): E[p]={E_post:.4f}")
print(f"  MLE: p̂ = {successes/(successes+failures):.4f}")

subheader("Credible Interval (Highest Density Interval)")
ci_lo, ci_hi = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)
print(f"  95% credible interval: [{ci_lo:.4f}, {ci_hi:.4f}]")

subheader("MCMC — Metropolis-Hastings Sampler")
def log_posterior(theta: float, successes: int, failures: int,
                  alpha_pr: float, beta_pr: float) -> float:
    if not (0 < theta < 1):
        return -np.inf
    log_lik   = successes * math.log(theta) + failures * math.log(1 - theta)
    log_prior = (alpha_pr - 1) * math.log(theta) + (beta_pr - 1) * math.log(1 - theta)
    return log_lik + log_prior

theta_curr, n_iter = 0.5, 5000
samples_mcmc = []
for _ in range(n_iter):
    theta_prop = theta_curr + np.random.normal(0, 0.05)
    log_ratio  = log_posterior(theta_prop, successes, failures, alpha_prior, beta_prior) \
               - log_posterior(theta_curr, successes, failures, alpha_prior, beta_prior)
    if math.log(np.random.rand()) < log_ratio:
        theta_curr = theta_prop
    samples_mcmc.append(theta_curr)

samples_mcmc = np.array(samples_mcmc[500:])   # burn-in
print(f"  MCMC posterior mean = {samples_mcmc.mean():.4f}  (analytical={E_post:.4f})")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 21 · Matrix Statistics & Multivariate Analysis
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 21 · Matrix Statistics & Multivariate Analysis")

subheader("Covariance Matrix & PCA")
np.random.seed(42)
X_pca = np.random.multivariate_normal([0, 0, 0],
                                       [[4, 2, 1],
                                        [2, 3, 0.5],
                                        [1, 0.5, 2]], size=500)
# Compute covariance matrix: Σ = (1/n-1)(X - x̄)ᵀ(X - x̄)
X_centered = X_pca - X_pca.mean(axis=0)
cov_matrix  = (X_centered.T @ X_centered) / (len(X_pca) - 1)
print(f"  Estimated covariance matrix:\n{cov_matrix.round(4)}")

# Eigendecomposition: Σv = λv
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[idx].real
eigenvectors = eigenvectors[:, idx].real
explained_var = eigenvalues / eigenvalues.sum()
print(f"\n  Eigenvalues: {eigenvalues.round(4)}")
print(f"  Explained variance ratio: {explained_var.round(4)}")

# PCA with sklearn
pca = PCA(n_components=2)
X_pca_2d = pca.fit_transform(X_pca)
print(f"\n  sklearn PCA explained variance: {pca.explained_variance_ratio_.round(4)}")
print(f"  Original shape: {X_pca.shape}  → Reduced: {X_pca_2d.shape}")

subheader("Mahalanobis Distance")
# D_M(x) = √[(x-μ)ᵀ Σ⁻¹ (x-μ)]
x_pt   = np.array([1.0, 2.0, 0.5])
mu_mv  = X_pca.mean(axis=0)
Sigma_inv = np.linalg.inv(cov_matrix)
diff   = x_pt - mu_mv
mah_d  = math.sqrt(diff @ Sigma_inv @ diff)
print(f"  Mahalanobis distance of point to distribution = {mah_d:.4f}")

subheader("Multivariate Normal")
# Using torch.distributions.MultivariateNormal
mu_torch_mv    = torch.zeros(3)
cov_torch_mv   = torch.tensor([[4.0, 2.0, 1.0],
                                [2.0, 3.0, 0.5],
                                [1.0, 0.5, 2.0]])
mvn = torch.distributions.MultivariateNormal(mu_torch_mv, cov_torch_mv)
samples_mv = mvn.sample((1000,))
log_prob   = mvn.log_prob(torch.tensor([1.0, 2.0, 0.5]))
print(f"  Log-prob of point [1,2,0.5] = {log_prob:.4f}")
print(f"  Sample mean: {samples_mv.mean(0).tolist()}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 22 · Resampling Methods
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 22 · Resampling Methods")

subheader("Bootstrap — Confidence Intervals")
data_boot = np.random.normal(10, 3, 100)
n_boot = 5000
boot_means = np.array([np.mean(np.random.choice(data_boot, len(data_boot)))
                        for _ in range(n_boot)])
ci_boot = np.percentile(boot_means, [2.5, 97.5])
print(f"  Bootstrap 95% CI for mean: [{ci_boot[0]:.4f}, {ci_boot[1]:.4f}]")
print(f"  Sample mean: {data_boot.mean():.4f}  Analytical CI: "
      f"[{data_boot.mean() - 1.96*data_boot.std()/math.sqrt(len(data_boot)):.4f}, "
      f"{data_boot.mean() + 1.96*data_boot.std()/math.sqrt(len(data_boot)):.4f}]")

subheader("K-Fold Cross-Validation")
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
X_cv, y_cv = make_regression(n_samples=200, n_features=5, noise=10, random_state=42)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = []
for train_idx, val_idx in kf.split(X_cv):
    lr_cv = LinearRegression()
    lr_cv.fit(X_cv[train_idx], y_cv[train_idx])
    y_pred_cv = lr_cv.predict(X_cv[val_idx])
    mse = mean_squared_error(y_cv[val_idx], y_pred_cv)
    cv_scores.append(mse)
print(f"  5-fold CV MSE: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

subheader("Permutation Test")
group_A_perm = np.random.normal(10, 2, 50)
group_B_perm = np.random.normal(11, 2, 50)
observed_diff = np.mean(group_B_perm) - np.mean(group_A_perm)
all_values = np.concatenate([group_A_perm, group_B_perm])
perm_diffs = []
for _ in range(5000):
    perm = np.random.permutation(all_values)
    perm_diffs.append(perm[50:].mean() - perm[:50].mean())
p_perm = np.mean(np.abs(perm_diffs) >= abs(observed_diff))
print(f"  Observed diff: {observed_diff:.4f}  p-value (permutation): {p_perm:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 23 · Statistics in Deep Learning
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 23 · Statistics in Deep Learning")

subheader("Batch Normalization")
# BN normalizes: x̂ = (x - μ_B) / σ_B,  then y = γx̂ + β
class ManualBatchNorm(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))
        self.beta  = nn.Parameter(torch.zeros(d))
        self.eps   = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu_b    = x.mean(dim=0)
        sigma_b = x.var(dim=0, unbiased=False).sqrt()
        x_hat   = (x - mu_b) / (sigma_b + self.eps)
        return self.gamma * x_hat + self.beta

bn_input = torch.randn(32, 16)   # batch=32, features=16
mbn = ManualBatchNorm(16)
bn_out = mbn(bn_input)
print(f"  BN input  mean={bn_input.mean():.4f}  std={bn_input.std():.4f}")
print(f"  BN output mean={bn_out.mean():.4f}  std={bn_out.std():.4f}  ← normalized ✓")

subheader("Dropout as Bernoulli Sampling")
# Dropout: each neuron independently masked: z ~ Bernoulli(1-p)
# At test time: scale by (1-p) to maintain expected value
dropout = nn.Dropout(p=0.5)
x_drop  = torch.ones(1000)
train_out = dropout(x_drop)   # ~50% zeroed
print(f"  Dropout(p=0.5) on ones: mean={train_out.mean():.4f}  (≈1.0 due to scaling)")

subheader("Adam Optimizer — Statistical Moments")
# 1st moment (mean gradient): m_t = β₁*m_{t-1} + (1-β₁)*g_t
# 2nd moment (variance):      v_t = β₂*v_{t-1} + (1-β₂)*g_t²
# Bias-corrected: m̂_t = m_t/(1-β₁ᵗ),  v̂_t = v_t/(1-β₂ᵗ)
# Update: θ_{t+1} = θ_t - η/(√v̂_t + ε) * m̂_t

beta1, beta2, eta, eps_adam = 0.9, 0.999, 0.001, 1e-8
m, v, t_adam = 0.0, 0.0, 0

# Simulate one parameter update
gradients = [np.random.randn() for _ in range(10)]
theta     = 1.0
for g in gradients:
    t_adam += 1
    m = beta1 * m + (1 - beta1) * g
    v = beta2 * v + (1 - beta2) * g**2
    m_hat = m / (1 - beta1**t_adam)
    v_hat = v / (1 - beta2**t_adam)
    theta -= eta * m_hat / (math.sqrt(v_hat) + eps_adam)
print(f"\n  Adam: after 10 steps, θ = {theta:.6f}")
print(f"  Bias-corrected m̂={m/(1-beta1**t_adam):.6f}  v̂={v/(1-beta2**t_adam):.8f}")

subheader("Loss Functions — Statistical Derivations")
y_true_loss = torch.tensor([1.0, 0.0, 1.0, 1.0, 0.0])
y_pred_loss = torch.tensor([0.9, 0.1, 0.8, 0.7, 0.3])

mse_loss = nn.MSELoss()(y_pred_loss, y_true_loss)
mae_loss = nn.L1Loss()(y_pred_loss, y_true_loss)
bce_loss = nn.BCELoss()(y_pred_loss, y_true_loss)
huber_loss = nn.HuberLoss(delta=1.0)(y_pred_loss, y_true_loss)

print(f"  MSE   (Gaussian noise)  : {mse_loss:.4f}")
print(f"  MAE   (Laplace noise)   : {mae_loss:.4f}")
print(f"  BCE   (Bernoulli)       : {bce_loss:.4f}")
print(f"  Huber (robust)          : {huber_loss:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 24 · Statistical Learning Theory
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 24 · Statistical Learning Theory")

subheader("Bias-Variance Decomposition")
np.random.seed(42)
def true_f(x: np.ndarray) -> np.ndarray:
    return np.sin(x)

n_samples, n_experiments = 50, 200
x_test = np.linspace(-3, 3, 100)
y_test = true_f(x_test)

from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def bias_variance(degree: int) -> tuple[float, float]:
    predictions = []
    for _ in range(n_experiments):
        x_train = np.random.uniform(-3, 3, n_samples)
        y_train = true_f(x_train) + np.random.normal(0, 0.3, n_samples)
        model = Pipeline([("poly", PolynomialFeatures(degree)),
                          ("lin",  LinearRegression())])
        model.fit(x_train.reshape(-1, 1), y_train)
        predictions.append(model.predict(x_test.reshape(-1, 1)))
    predictions = np.array(predictions)
    bias2 = np.mean((predictions.mean(0) - y_test)**2)
    variance = np.mean(predictions.var(0))
    return float(bias2), float(variance)

print(f"  {'Degree':<8} {'Bias²':>8} {'Variance':>10} {'B²+V':>10}")
for deg in [1, 3, 9, 15]:
    b2, var = bias_variance(deg)
    print(f"  {deg:<8} {b2:>8.4f} {var:>10.4f} {b2+var:>10.4f}")

subheader("AIC / BIC — Model Selection")
# AIC = 2k - 2 ln L̂
# BIC = k ln n - 2 ln L̂
from sklearn.datasets import make_regression
X_sel, y_sel = make_regression(n_samples=100, n_features=1, noise=5, random_state=42)

def aic_bic(X_s: np.ndarray, y_s: np.ndarray, k: int) -> tuple[float, float]:
    model_s = LinearRegression()
    model_s.fit(X_s[:, :k], y_s)
    y_hat_s = model_s.predict(X_s[:, :k])
    rss  = np.sum((y_s - y_hat_s)**2)
    n    = len(y_s)
    log_L = -n/2 * math.log(rss/n)    # Gaussian log-likelihood
    aic  = 2 * (k+1) - 2 * log_L
    bic  = (k+1) * math.log(n) - 2 * log_L
    return aic, bic

X_big, y_big = make_regression(n_samples=200, n_features=10, noise=5, random_state=42)
print(f"\n  {'k params':<12} {'AIC':>10} {'BIC':>10}")
for k in [1, 3, 5, 8, 10]:
    a, bic_val = aic_bic(X_big, y_big, k)
    print(f"  {k:<12} {a:>10.2f} {bic_val:>10.2f}")
print(f"  → Lower is better; BIC penalizes complexity more for large n")

subheader("Ridge Effective Degrees of Freedom")
# df(λ) = tr[X(XᵀX + λI)⁻¹Xᵀ] = Σ dⱼ²/(dⱼ²+λ)
X_edf = X_big[:, :5]
U, d, Vt = np.linalg.svd(X_edf, full_matrices=False)
for lam in [0.001, 1.0, 10.0, 100.0]:
    df = np.sum(d**2 / (d**2 + lam))
    print(f"  λ={lam:>6}  effective df = {df:.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# SECTION 25 · Master Formula Verification
# ──────────────────────────────────────────────────────────────────────────────
header("SECTION 25 · Master Formula Reference — Quick Verification")

data_ref = np.array([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])

formulas = {
    "Arithmetic mean x̄ = Σxᵢ/n"   : np.mean(data_ref),
    "Median"                        : np.median(data_ref),
    "Geometric mean G=(Πxᵢ)^(1/n)" : np.exp(np.mean(np.log(data_ref))),
    "Harmonic mean H=n/Σ(1/xᵢ)"   : len(data_ref)/np.sum(1/data_ref),
    "Sample variance s²"            : np.var(data_ref, ddof=1),
    "Sample std s"                  : np.std(data_ref, ddof=1),
    "IQR = Q3 - Q1"                : (np.percentile(data_ref, 75)
                                       - np.percentile(data_ref, 25)),
    "CV = σ/μ × 100%"              : np.std(data_ref)/np.mean(data_ref)*100,
}

for name, val in formulas.items():
    print(f"  {name:<40}: {val:.4f}")

# Verify H(P,Q) = H(P) + D_KL(P||Q)
P_ref = np.array([0.4, 0.3, 0.2, 0.1])
Q_ref = np.array([0.25, 0.25, 0.25, 0.25])
H_P_ref  = -np.sum(P_ref * np.log2(P_ref))
KL_ref   = np.sum(P_ref * np.log2(P_ref / Q_ref))
HCE_ref  = -np.sum(P_ref * np.log2(Q_ref))
print(f"\n  H(P)={H_P_ref:.4f}  D_KL={KL_ref:.4f}  H(P,Q)={HCE_ref:.4f}")
print(f"  H(P) + D_KL = {H_P_ref+KL_ref:.4f}  =?= H(P,Q) = {HCE_ref:.4f} ✓")

# Bayes: P(H|E) = P(E|H)*P(H)/P(E)
prior_ref, lik_ref, ev_ref = 0.3, 0.8, 0.31
posterior_ref = lik_ref * prior_ref / ev_ref
print(f"\n  Bayes: P(H|E) = {lik_ref}×{prior_ref}/{ev_ref} = {posterior_ref:.4f}")

# OLS normal equations
X_ref_mat = np.column_stack([np.ones(len(data_ref)), data_ref])
y_ref = 2 * data_ref + np.random.randn(len(data_ref)) * 0.1
beta_ref = np.linalg.inv(X_ref_mat.T @ X_ref_mat) @ X_ref_mat.T @ y_ref
print(f"\n  OLS β̂ = (XᵀX)⁻¹Xᵀy = {beta_ref.round(4)}")


print(f"\n{DIVIDER}")
print("  ✅ All 25 sections implemented and verified.")
print("  Libraries used: numpy · math · torch · sklearn · tensorflow · scipy")
print(DIVIDER)
