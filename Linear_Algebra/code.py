"""
============================================================
  LINEAR ALGEBRA FOR MACHINE LEARNING & DEEP LEARNING
  Complete Python Implementation — NumPy, TensorFlow, PyTorch
============================================================
Topics 1–20 from Linear_Algebra.md
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tensorflow as tf
from scipy import linalg as scipy_linalg
import warnings
warnings.filterwarnings("ignore")

SEP = "=" * 65

def section(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")

def subsection(title):
    print(f"\n--- {title} ---")


# ============================================================
# 1. SCALARS, VECTORS, MATRICES & TENSORS
# ============================================================
section("1. Scalars, Vectors, Matrices & Tensors")

# --- 1.1 Scalar ---
subsection("1.1 Scalar")
alpha = np.float64(0.01)          # learning rate
lam   = np.float64(0.001)         # regularization lambda
pixel = np.float64(255.0)         # pixel value
print(f"Learning rate α = {alpha}")
print(f"Regularization λ = {lam}")
print(f"Pixel value = {pixel}")

# PyTorch scalar
t_scalar = torch.tensor(0.01)
print(f"PyTorch scalar: {t_scalar}, shape: {t_scalar.shape}")

# TF scalar
tf_scalar = tf.constant(0.01)
print(f"TF scalar: {tf_scalar.numpy()}, shape: {tf_scalar.shape}")

# --- 1.2 Vector ---
subsection("1.2 Vector")
x_col = np.array([[175], [70], [25]], dtype=float)   # column vector
x_row = x_col.T                                       # row vector
print(f"Column vector (height, weight, age):\n{x_col}")
print(f"Row vector:\n{x_row}")

t_vec = torch.tensor([175.0, 70.0, 25.0])
tf_vec = tf.constant([175.0, 70.0, 25.0])
print(f"PyTorch vector shape: {t_vec.shape}")
print(f"TensorFlow vector shape: {tf_vec.shape}")

# --- 1.3 Matrix ---
subsection("1.3 Matrix")
X = np.array([
    [175, 70, 25],
    [160, 55, 30],
    [180, 85, 22],
    [165, 60, 28]
], dtype=float)
print(f"Dataset matrix X (4×3):\n{X}")
print(f"Element at row 1, col 2: X[1,2] = {X[1,2]}")

t_mat = torch.tensor(X)
tf_mat = tf.constant(X)
print(f"PyTorch matrix shape: {t_mat.shape}")
print(f"TensorFlow matrix shape: {tf_mat.shape}")

# --- 1.4 Tensor ---
subsection("1.4 Tensor — batch of images")
T = np.random.randn(32, 3, 64, 64)    # batch=32, channels=3, H=64, W=64
print(f"NumPy 4D tensor shape: {T.shape}")

t_tensor = torch.randn(32, 3, 64, 64)
tf_tensor = tf.random.normal([32, 3, 64, 64])
print(f"PyTorch 4D tensor shape: {t_tensor.shape}")
print(f"TensorFlow 4D tensor shape: {tf_tensor.shape}")


# ============================================================
# 2. VECTOR OPERATIONS
# ============================================================
section("2. Vector Operations")

u = np.array([1.0, 2.0, 3.0])
v = np.array([4.0, 5.0, 6.0])

# --- 2.1 Addition & Subtraction ---
subsection("2.1 Addition & Subtraction")
print(f"u + v = {u + v}")
print(f"u - v = {u - v}")
# PyTorch
tu, tv = torch.tensor(u), torch.tensor(v)
print(f"PyTorch u + v = {tu + tv}")

# --- 2.2 Scalar Multiplication ---
subsection("2.2 Scalar Multiplication (Gradient Descent update)")
w = np.array([0.5, -0.3, 0.8])
grad = np.array([0.1, 0.05, -0.2])
lr = 0.01
w_new = w - lr * grad
print(f"w (old) = {w}")
print(f"w (new after GD step) = {w_new}")

# PyTorch autograd demo
tw = torch.tensor(w, requires_grad=True)
loss = (tw ** 2).sum()
loss.backward()
print(f"PyTorch gradient: {tw.grad.numpy()}")

# --- 2.3 Dot Product ---
subsection("2.3 Dot Product")
dot = np.dot(u, v)
print(f"u · v = {dot}")
norm_u, norm_v = np.linalg.norm(u), np.linalg.norm(v)
cos_theta = dot / (norm_u * norm_v)
print(f"cos(θ) = {cos_theta:.4f},  θ = {np.degrees(np.arccos(cos_theta)):.2f}°")

# Linear prediction: ŷ = w^T x
weights = np.array([0.3, 0.5, -0.2])
x_feat  = np.array([1.0, 2.0, 3.0])
y_hat   = np.dot(weights, x_feat)
print(f"Linear prediction ŷ = w·x = {y_hat:.4f}")

print(f"PyTorch dot: {torch.dot(tu, tv)}")
print(f"TF dot: {tf.tensordot(tf.constant(u), tf.constant(v), axes=1).numpy()}")

# --- 2.4 Outer Product ---
subsection("2.4 Outer Product (backprop gradient)")
delta = np.array([0.1, 0.2])     # error signal
x_in  = np.array([1.0, 2.0, 3.0]) # input
dL_dW = np.outer(delta, x_in)   # gradient of weight matrix
print(f"∂L/∂W = δ ⊗ x^T:\n{dL_dW}")
print(f"PyTorch outer: \n{torch.outer(torch.tensor(delta), torch.tensor(x_in))}")

# --- 2.5 Hadamard (Element-wise) Product ---
subsection("2.5 Hadamard Product (Dropout mask)")
h = np.array([0.5, 1.2, -0.8, 0.3])
mask = np.array([1.0, 0.0, 1.0, 1.0])   # dropout mask
h_dropped = h * mask
print(f"h = {h}")
print(f"mask = {mask}")
print(f"h ⊙ mask = {h_dropped}")
print(f"PyTorch Hadamard: {torch.tensor(h) * torch.tensor(mask)}")

# --- 2.6 Cross Product (3D) ---
subsection("2.6 Cross Product")
u3 = np.array([1.0, 0.0, 0.0])
v3 = np.array([0.0, 1.0, 0.0])
cross = np.cross(u3, v3)
print(f"u × v = {cross}  (z-axis, perpendicular to both)")


# ============================================================
# 3. MATRIX OPERATIONS
# ============================================================
section("3. Matrix Operations")

A = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
B = np.array([[9,8,7],[6,5,4],[3,2,1]], dtype=float)

# --- 3.1–3.2 Addition & Scalar Multiplication ---
subsection("3.1–3.2 Matrix Addition & Scalar Multiply")
print(f"A + B =\n{A + B}")
print(f"3 * A =\n{3 * A}")

# --- 3.3 Matrix-Vector Multiply ---
subsection("3.3 Matrix-Vector Multiply (Linear layer: z = Wx + b)")
W = np.random.randn(4, 3)
x = np.array([1.0, 2.0, 3.0])
b = np.zeros(4)
z = W @ x + b
print(f"z = Wx + b, shape: {z.shape}, values: {np.round(z, 3)}")

# PyTorch Linear layer
linear = nn.Linear(3, 4, bias=True)
xt = torch.tensor(x, dtype=torch.float32)
zt = linear(xt)
print(f"PyTorch Linear layer output shape: {zt.shape}")

# TF Dense layer
dense = tf.keras.layers.Dense(4)
xf = tf.constant([[1.0, 2.0, 3.0]])
zf = dense(xf)
print(f"TF Dense layer output shape: {zf.shape}")

# --- 3.4 Matrix-Matrix Multiply ---
subsection("3.4 Matrix-Matrix Multiply")
C = np.random.randn(3, 4)
D = np.random.randn(4, 5)
E = C @ D
print(f"C ({C.shape}) @ D ({D.shape}) = E {E.shape}")
print(f"PyTorch: {(torch.tensor(C) @ torch.tensor(D)).shape}")
print(f"TensorFlow: {tf.matmul(tf.constant(C), tf.constant(D)).shape}")

# Multi-layer forward pass: y = W2 * σ(W1*x + b1) + b2
W1, W2 = np.random.randn(4, 3), np.random.randn(2, 4)
b1, b2 = np.zeros(4), np.zeros(2)
x_in = np.array([1.0, 2.0, 3.0])
h1 = np.maximum(0, W1 @ x_in + b1)  # ReLU
y_out = W2 @ h1 + b2
print(f"2-layer network output: {np.round(y_out, 3)}")

# --- 3.5 Transpose ---
subsection("3.5 Transpose")
M = np.array([[1,2,3],[4,5,6]])
print(f"M:\n{M}")
print(f"M^T:\n{M.T}")
print(f"(AB)^T == B^T A^T: {np.allclose((A@B).T, B.T @ A.T)}")

# --- 3.6 Matrix Inverse ---
subsection("3.6 Matrix Inverse (Normal Equations)")
A_sq = np.array([[4,1],[2,3]], dtype=float)
A_inv = np.linalg.inv(A_sq)
print(f"A:\n{A_sq}")
print(f"A^-1:\n{np.round(A_inv, 3)}")
print(f"A @ A^-1 = I: {np.allclose(A_sq @ A_inv, np.eye(2))}")

# Normal equations: θ̂ = (X^T X)^-1 X^T y
X_data = np.column_stack([np.ones(5), np.array([1,2,3,4,5], dtype=float)])
y_data = np.array([2.1, 3.9, 6.2, 7.8, 10.1])
theta_hat = np.linalg.inv(X_data.T @ X_data) @ X_data.T @ y_data
print(f"Normal equations θ̂ (intercept, slope): {np.round(theta_hat, 3)}")

# --- 3.7 Trace ---
subsection("3.7 Trace")
M_sq = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
print(f"tr(A) = {np.trace(M_sq)}")
print(f"tr(A^T A) = Frobenius^2 = {np.trace(M_sq.T @ M_sq):.2f}")
print(f"||A||_F^2 = {np.linalg.norm(M_sq, 'fro')**2:.2f}")


# ============================================================
# 4. SPECIAL MATRICES
# ============================================================
section("4. Special Matrices")

subsection("4.1 Identity Matrix")
I4 = np.eye(4)
print(f"I_4:\n{I4}")

subsection("4.2 Zero Matrix")
Z = np.zeros((3, 3))
print(f"Zero matrix:\n{Z}")

subsection("4.3 Diagonal Matrix")
D_diag = np.diag([3.0, 1.0, 2.0])
print(f"Diagonal matrix:\n{D_diag}")
print(f"D^-1:\n{np.diag([1/3, 1.0, 0.5])}")

subsection("4.4 Symmetric Matrix")
Sym = np.array([[4,2,1],[2,5,3],[1,3,6]], dtype=float)
print(f"Symmetric: {np.allclose(Sym, Sym.T)}")
eigs = np.linalg.eigvalsh(Sym)
print(f"Eigenvalues (all real): {np.round(eigs, 3)}")

subsection("4.5 Orthogonal Matrix")
theta = np.pi / 4
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])
print(f"Rotation matrix Q (45°):\n{np.round(Q, 3)}")
print(f"Q^T Q = I: {np.allclose(Q.T @ Q, np.eye(2))}")
print(f"det(Q) = {np.linalg.det(Q):.2f}")

subsection("4.6 Triangular Matrices (LU)")
A_lu = np.array([[2,1,1],[4,3,3],[8,7,9]], dtype=float)
P, L, U = scipy_linalg.lu(A_lu)
print(f"L:\n{np.round(L, 3)}")
print(f"U:\n{np.round(U, 3)}")
print(f"LU reconstruction matches A: {np.allclose(P @ L @ U, A_lu)}")

subsection("4.7 Positive Semidefinite Matrix")
X_psd = np.random.randn(5, 3)
Cov = X_psd.T @ X_psd        # always PSD
eigs_psd = np.linalg.eigvalsh(Cov)
print(f"Covariance matrix eigenvalues (all ≥ 0): {np.round(eigs_psd, 3)}")


# ============================================================
# 5. SYSTEMS OF LINEAR EQUATIONS
# ============================================================
section("5. Systems of Linear Equations")

subsection("5.1 Exact Solution: Ax = b")
A5 = np.array([[2,1],[-1,3]], dtype=float)
b5 = np.array([5, 7], dtype=float)
x5 = np.linalg.solve(A5, b5)
print(f"Solution x: {x5}")
print(f"Verify Ax = b: {np.allclose(A5 @ x5, b5)}")

subsection("5.2 Gaussian Elimination (row echelon)")
Aug = np.column_stack([A5, b5])
print(f"Augmented matrix [A|b]:\n{Aug}")
# scipy solve via LU
print(f"scipy linalg.solve: {scipy_linalg.solve(A5, b5)}")

subsection("5.3 Least Squares (overdetermined)")
A_over = np.array([[1,1],[1,2],[1,3],[1,4]], dtype=float)
b_over = np.array([2, 3, 5, 4], dtype=float)
x_ls, _, _, _ = np.linalg.lstsq(A_over, b_over, rcond=None)
print(f"Least squares solution: {np.round(x_ls, 3)}")
print(f"Normal equations solution: "
      f"{np.round(np.linalg.inv(A_over.T @ A_over) @ A_over.T @ b_over, 3)}")


# ============================================================
# 6. VECTOR SPACES & SUBSPACES
# ============================================================
section("6. Vector Spaces & Subspaces")

A6 = np.array([[1,2,3],[4,5,6],[7,8,9]], dtype=float)
rank6 = np.linalg.matrix_rank(A6)
n6 = A6.shape[1]
print(f"rank(A) = {rank6}")
print(f"nullity(A) = n - rank = {n6 - rank6}")

subsection("Four Fundamental Subspaces")
U6, S6, Vt6 = np.linalg.svd(A6)
r6 = np.sum(S6 > 1e-10)
print(f"Column space dim = {r6} (= rank)")
print(f"Null space dim = {n6 - r6}")
print(f"Row space = Right singular vectors (Vt rows 0:{r6})")

subsection("6.4 Span (linear combination)")
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
# Span{v1, v2} covers xy-plane: c1*v1 + c2*v2
c1, c2 = 3.0, -2.0
span_vec = c1 * v1 + c2 * v2
print(f"{c1}*v1 + {c2}*v2 = {span_vec}")


# ============================================================
# 7. LINEAR INDEPENDENCE, BASIS & RANK
# ============================================================
section("7. Linear Independence, Basis & Rank")

subsection("7.1 Linear Independence check")
V_indep = np.array([[1,0],[0,1]], dtype=float)
V_dep   = np.array([[1,2],[2,4]], dtype=float)
print(f"rank([[1,0],[0,1]]) = {np.linalg.matrix_rank(V_indep)}  → independent")
print(f"rank([[1,2],[2,4]]) = {np.linalg.matrix_rank(V_dep)}  → dependent")

subsection("7.2 Standard Basis")
e1, e2, e3 = np.eye(3)
print(f"e1={e1}, e2={e2}, e3={e3}")

subsection("7.2 Change of Basis")
B_basis = np.array([[1, 1],[0, 1]], dtype=float)   # new basis
x_std  = np.array([3.0, 2.0])
x_new_basis = np.linalg.inv(B_basis) @ x_std
print(f"x in standard basis: {x_std}")
print(f"x in new basis: {x_new_basis}")

subsection("7.3 Rank-Nullity Theorem")
M7 = np.random.randn(4, 6)
r7 = np.linalg.matrix_rank(M7)
print(f"rank(M) = {r7},  nullity(M) = {M7.shape[1] - r7},  n = {M7.shape[1]}")
print(f"rank + nullity = {r7 + (M7.shape[1] - r7)} = n = {M7.shape[1]}  ✓")


# ============================================================
# 8. NORMS & DISTANCE METRICS
# ============================================================
section("8. Norms & Distance Metrics")

x_n = np.array([3.0, -4.0, 1.0, -2.0])

subsection("8.1 Vector Norms")
print(f"L1 norm  = {np.linalg.norm(x_n, 1):.4f}")
print(f"L2 norm  = {np.linalg.norm(x_n, 2):.4f}")
print(f"Linf norm = {np.linalg.norm(x_n, np.inf):.4f}")
print(f"L0 pseudo-norm (nnz) = {np.count_nonzero(x_n)}")

# PyTorch norms
xt_n = torch.tensor(x_n)
print(f"PyTorch L1: {torch.norm(xt_n, p=1):.4f}")
print(f"PyTorch L2: {torch.norm(xt_n, p=2):.4f}")

# TF norms
print(f"TF L2: {tf.norm(tf.constant(x_n)).numpy():.4f}")

subsection("8.2 Matrix Norms")
M8 = np.random.randn(4, 3)
print(f"Frobenius norm = {np.linalg.norm(M8, 'fro'):.4f}")
print(f"Spectral norm (σ_max) = {np.linalg.norm(M8, 2):.4f}")
nuclear = np.sum(np.linalg.svd(M8, compute_uv=False))
print(f"Nuclear norm (Σσ_i) = {nuclear:.4f}")

subsection("8.3 Distance Metrics")
u8 = np.array([1.0, 2.0, 3.0])
v8 = np.array([4.0, 5.0, 6.0])

# Euclidean
eucl = np.linalg.norm(u8 - v8)
print(f"Euclidean distance = {eucl:.4f}")

# Cosine similarity
cos_sim = np.dot(u8, v8) / (np.linalg.norm(u8) * np.linalg.norm(v8))
print(f"Cosine similarity = {cos_sim:.4f},  cosine distance = {1 - cos_sim:.4f}")

# Mahalanobis
Sigma_cov = np.array([[2, 1],[1, 3]], dtype=float)
u_m, v_m = np.array([1.0, 2.0]), np.array([3.0, 4.0])
diff_m = u_m - v_m
mahal = np.sqrt(diff_m @ np.linalg.inv(Sigma_cov) @ diff_m)
print(f"Mahalanobis distance = {mahal:.4f}")

subsection("8.4 Cauchy–Schwarz Inequality")
lhs = abs(np.dot(u8, v8))
rhs = np.linalg.norm(u8) * np.linalg.norm(v8)
print(f"|u·v| = {lhs:.4f} ≤ ||u||·||v|| = {rhs:.4f}  ✓" if lhs <= rhs else "Violated!")


# ============================================================
# 9. PROJECTIONS
# ============================================================
section("9. Projections")

subsection("9.1 Projection onto a vector")
a = np.array([2.0, 1.0])
b = np.array([3.0, 4.0])
proj_scalar = (a @ b) / (a @ a)
proj_vec = proj_scalar * a
P_matrix = np.outer(a, a) / (a @ a)   # projection matrix
print(f"proj_a(b) = {proj_vec}")
print(f"Projection matrix P:\n{np.round(P_matrix, 3)}")
print(f"P is idempotent (P²=P): {np.allclose(P_matrix @ P_matrix, P_matrix)}")

subsection("9.2 Projection onto column space (Hat matrix)")
X_hat = np.array([[1,1],[1,2],[1,3],[1,4]], dtype=float)
P_hat = X_hat @ np.linalg.inv(X_hat.T @ X_hat) @ X_hat.T
print(f"Hat matrix shape: {P_hat.shape}")
print(f"P^2 = P (idempotent): {np.allclose(P_hat @ P_hat, P_hat)}")
print(f"P^T = P (symmetric): {np.allclose(P_hat.T, P_hat)}")

y_proj = np.array([2.1, 3.9, 6.2, 7.8])
y_hat_proj = P_hat @ y_proj
residual = y_proj - y_hat_proj
print(f"Residual ⊥ column space: {np.allclose(X_hat.T @ residual, 0, atol=1e-10)}")


# ============================================================
# 10. DETERMINANTS
# ============================================================
section("10. Determinants")

A10 = np.array([[3,1,2],[1,4,3],[2,3,5]], dtype=float)
det_A = np.linalg.det(A10)
print(f"det(A) = {det_A:.4f}")
print(f"det(I) = {np.linalg.det(np.eye(4)):.1f}")

B10 = np.array([[1,2],[3,4]], dtype=float)
print(f"det(A)·det(B) = det(AB): "
      f"{np.isclose(np.linalg.det(A10[:2,:2]) * np.linalg.det(B10), np.linalg.det(A10[:2,:2] @ B10))}")
print(f"det(A^T) = det(A): {np.isclose(np.linalg.det(A10), np.linalg.det(A10.T))}")

eigs_det = np.linalg.eigvals(A10)
print(f"Product of eigenvalues = det(A): "
      f"{np.isclose(np.prod(eigs_det).real, det_A, atol=1e-6)}")

subsection("Normalizing Flow: log|det Jacobian|")
J = np.random.randn(3, 3)
log_det_J = np.log(abs(np.linalg.det(J)))
print(f"log|det(J)| = {log_det_J:.4f}")


# ============================================================
# 11. EIGENVALUES & EIGENVECTORS
# ============================================================
section("11. Eigenvalues & Eigenvectors")

A11 = np.array([[4, 2], [1, 3]], dtype=float)
eigenvalues, eigenvectors = np.linalg.eig(A11)
print(f"Eigenvalues: {eigenvalues}")
print(f"Eigenvectors (columns):\n{np.round(eigenvectors, 3)}")

# Verify Av = λv
for i, (lam, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
    diff = np.allclose(A11 @ vec, lam * vec)
    print(f"λ{i+1}={lam:.2f}: Av = λv? {diff}")

subsection("11.4 Trace = Σλ,  det = Πλ")
print(f"tr(A) = {np.trace(A11):.2f}, Σλ = {sum(eigenvalues).real:.2f}")
print(f"det(A) = {np.linalg.det(A11):.2f}, Πλ = {np.prod(eigenvalues).real:.2f}")

subsection("11.5 Eigendecomposition A = VΛV⁻¹")
V11 = eigenvectors
Lambda11 = np.diag(eigenvalues.real)
A_recon = V11 @ Lambda11 @ np.linalg.inv(V11)
print(f"A reconstructed == A: {np.allclose(A_recon, A11)}")

# Symmetric: A = QΛQ^T
Sym11 = np.array([[3,1],[1,3]], dtype=float)
lam_s, Q_s = np.linalg.eigh(Sym11)
A_s_recon = Q_s @ np.diag(lam_s) @ Q_s.T
print(f"Symmetric eigen-recon: {np.allclose(A_s_recon, Sym11)}")

subsection("11.5 Matrix Power & Exponential")
A_pow3 = V11 @ np.diag(eigenvalues.real**3) @ np.linalg.inv(V11)
print(f"A³ via eigendecomp:\n{np.round(A_pow3, 2)}")
print(f"A³ direct:\n{np.round(np.linalg.matrix_power(A11.astype(int), 3).astype(float), 2)}")

subsection("11.6 Rayleigh Quotient — PCA direction")
Cov11 = np.array([[3,2],[2,2]], dtype=float)
lam_max = np.linalg.eigvalsh(Cov11).max()
print(f"Max Rayleigh quotient = λ_max = {lam_max:.4f}")


# ============================================================
# 12. MATRIX DECOMPOSITIONS
# ============================================================
section("12. Matrix Decompositions")

A12 = np.array([[2,1,1],[4,3,3],[8,7,9]], dtype=float)

subsection("12.1 LU Decomposition")
P12, L12, U12 = scipy_linalg.lu(A12)
print(f"PA = LU reconstruction: {np.allclose(P12 @ A12, L12 @ U12)}")
x_lu = scipy_linalg.solve(A12, np.array([1.0, 2.0, 3.0]))
print(f"Solve Ax=b via LU: {np.round(x_lu, 3)}")

subsection("12.2 Cholesky Decomposition")
A_pd = np.array([[4,2,1],[2,5,3],[1,3,6]], dtype=float)
L_chol = np.linalg.cholesky(A_pd)
print(f"L:\n{np.round(L_chol, 3)}")
print(f"LL^T = A: {np.allclose(L_chol @ L_chol.T, A_pd)}")

# Multivariate Gaussian sampling: x = Lz + μ
mu = np.array([1.0, 2.0, 3.0])
Sigma_gauss = np.array([[4,2,1],[2,5,3],[1,3,6]], dtype=float)
L_gauss = np.linalg.cholesky(Sigma_gauss)
z_sample = np.random.randn(3)
x_gauss = L_gauss @ z_sample + mu
print(f"Multivariate Gaussian sample: {np.round(x_gauss, 3)}")

subsection("12.3 QR Decomposition")
A_qr = np.random.randn(5, 3)
Q_qr, R_qr = np.linalg.qr(A_qr)
print(f"Q shape: {Q_qr.shape}, R shape: {R_qr.shape}")
print(f"QR = A: {np.allclose(Q_qr @ R_qr, A_qr)}")
print(f"Q^T Q = I: {np.allclose(Q_qr.T @ Q_qr, np.eye(3))}")


# ============================================================
# 13. SINGULAR VALUE DECOMPOSITION (SVD)
# ============================================================
section("13. Singular Value Decomposition")

A13 = np.array([[3,2,2],[2,3,-2]], dtype=float)
U13, S13, Vt13 = np.linalg.svd(A13, full_matrices=True)
print(f"U shape: {U13.shape}, S: {np.round(S13, 3)}, Vt shape: {Vt13.shape}")

# Reconstruct A from full SVD
S_full = np.zeros(A13.shape)
for i, s in enumerate(S13):
    S_full[i, i] = s
print(f"U Σ V^T = A: {np.allclose(U13 @ S_full @ Vt13, A13)}")

subsection("13.3 Rank-k Approximation")
np.random.seed(42)
A_img = np.random.randn(50, 50)
U_i, S_i, Vt_i = np.linalg.svd(A_img)
for k in [1, 5, 10, 20]:
    A_k = U_i[:, :k] @ np.diag(S_i[:k]) @ Vt_i[:k, :]
    err_frob = np.linalg.norm(A_img - A_k, 'fro')
    print(f"  k={k:2d}: Frobenius error = {err_frob:.4f}")

subsection("13.5 Moore-Penrose Pseudoinverse")
A_rect = np.array([[1,2],[3,4],[5,6]], dtype=float)
A_pinv = np.linalg.pinv(A_rect)    # via SVD internally
print(f"A⁺ shape: {A_pinv.shape}")
b_rect = np.array([1.0, 2.0, 3.0])
x_pinv = A_pinv @ b_rect
print(f"Least-squares solution x = A⁺b: {np.round(x_pinv, 3)}")

subsection("13 — PyTorch & TF SVD")
A13t = torch.tensor(A13)
U_t, S_t, Vh_t = torch.linalg.svd(A13t)
print(f"PyTorch SVD — S: {S_t.numpy().round(3)}")

A13f = tf.constant(A13)
s_tf, u_tf, v_tf = tf.linalg.svd(A13f)
print(f"TensorFlow SVD — S: {s_tf.numpy().round(3)}")


# ============================================================
# 14. PRINCIPAL COMPONENT ANALYSIS (PCA)
# ============================================================
section("14. PCA")

np.random.seed(0)
n_samples, n_features = 100, 5
X14 = np.random.randn(n_samples, n_features)
X14[:, 1] = 2 * X14[:, 0] + 0.3 * np.random.randn(n_samples)  # correlated

subsection("14.2 PCA from scratch")
# Step 1: Center
X_c = X14 - X14.mean(axis=0)

# Step 2: Covariance matrix
Cov14 = (X_c.T @ X_c) / (n_samples - 1)
print(f"Covariance matrix shape: {Cov14.shape}")

# Step 3: Eigen-decompose
lam14, Q14 = np.linalg.eigh(Cov14)
idx = np.argsort(lam14)[::-1]
lam14, Q14 = lam14[idx], Q14[:, idx]

# Step 4: Project to k=2
k = 2
Z14 = X_c @ Q14[:, :k]
print(f"Projected data shape: {Z14.shape}")

subsection("14.3 Variance Explained")
var_ratio = lam14 / lam14.sum()
cum_var   = np.cumsum(var_ratio)
for i, (v, c) in enumerate(zip(var_ratio, cum_var)):
    print(f"  PC{i+1}: {v*100:.1f}%  (cumulative {c*100:.1f}%)")

subsection("14.4 PCA via SVD")
U14, S14, Vt14 = np.linalg.svd(X_c, full_matrices=False)
Z14_svd = X_c @ Vt14[:k, :].T
print(f"PCA via SVD — scores shape: {Z14_svd.shape}")

subsection("14.5 Reconstruction")
X_recon = Z14 @ Q14[:, :k].T + X14.mean(axis=0)
err_recon = np.linalg.norm(X14 - X_recon, 'fro')
print(f"Reconstruction error (k=2): {err_recon:.4f}")

# PyTorch PCA
X14t = torch.tensor(X_c, dtype=torch.float32)
U_pca, S_pca, V_pca = torch.pca_lowrank(X14t, q=k)
Z_pca_t = X14t @ V_pca
print(f"PyTorch pca_lowrank output shape: {Z_pca_t.shape}")


# ============================================================
# 15. POSITIVE DEFINITE MATRICES
# ============================================================
section("15. Positive Definite Matrices")

A_pd15 = np.array([[4,2,1],[2,5,3],[1,3,6]], dtype=float)

subsection("15.1 Tests for Positive Definiteness")
eigs_pd = np.linalg.eigvalsh(A_pd15)
print(f"Eigenvalues (all > 0 → PD): {np.round(eigs_pd, 3)}")

try:
    L_test = np.linalg.cholesky(A_pd15)
    print(f"Cholesky exists → PD confirmed")
except:
    print("Not positive definite")

# Sylvester's criterion: all leading minors > 0
minors = [np.linalg.det(A_pd15[:k, :k]) for k in range(1, 4)]
print(f"Leading principal minors: {[round(m, 3) for m in minors]}  (all > 0 → PD)")

subsection("15.2 Quadratic Forms")
x15 = np.array([1.0, 2.0, 3.0])
quad_form = x15 @ A_pd15 @ x15
print(f"x^T A x = {quad_form:.2f} > 0  (PD confirms bowl shape)")

# Hessian of MSE loss (always PSD)
X_hess = np.random.randn(20, 3)
H_mse = (1/20) * X_hess.T @ X_hess    # ∇²L_MSE = (1/n) X^T X
eigs_H = np.linalg.eigvalsh(H_mse)
print(f"MSE Hessian eigenvalues (all ≥ 0 → convex): {np.round(eigs_H, 3)}")


# ============================================================
# 16. MATRIX CALCULUS
# ============================================================
section("16. Matrix Calculus")

subsection("16.1–16.3 Gradient, Jacobian, Hessian via PyTorch Autograd")
# Gradient of f(x) = x^T A x
A16 = torch.tensor([[3.0,1.0],[1.0,2.0]], requires_grad=False)
x16 = torch.tensor([1.0, 2.0], requires_grad=True)
f = x16 @ A16 @ x16
f.backward()
print(f"∂(x^T A x)/∂x = 2Ax = {(2 * A16 @ x16).detach().numpy()}")
print(f"PyTorch autograd: {x16.grad.numpy()}")

# Jacobian of vector function f: R² → R²
x_j = torch.tensor([1.0, 2.0], requires_grad=True)
def vec_func(x):
    return torch.stack([x[0]**2 + x[1], x[0] * x[1]**2])
J_pt = torch.autograd.functional.jacobian(vec_func, x_j)
print(f"Jacobian:\n{J_pt.numpy()}")

# Hessian of scalar function
x_h = torch.tensor([1.0, 2.0], requires_grad=True)
def scalar_func(x):
    return x[0]**2 + 3*x[0]*x[1] + x[1]**2
H_pt = torch.autograd.functional.hessian(scalar_func, x_h)
print(f"Hessian:\n{H_pt.numpy()}")

subsection("16.6 MSE Gradient")
X_mse = np.random.randn(10, 3)
y_mse = np.random.randn(10)
theta = np.zeros(3)
grad_mse = (1/10) * X_mse.T @ (X_mse @ theta - y_mse)
print(f"∇L_MSE = {np.round(grad_mse, 3)}")

# TensorFlow GradientTape
X_tf = tf.constant(X_mse, dtype=tf.float32)
y_tf = tf.constant(y_mse, dtype=tf.float32)
theta_tf = tf.Variable(np.zeros(3), dtype=tf.float32)
with tf.GradientTape() as tape:
    loss_tf = 0.5 * tf.reduce_mean((X_tf @ theta_tf - y_tf)**2)
grad_tf = tape.gradient(loss_tf, theta_tf)
print(f"TF GradientTape ∇L_MSE: {np.round(grad_tf.numpy(), 3)}")

subsection("16.6 Cross-entropy gradient ∂L/∂z = ŷ - y")
z_ce = np.array([1.0, 2.0, 0.5])
y_ce = np.array([0, 1, 0])
def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()
y_hat_ce = softmax(z_ce)
print(f"Softmax CE gradient ŷ - y = {np.round(y_hat_ce - y_ce, 3)}")


# ============================================================
# 17. TENSORS FOR DEEP LEARNING
# ============================================================
section("17. Tensors for Deep Learning")

subsection("17.1 Tensor creation & indexing")
T_rank3 = torch.randn(4, 3, 5)    # (batch, seq, features)
print(f"Rank-3 tensor shape: {T_rank3.shape}")
print(f"Fiber (row 0, col 0): {T_rank3[0, 0, :]}")
print(f"Slice (batch 0): {T_rank3[0].shape}")

subsection("17.2 Tensor Contraction (einsum)")
A17 = torch.randn(2, 3, 4)
B17 = torch.randn(2, 4, 5)
C17 = torch.einsum('bik,bkj->bij', A17, B17)    # batch matmul
print(f"Batch matmul via einsum: {C17.shape}")

subsection("17.3 Vectorization & Reshape")
M17 = torch.arange(6, dtype=torch.float32).reshape(2, 3)
print(f"Matrix:\n{M17}")
print(f"vec(M) = {M17.flatten()}")
print(f"reshape to (3,2):\n{M17.reshape(3,2)}")

subsection("17.4 Kronecker Product")
A_kron = np.array([[1,0],[0,1]], dtype=float)
B_kron = np.array([[1,2],[3,4]], dtype=float)
Kron = np.kron(A_kron, B_kron)
print(f"A ⊗ B:\n{Kron}")
print(f"Shape: {Kron.shape}")

# PyTorch kronecker
print(f"PyTorch kron shape: {torch.kron(torch.tensor(A_kron), torch.tensor(B_kron)).shape}")

subsection("17.5 Batch Matrix Multiply (BMM)")
A_bmm = torch.randn(8, 4, 3)   # batch=8, 4×3 matrices
B_bmm = torch.randn(8, 3, 5)   # batch=8, 3×5 matrices
C_bmm = torch.bmm(A_bmm, B_bmm)
print(f"BMM result shape: {C_bmm.shape}  (batch=8, 4×5 each)")

# TF batch matmul
A_bmm_tf = tf.random.normal([8, 4, 3])
B_bmm_tf = tf.random.normal([8, 3, 5])
C_bmm_tf = tf.matmul(A_bmm_tf, B_bmm_tf)
print(f"TF batch matmul shape: {C_bmm_tf.shape}")


# ============================================================
# 18. LINEAR TRANSFORMATIONS
# ============================================================
section("18. Linear Transformations")

subsection("18.2 Fundamental Transformations")
theta18 = np.pi / 3   # 60 degrees

# Scaling
S18 = np.diag([2.0, 3.0])
# Rotation
R18 = np.array([[np.cos(theta18), -np.sin(theta18)],
                [np.sin(theta18),  np.cos(theta18)]])
# Reflection (about x-axis)
F18 = np.array([[1, 0],[0, -1]], dtype=float)
# Shear
H18 = np.array([[1, 0.5],[0, 1]], dtype=float)
# Projection onto x-axis
P18 = np.array([[1, 0],[0, 0]], dtype=float)

x18 = np.array([1.0, 1.0])
print(f"Original x: {x18}")
print(f"Scaled (2x, 3y): {S18 @ x18}")
print(f"Rotated 60°: {np.round(R18 @ x18, 3)}")
print(f"Reflected (x-axis): {F18 @ x18}")
print(f"Sheared: {H18 @ x18}")
print(f"Projected to x-axis: {P18 @ x18}")

subsection("18.3 Composition of Transformations")
T_composed = R18 @ S18      # scale then rotate
print(f"Composed (scale then rotate) x: {np.round(T_composed @ x18, 3)}")

subsection("18.4 Affine Transformation (Neural Layer)")
A18 = np.array([[2.0, 0.5],[0.3, 1.5]])
b18 = np.array([1.0, -1.0])
x18_affine = np.array([1.0, 2.0])
print(f"Affine T(x) = Ax + b = {A18 @ x18_affine + b18}")

# Homogeneous coordinates
A18_h = np.block([[A18, b18.reshape(-1,1)], [np.zeros((1,2)), [[1]]]])
x18_h = np.array([1.0, 2.0, 1.0])
print(f"Homogeneous result: {A18_h @ x18_h}")


# ============================================================
# 19. DIMENSIONALITY REDUCTION
# ============================================================
section("19. Dimensionality Reduction")

np.random.seed(1)
n19, d19 = 200, 10
X19 = np.random.randn(n19, d19)
# Introduce class structure
y19 = (X19[:, 0] + X19[:, 1] > 0).astype(int)

subsection("19.2 LDA (Linear Discriminant Analysis)")
classes = np.unique(y19)
mu_overall = X19.mean(axis=0)
S_B = np.zeros((d19, d19))   # between-class scatter
S_W = np.zeros((d19, d19))   # within-class scatter
for c in classes:
    X_c_lda = X19[y19 == c]
    mu_c = X_c_lda.mean(axis=0)
    n_c = len(X_c_lda)
    S_B += n_c * np.outer(mu_c - mu_overall, mu_c - mu_overall)
    diff = X_c_lda - mu_c
    S_W += diff.T @ diff
# Solve generalized eigenvalue problem
lam_lda, W_lda = scipy_linalg.eigh(S_B, S_W)
idx_lda = np.argsort(lam_lda)[::-1]
W_lda = W_lda[:, idx_lda[:1]]    # top-1 LDA direction
Z_lda = X19 @ W_lda
print(f"LDA projected data shape: {Z_lda.shape}")

subsection("19.3 Johnson-Lindenstrauss Random Projection")
k_jl = 20      # reduced dimension
R_jl = np.random.randn(d19, k_jl) / np.sqrt(k_jl)   # JL projection
Z_jl = X19 @ R_jl
print(f"Random projection {d19}D → {k_jl}D: {Z_jl.shape}")

# Check distance preservation
i, j = 0, 5
orig_dist = np.linalg.norm(X19[i] - X19[j])
proj_dist = np.linalg.norm(Z_jl[i] - Z_jl[j])
print(f"Original dist: {orig_dist:.4f}, Projected dist: {proj_dist:.4f}")


# ============================================================
# 20. APPLICATIONS IN ML/DL
# ============================================================
section("20. Applications in ML/DL")

# --- 20.1 Linear Regression ---
subsection("20.1 Linear Regression (closed-form + gradient descent)")
np.random.seed(42)
n20 = 50
X_lr = np.column_stack([np.ones(n20), np.random.randn(n20, 2)])
theta_true = np.array([1.0, 2.0, -1.5])
y_lr = X_lr @ theta_true + 0.2 * np.random.randn(n20)

# Closed-form normal equations
theta_cf = np.linalg.inv(X_lr.T @ X_lr) @ X_lr.T @ y_lr
print(f"True θ: {theta_true}")
print(f"Closed-form θ̂: {np.round(theta_cf, 3)}")
print(f"Least squares via lstsq: "
      f"{np.round(np.linalg.lstsq(X_lr, y_lr, rcond=None)[0], 3)}")

# PyTorch gradient descent
X_lr_t = torch.tensor(X_lr, dtype=torch.float32)
y_lr_t = torch.tensor(y_lr, dtype=torch.float32)
theta_gd = torch.zeros(3, requires_grad=True)
lr_gd = 0.05
for step in range(500):
    y_pred = X_lr_t @ theta_gd
    loss_gd = 0.5 * ((y_pred - y_lr_t)**2).mean()
    loss_gd.backward()
    with torch.no_grad():
        theta_gd -= lr_gd * theta_gd.grad
    theta_gd.grad.zero_()
print(f"GD θ̂ (500 steps): {np.round(theta_gd.detach().numpy(), 3)}")

# --- 20.2 Neural Network Forward Pass ---
subsection("20.2 Neural Network Forward Pass (NumPy & PyTorch)")
def relu(x): return np.maximum(0, x)

W1_nn = np.random.randn(8, 3) * 0.1
b1_nn = np.zeros(8)
W2_nn = np.random.randn(4, 8) * 0.1
b2_nn = np.zeros(4)
W3_nn = np.random.randn(2, 4) * 0.1
b3_nn = np.zeros(2)

x_nn = np.array([1.0, 2.0, 3.0])
h1_nn = relu(W1_nn @ x_nn + b1_nn)
h2_nn = relu(W2_nn @ h1_nn + b2_nn)
y_nn  = W3_nn @ h2_nn + b3_nn
print(f"NumPy 3-layer forward pass output: {np.round(y_nn, 3)}")

# PyTorch MLP
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 8), nn.ReLU(),
            nn.Linear(8, 4), nn.ReLU(),
            nn.Linear(4, 2)
        )
    def forward(self, x):
        return self.net(x)

mlp = MLP()
x_mlp = torch.tensor([[1.0, 2.0, 3.0]])
print(f"PyTorch MLP output: {mlp(x_mlp).detach().numpy().round(3)}")

# TF Keras model
model_tf = tf.keras.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_shape=(3,)),
    tf.keras.layers.Dense(4, activation='relu'),
    tf.keras.layers.Dense(2)
])
x_tf_nn = tf.constant([[1.0, 2.0, 3.0]])
print(f"TF Keras output: {model_tf(x_tf_nn).numpy().round(3)}")

# --- 20.3 Backpropagation ---
subsection("20.3 Backpropagation via Matrix Calculus")
W_bp = torch.randn(4, 3, requires_grad=False)
x_bp = torch.randn(3)
b_bp = torch.zeros(4)

W_bp_req = W_bp.clone().requires_grad_(True)
z_bp = W_bp_req @ x_bp + b_bp
loss_bp = z_bp.sum()
loss_bp.backward()

# Manual: ∂L/∂W = δ x^T (outer product), ∂L/∂x = W^T δ
delta_bp = torch.ones(4)    # ∂L/∂z = 1 (sum loss)
dL_dW_manual = torch.outer(delta_bp, x_bp)
dL_dx_manual = W_bp_req.T @ delta_bp

print(f"∂L/∂W autograd == outer product: {torch.allclose(W_bp_req.grad, dL_dW_manual)}")
print(f"∂L/∂b = δ = {delta_bp.numpy()}")

# --- 20.4 Convolutional Layer as Matrix Multiply ---
subsection("20.4 Convolutional Layer")
conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1)
x_conv = torch.randn(1, 1, 8, 8)    # batch=1, channels=1, 8×8
y_conv = conv(x_conv)
print(f"Conv2d input: {x_conv.shape}, output: {y_conv.shape}")

tf_conv = tf.keras.layers.Conv2D(4, kernel_size=3, padding='same')
x_conv_tf = tf.random.normal([1, 8, 8, 1])
y_conv_tf = tf_conv(x_conv_tf)
print(f"TF Conv2D output: {y_conv_tf.shape}")

# --- 20.5 Attention Mechanism ---
subsection("20.5 Self-Attention Mechanism")
def self_attention(Q, K, V):
    d_k = Q.shape[-1]
    scores = (Q @ K.transpose(-2, -1)) / np.sqrt(d_k)
    attn_weights = torch.softmax(torch.tensor(scores), dim=-1).numpy()
    return attn_weights @ V, attn_weights

seq_len, d_k20, d_v20 = 5, 8, 8
Q20 = np.random.randn(seq_len, d_k20)
K20 = np.random.randn(seq_len, d_k20)
V20 = np.random.randn(seq_len, d_v20)
out_attn, weights = self_attention(Q20, K20, V20)
print(f"Attention output shape: {out_attn.shape}")
print(f"Attention weights (row sums = 1): {weights.sum(axis=1).round(3)}")

# PyTorch MultiheadAttention
mha = nn.MultiheadAttention(embed_dim=8, num_heads=2, batch_first=True)
Q_t = torch.randn(1, 5, 8)
attn_out, attn_w = mha(Q_t, Q_t, Q_t)
print(f"PyTorch MHA output: {attn_out.shape}")

# TF Attention
mha_tf = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=4)
Q_tf = tf.random.normal([1, 5, 8])
attn_tf = mha_tf(Q_tf, Q_tf)
print(f"TF MultiHeadAttention output: {attn_tf.shape}")

# --- 20.6 PCA in ML Pipeline ---
subsection("20.6 PCA Pipeline with Whitening")
X_pipe = np.random.randn(100, 5)
X_cent = X_pipe - X_pipe.mean(axis=0)
lam_p, Q_p = np.linalg.eigh((X_cent.T @ X_cent) / (len(X_pipe) - 1))
idx_p = np.argsort(lam_p)[::-1]
lam_p, Q_p = lam_p[idx_p], Q_p[:, idx_p]
Z_pipe = X_cent @ Q_p[:, :2]

# Whitening: decorrelate and unit variance
Lambda_half_inv = np.diag(1.0 / np.sqrt(lam_p + 1e-8))
X_white = X_cent @ Q_p @ Lambda_half_inv
print(f"PCA projection: {X_pipe.shape} → {Z_pipe.shape}")
print(f"Whitened data variance ≈ 1: {np.round(X_white.var(axis=0), 2)}")

# --- 20.7 Covariance & Gram Matrices ---
subsection("20.7 Covariance & Gram Matrices")
X_cg = np.random.randn(20, 4)
X_cg_c = X_cg - X_cg.mean(axis=0)
Cov_cg = X_cg_c.T @ X_cg_c / (len(X_cg) - 1)   # d×d
Gram_cg = X_cg_c @ X_cg_c.T                       # n×n
print(f"Covariance matrix shape: {Cov_cg.shape}  (captures feature correlations)")
print(f"Gram matrix shape: {Gram_cg.shape}  (pairwise dot products)")
print(f"Gram is symmetric: {np.allclose(Gram_cg, Gram_cg.T)}")

# SVD relationship: Cov = V Σ² V^T / (n-1)
U_rel, S_rel, Vt_rel = np.linalg.svd(X_cg_c, full_matrices=False)
Cov_from_svd = Vt_rel.T @ np.diag(S_rel**2) @ Vt_rel / (len(X_cg) - 1)
print(f"Cov from SVD == Cov direct: {np.allclose(Cov_from_svd, Cov_cg, atol=1e-10)}")

# --- 20.8 Gradient Descent & Newton's Method ---
subsection("20.8 Gradient Descent & Newton's Method")
# Minimize f(x) = x^2 + 2y^2  — quadratic bowl
def f(x):     return x[0]**2 + 2*x[1]**2
def grad_f(x): return np.array([2*x[0], 4*x[1]])
def hess_f(x): return np.array([[2,0],[0,4]], dtype=float)

# Gradient Descent
x_gd = np.array([4.0, 3.0])
for _ in range(20):
    x_gd = x_gd - 0.1 * grad_f(x_gd)
print(f"GD solution: {np.round(x_gd, 4)}  (should be [0,0])")

# Newton's Method
x_newton = np.array([4.0, 3.0])
for _ in range(5):
    x_newton = x_newton - np.linalg.inv(hess_f(x_newton)) @ grad_f(x_newton)
print(f"Newton's method solution: {np.round(x_newton, 6)}  (faster!)")

# --- 20.9 Batch Normalization ---
subsection("20.9 Batch Normalization")
def batch_norm(z, gamma, beta, eps=1e-8):
    mu    = z.mean(axis=0)
    sigma2 = z.var(axis=0)
    z_hat = (z - mu) / np.sqrt(sigma2 + eps)
    return gamma * z_hat + beta

z_bn = np.random.randn(16, 4) * 5 + 3
gamma_bn = np.ones(4)
beta_bn  = np.zeros(4)
z_normed = batch_norm(z_bn, gamma_bn, beta_bn)
print(f"Before BN — mean: {z_bn.mean(axis=0).round(2)}, std: {z_bn.std(axis=0).round(2)}")
print(f"After  BN — mean: {z_normed.mean(axis=0).round(2)}, std: {z_normed.std(axis=0).round(2)}")

# PyTorch BatchNorm
bn_layer = nn.BatchNorm1d(4)
z_bn_t = torch.tensor(z_bn, dtype=torch.float32)
z_bn_out = bn_layer(z_bn_t)
print(f"PyTorch BatchNorm output mean ≈ 0: {z_bn_out.mean(axis=0).detach().numpy().round(2)}")

# --- 20.10 SVD in NLP — Latent Semantic Analysis ---
subsection("20.10 SVD in NLP — Latent Semantic Analysis (LSA)")
# Term-document matrix (synthetic)
np.random.seed(7)
term_doc = np.random.poisson(lam=1.5, size=(20, 10)).astype(float)
U_lsa, S_lsa, Vt_lsa = np.linalg.svd(term_doc, full_matrices=False)
k_lsa = 3
A_k_lsa = U_lsa[:, :k_lsa] @ np.diag(S_lsa[:k_lsa]) @ Vt_lsa[:k_lsa, :]
print(f"Term-doc matrix: {term_doc.shape}")
print(f"LSA rank-{k_lsa} approx: {A_k_lsa.shape}")
print(f"Top-3 singular values: {np.round(S_lsa[:3], 2)}")
frob_err = np.linalg.norm(term_doc - A_k_lsa, 'fro')
print(f"Frobenius error (k=3): {frob_err:.4f}")

# Word embedding factorization M ≈ W C^T (via SVD)
M_pmi = np.random.randn(50, 100)   # PMI matrix
U_we, S_we, Vt_we = np.linalg.svd(M_pmi, full_matrices=False)
k_emb = 10
W_emb = U_we[:, :k_emb] @ np.diag(np.sqrt(S_we[:k_emb]))   # word embeddings
C_emb = Vt_we[:k_emb, :].T @ np.diag(np.sqrt(S_we[:k_emb]))  # context embeddings
print(f"\nWord embedding matrix W: {W_emb.shape}")
print(f"Context embedding matrix C: {C_emb.shape}")
print(f"M ≈ W C^T reconstruction error: "
      f"{np.linalg.norm(M_pmi - W_emb @ C_emb.T, 'fro'):.3f}")


# ============================================================
# QUICK REFERENCE — KEY FORMULAS SUMMARY
# ============================================================
section("Quick Reference: Key Formula Verifications")

np.random.seed(99)
u_qr = np.random.randn(4)
v_qr = np.random.randn(4)
A_qr = np.random.randn(4, 4)
A_sym = A_qr.T @ A_qr   # symmetric PSD

print(f"Dot product u·v = ||u||·||v||·cos(θ): "
      f"{np.isclose(np.dot(u_qr, v_qr), np.linalg.norm(u_qr)*np.linalg.norm(v_qr)*np.cos(np.arccos(np.dot(u_qr,v_qr)/(np.linalg.norm(u_qr)*np.linalg.norm(v_qr)))))}")

X_ref = np.random.randn(20, 3)
y_ref = np.random.randn(20)
theta_ref = np.linalg.inv(X_ref.T @ X_ref) @ X_ref.T @ y_ref
print(f"Normal equations θ̂ = (X^T X)^-1 X^T y: shape {theta_ref.shape}")

lam_max_ref = np.linalg.eigvalsh(A_sym).max()
w_pca = np.linalg.eigh(A_sym)[1][:, -1]
print(f"PCA: max(w^T Σ w) = λ_max(Σ): "
      f"{np.isclose(w_pca @ A_sym @ w_pca, lam_max_ref, atol=1e-8)}")

W_ref = torch.randn(3, 4, requires_grad=True)
x_ref = torch.randn(4)
z_ref = W_ref @ x_ref
L_ref = z_ref.sum()
L_ref.backward()
delta_ref = torch.ones(3)
print(f"∂L/∂W = δ x^T: {torch.allclose(W_ref.grad, torch.outer(delta_ref, x_ref))}")

U_svd, S_svd, Vt_svd = np.linalg.svd(A_qr[:2, :3])
k_ref = 1
A_rank1 = S_svd[0] * np.outer(U_svd[:, 0], Vt_svd[0, :])
print(f"Rank-1 SVD approx: A_1 = σ_1 u_1 v_1^T, shape: {A_rank1.shape}")

print(f"\n{SEP}")
print("  ALL 20 TOPICS IMPLEMENTED SUCCESSFULLY")
print(f"  NumPy | PyTorch | TensorFlow | SciPy")
print(SEP)
