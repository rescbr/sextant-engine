"""Verify jacobi_eigen (C++) at d=768 against numpy.linalg.eigh.

Generates a random symmetric PSD covariance, feeds it to the C++ jacobi_check
binary, and compares eigenvalues + the reconstruction C = V diag(λ) V^T.
Also times the eigendecomposition.
"""
import numpy as np
import subprocess
import time
import struct

DIM = 768
SEED = 12345
rng = np.random.default_rng(SEED)

# Build a random symmetric PSD matrix: C = X X^T / n + small diag (realistic
# covariance structure with correlated dims).
X = rng.standard_normal((DIM, DIM)).astype(np.float64)
C = (X @ X.T) / DIM
# Symmetrize exactly.
C = (C + C.T) / 2.0

# numpy reference.
t0 = time.time()
w_np, V_np = np.linalg.eigh(C)
t_np = time.time() - t0
print(f"numpy eigh: {t_np:.3f}s")

# Feed to C++ jacobi_check.
buf = struct.pack("<I", DIM) + C.astype("<f8").tobytes()
t0 = time.time()
res = subprocess.run(["/tmp/jacobi_check"], input=buf, capture_output=True)
t_jacobi = time.time() - t0
if res.returncode != 0:
    print("C++ FAILED:", res.stderr.decode())
    raise SystemExit(1)
print(f"jacobi C++: {t_jacobi:.3f}s")

out = res.stdout
ev = np.frombuffer(out[: DIM * 8], dtype="<f8")
evec = np.frombuffer(out[DIM * 8 :], dtype="<f8").reshape(DIM, DIM)

# Compare eigenvalues (sorted — jacobi returns unordered).
ev_s = np.sort(ev)
w_s = np.sort(w_np)
rel_err = np.abs(ev_s - w_s) / (np.abs(w_s) + 1e-12)
print(f"eigval max rel err: {rel_err.max():.2e}")
print(f"eigval median rel err: {np.median(rel_err):.2e}")

# Verify eigvecs reconstruct C: V diag(λ) V^T == C.
# jacobi stores eigvecs as columns: evec[:, j] is eigenvector j.
C_reconstructed = evec @ np.diag(ev) @ evec.T
recon_err = np.abs(C_reconstructed - C).max() / (np.abs(C).max() + 1e-12)
print(f"reconstruction V diag(λ) V^T rel err: {recon_err:.2e}")

# Verify orthonormality of eigenvectors.
ortho = evec.T @ evec
ortho_err = np.abs(ortho - np.eye(DIM)).max()
print(f"orthonormality err (V^T V - I): {ortho_err:.2e}")

# Check A v = λ v for a few vectors.
ok_av = True
for j in [0, 100, 400, 767]:
    v = evec[:, j]
    lhs = C @ v
    rhs = ev[j] * v
    if np.abs(lhs - rhs).max() / (np.abs(rhs).max() + 1e-12) > 1e-6:
        ok_av = False
        print(f"  A v != λ v for col {j}: err={np.abs(lhs - rhs).max():.2e}")
print(f"A v = λ v holds: {ok_av}")

if recon_err < 1e-8 and ortho_err < 1e-8 and rel_err.max() < 1e-9:
    print("PASS: jacobi_eigen matches numpy eigh at d=768")
else:
    print("FAIL")
    raise SystemExit(1)
