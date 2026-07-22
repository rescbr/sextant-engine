"""Time jacobi on the REAL arxiv100k covariance (not synthetic)."""
import numpy as np
import subprocess
import time
import struct

DIM = 768
N_SAMPLE = 20000

# Load arxiv100k, take a 20K sample (same as PQ reservoir).
import mmap


def read_fbin(path):
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        n = np.frombuffer(mm[:4], dtype=np.int32)[0]
        d = np.frombuffer(mm[4:8], dtype=np.int32)[0]
        arr = np.frombuffer(mm[8:], dtype=np.float32, count=n * d).reshape(n, d)
        return arr


data = read_fbin("datasets/arxiv100k_base.fbin")
print(f"loaded arxiv100k: {data.shape}")
rng = np.random.default_rng(0)
idx = rng.choice(data.shape[0], size=min(N_SAMPLE, data.shape[0]), replace=False)
sample = data[idx].astype(np.float64)

# Covariance (same pattern as anisotropic_scale).
mean = sample.mean(axis=0)
centered = sample - mean
C = (centered.T @ centered) / (sample.shape[0] - 1)
C = (C + C.T) / 2.0
print(f"cov shape: {C.shape}, trace={np.trace(C):.3f}")

buf = struct.pack("<I", DIM) + C.astype("<f8").tobytes()
t0 = time.time()
res = subprocess.run(["/tmp/jacobi_check"], input=buf, capture_output=True)
t_jacobi = time.time() - t0
print(f"jacobi C++ on REAL arxiv100k cov: {t_jacobi:.3f}s")
if res.returncode != 0:
    print("FAILED:", res.stderr.decode())
    raise SystemExit(1)

out = res.stdout
ev = np.frombuffer(out[: DIM * 8], dtype="<f8")
evec = np.frombuffer(out[DIM * 8 :], dtype="<f8").reshape(DIM, DIM)

# Cross-check eigenvalues vs numpy.
w_np = np.linalg.eigvalsh(C)
rel = np.abs(np.sort(ev) - np.sort(w_np)) / (np.abs(np.sort(w_np)) + 1e-12)
print(f"vs numpy eigval max rel err: {rel.max():.2e}")

# reconstruction
C_recon = evec @ np.diag(ev) @ evec.T
print(f"reconstruction rel err: {np.abs(C_recon - C).max() / (np.abs(C).max()):.2e}")
print(f"orthonormality err: {np.abs(evec.T @ evec - np.eye(DIM)).max():.2e}")
print(f"eigval range: [{ev.min():.4e}, {ev.max():.4e}], "
      f"top-10/total variance: {np.sort(ev)[-10:].sum()/ev.sum():.3f}")
