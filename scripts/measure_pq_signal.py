"""Measure PQ approximation quality for a dataset's .fbin.
Computes: ratio of PQ-code distance to true L2 distance for random pairs.
A ratio close to 1.0 means PQ is faithful → aggressive build params OK.
A ratio >> 1.0 means PQ overestimates distances → need conservative params.
"""
import struct, math, random, sys

def read_fbin(path, max_n=None):
    with open(path, 'rb') as f:
        n, d = struct.unpack('<II', f.read(8))
        if max_n and n > max_n:
            n = max_n
        import array
        data = array.array('f')
        data.fromfile(f, n * d)
    return data, n, d

def pq_train(data, n, d, m, K=256, iters=25, seed=42):
    """Train PQ codebook (m segments, K centroids each) via k-means."""
    sub_d = d // m
    random.seed(seed)
    centroids = [[None]*K for _ in range(m)]
    codes = bytearray(n * m)
    for s in range(m):
        # Extract sub-vectors for this segment
        sub = [data[i*d + s*sub_d : i*d + s*sub_d + sub_d] for i in range(n)]
        # k-means++ init (random for speed)
        for k in range(K):
            idx = random.randint(0, n-1)
            centroids[s][k] = list(sub[idx])
        # Lloyd iterations
        for _ in range(iters):
            assign = [0]*n
            for i in range(n):
                best_d = 1e18; best_k = 0
                for k in range(K):
                    dd = sum((sub[i][j]-centroids[s][k][j])**2 for j in range(sub_d))
                    if dd < best_d: best_d = dd; best_k = k
                assign[i] = best_k
            sums = [[0.0]*sub_d for _ in range(K)]
            counts = [0]*K
            for i in range(n):
                k = assign[i]
                counts[k] += 1
                for j in range(sub_d):
                    sums[k][j] += sub[i][j]
            for k in range(K):
                if counts[k] > 0:
                    centroids[s][k] = [sums[k][j]/counts[k] for j in range(sub_d)]
        # Final assignment
        for i in range(n):
            best_d = 1e18; best_k = 0
            for k in range(K):
                dd = sum((sub[i][j]-centroids[s][k][j])**2 for j in range(sub_d))
                if dd < best_d: best_d = dd; best_k = k
            codes[i*m + s] = best_k
    # Build cross-distance table
    table = [[0.0]*K*K for _ in range(m)]
    for s in range(m):
        for a in range(K):
            for b in range(a, K):
                dd = sum((centroids[s][a][j]-centroids[s][b][j])**2 for j in range(sub_d))
                table[s][a*K+b] = dd
                table[s][b*K+a] = dd
    return codes, m, K, table

def code_dist(codes, m, K, table, i, j):
    acc = 0.0
    for s in range(m):
        acc += table[s][codes[i*m+s]*K + codes[j*m+s]]
    return acc

def true_dist(data, d, i, j):
    acc = 0.0
    for k in range(d):
        diff = data[i*d+k] - data[j*d+k]
        acc += diff*diff
    return acc

path = sys.argv[1]
N_SAMPLE = 10000  # train PQ on this many
N_PAIRS = 50000   # measure this many pairs

print(f"Reading {path} (sampling {N_SAMPLE})...")
data, full_n, d = read_fbin(path, N_SAMPLE)
n = N_SAMPLE
print(f"  n={n} d={d}")

m = min(d//4, 64)
while d % m != 0: m -= 1
print(f"Training PQ (m={m})...")
codes, m, K, table = pq_train(data, n, d, m)

print(f"Measuring {N_PAIRS} random pairs...")
random.seed(123)
ratios = []
pq_dists = []
true_dists = []
for _ in range(N_PAIRS):
    i = random.randint(0, n-1)
    j = random.randint(0, n-1)
    if i == j: continue
    cd = code_dist(codes, m, K, table, i, j)
    td = true_dist(data, d, i, j)
    if td > 1e-10:
        ratios.append(cd / td)
        pq_dists.append(cd)
        true_dists.append(td)

mean_ratio = sum(ratios)/len(ratios)
mean_pq = sum(pq_dists)/len(pq_dists)
mean_true = sum(true_dists)/len(true_dists)
var_ratio = sum((r-mean_ratio)**2 for r in ratios)/len(ratios)
std_ratio = math.sqrt(var_ratio)
# Correlation
import math
mean_x = sum(true_dists)/len(true_dists)
mean_y = sum(pq_dists)/len(pq_dists)
cov = sum((true_dists[i]-mean_x)*(pq_dists[i]-mean_y) for i in range(len(ratios)))/len(ratios)
var_x = sum((x-mean_x)**2 for x in true_dists)/len(true_dists)
var_y = sum((y-mean_y)**2 for y in pq_dists)/len(pq_dists)
corr = cov / (math.sqrt(var_x*var_y) + 1e-10)

print(f"\n=== PQ Approximation Signal ===")
print(f"PQ distance / true distance ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")
print(f"Spearman-like correlation (PQ vs true): {corr:.4f}")
print(f"Mean true L2: {mean_true:.1f}")
print(f"Mean PQ L2:   {mean_pq:.1f}")
print(f"\nInterpretation:")
print(f"  ratio ≈ 1.0 + low std → PQ faithful → aggressive params OK")
print(f"  ratio >> 1.0 → PQ overestimates → conservative params needed")
print(f"  correlation < 0.9 → ranking errors → need higher L_build")
