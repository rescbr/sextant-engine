# PQ LUT SIMD — negative result (for single-code path)

> **Update (2026-07-23):** The single-code `lut_distance` scalar path was
> partially eliminated by the batch4 remainder fix (commit bf46fb1) — the
> 1-3 neighbor tail now uses a padded `lut_distance_batch4` call instead of
> scalar fallback. The single-code `lut_distance` still exists for entry-point
> seeding (which is rare and on FP16-ball nodes anyway) and for non-LUT build
> paths. The SIMD investigation below remains valid: the single-code path
> can't be SIMD-accelerated without a LUT layout change.

> Investigated 2026-07-20. Three SIMD attempts (manual SVE intrinsics v1,
> manual v2, clang auto-vec) all failed to meaningfully speed up
> `PqQuantizer::lut_distance`. The scalar loop remains. Documenting why so
> we don't re-attempt this without a layout change.

## Setup

After the multithread-scaling fix (see `docs/scaling_fix_results.md`),
`PqQuantizer::lut_distance` became the #1 hot spot at **28.99% of 1t cycles**
(was masked at ~5% by atomic contention). The plan was to vectorize it.

## Instruction-level profile of the scalar loop

```
0x6004c add w16, w8, w11         24.67%   ← address math: s*K_ + cid
0x60050 subs x14, x14, #0x2      50.00%   ← loop counter
0x6005c ldr s1, [x2, w15, *2]     1.81%   ← gather (f32 load from LUT)
0x60060 ldrb w15, [x13], #2      10.86%   ← byte load of code[s]
0x60058 add w11, w11, w12         1.81%   ← increment s by 2*K_
0x60064 fadd s0, s0, s1           1.48%   ← 1st fadd (loop is 2× unrolled)
0x6006c ldr s1, [x2, w15, *2]     0.99%   ← 2nd gather
0x60070 fadd s0, s0, s1           4.77%   ← 2nd fadd
0x60074 b.ne loop                 1.64%   ← branch
```

**Surprising finding:** the gathers themselves were only ~3% of cycles.
The hardware pipelines gather latency well. The bottleneck was the
scalar overhead around them: loop counter (50%) + address math (25%)
+ byte load (11%) = 86%.

This suggested SIMD would help dramatically — vectorize 4 subspaces at
once and the per-iter overhead collapses to 1/4 of the iterations.

## Three SIMD attempts — all flat

### Attempt 1: manual SVE intrinsics (4 subspaces per iter)

```cpp
for (uint32_t s = 0; s + 3 < m_; s += 4) {
    svuint32_t cids = svld1ub_u32(pg4, code + s);    // 4 bytes → 4 u32
    svuint32_t idx = svdup_u32(s * K_);
    idx = svadd_u32_m(pg4, idx, strides);
    idx = svadd_u32_m(pg4, idx, cids);
    svfloat32_t vals = svld1_gather_u32index_f32(pg4, lut, idx);
    acc = svadd_f32_m(pg4, acc, vals);
}
```

Result: lut_distance went from 28.99% → **31% of cycles**. **Worse.**

Annotation showed the new hot spot:
```
0x60040 add x15, x13, #0x7      73.09%   ← loop bound (s + 3 < m_)
0x6005c fadd z0.s, p0/m, ...    22.63%   ← actual SIMD add
```

The compiler emitted a per-iteration `(s + 7) < (m_ + 4)` check that
itself became the bottleneck. Scalar overhead just moved, didn't shrink.

### Attempt 2: fixed register pressure + loop bound

```cpp
const uint32_t iters = m_ >> 2;          // trip count once
const svuint32_t strides = svindex_u32(0u, K_);  // in registers, no stack
for (uint32_t i = 0; i < iters; ++i) {
    svuint32_t cids = svld1ub_u32(pg4, code_p);
    code_p += 4;
    svuint32_t idx = svadd_u32_m(pg4, strides, cids);
    ...
    lut_p += 4 * K_;
}
```

Result: lut_distance 30% of cycles. Marginally better than v1, still
**worse than scalar**.

Annotation:
```
0x60054 mov z3.d, z0.d   72.30%   ← register copy of strides
0x60068 add x13, x13, x11 19.94%  ← lut_p advance
0x6006c fadd z1.s, ...     3.96%
```

clang refused to keep `strides` in a stable register — it copied it
every iteration. Conservative register allocation under the predicate
register constraints.

### Attempt 3: clang auto-vectorization with `#pragma`

```cpp
#pragma clang loop vectorize(enable) interleave(enable)
for (uint32_t s = 0; s < m; ++s) {
    acc += lut[s * K + code[s]];
}
```

Result: lut_distance 30.5% of cycles. Same as our manual attempts.
clang did vectorize (8-wide, 2 vectors per iter) but produced
essentially the same code.

Annotation:
```
0x60088 add z5.s, z4.s, z2.s   74.04%   ← index computation
0x600a8 ld1w {z6.s}, ...       14.81%   ← gather (now doing real work)
0x600ac fadd z1.s, ...          7.96%
```

## Why none of them worked

The total work didn't change. For 96 subspaces:

- Compute 96 gather indices: `idx[s] = s * K_ + code[s]`. Each lane
  needs an independent multiply + add. This is **96 independent scalar
  operations** regardless of vectorization — the dependency is on
  `code[s]` (loaded) and `s` (loop counter), with no spatial locality
  in the index stream.
- Gather 96 f32 values.
- Add 96 f32 values.

SIMD parallelizes the gather and the add (4-8 at once), but the
**index computation has no ILP to extract** — each index is data-
dependent on a different `code[s]` byte. The scalar version's loop
overhead (50% of cycles) was just hiding the latency of this index
computation; removing it doesn't reduce the work, just makes the
index computation the new visible bottleneck.

## What would actually help

**LUT layout change: row-major → column-major.**

Current LUT (row-major, `m × K`): `lut[s * K + cid]`.
Gathering one code's distance requires 96 independent indices.

Transposed LUT (column-major, `K × m`): `lut[cid * m + s]`.
For a single code's cid sequence, the 96 values at `lut[c0*96+0..95]`,
`lut[c1*96+0..95]`, etc. would be... still scattered. Doesn't help
directly.

Better: **precompute per-cid offsets at preprocess_query time**. Walk
the codebook once and produce, for each candidate cid-sequence the
search might see, a packed distance vector. But we don't know cid-
sequences ahead of time.

The actually useful transform: **split the LUT per subspace and pack
each subspace's K centroids contiguously in cache-friendly tiles**.
Reduces gather to a single strided load per subspace. This is closer
to FAISS's PQ implementation. Non-trivial format change.

## Conclusion

**SIMD alone cannot speed up `lut_distance` without a LUT layout change.**
The index computation is the bottleneck, not the gather latency or the
scalar loop overhead. All three SIMD attempts landed at ~30% of cycles
(same as scalar's 29%) because they compute the same 96 indices.

The scalar loop stays. The path to a real single-core win here is a
format change to the LUT (per-subspace packing, column-major, or FAISS-
style precomputed distances) — deferred to a separate plan that
includes the serialization impact.

## What was learned

- Gather latency on Neoverse-V2 is **not** the bottleneck people
  assume. The hardware pipelines gathers well; the surrounding scalar
  work matters more.
- `#pragma clang loop vectorize(enable)` does work but doesn't beat
  careful scalar code when the index stream has no spatial locality.
- Profile at the instruction level before writing SIMD. The 28.99%
  cycle share looked like a clear SIMD target; the annotation revealed
  it was already 90%+ scalar overhead that SIMD would just relocate.
- The "obvious" next lever isn't always the right one. Worth the
  half-day of experimentation to find out.
