// Memory subsystem benchmark — Mac vs c4a comparison + multi-thread scaling.
//
// Usage:
//   bench_mem                  # single-threaded (per-core behavior)
//   bench_mem --threads N      # N threads, one buffer each (bandwidth scaling)
//
// Measures:
// (1) DRAM sequential read (large buffer, no cache)
// (2) L1/L2-resident sequential read (small buffer)
// (3) Latency-bound random chase (linked-list across 64B lines)
// (4) Stream-triad (a = b*scalar + c)
//
// For multi-threaded mode, each test runs N copies in parallel threads; the
// reported bandwidth is the SUM across threads (so 1→2 threads doubling the
// number means full scaling; plateau means bandwidth-saturated).
//
// Defeats dead-code elimination via `clobber()` (asm volatile memory barrier)
// — necessary at -O3, otherwise the compiler elides the loads.

#include <algorithm>
#include <arm_neon.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

using namespace std;

double secs() {
    return std::chrono::duration<double>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

template<typename T> inline void clobber(T& v) { asm volatile("" : "+r"(v) : : "memory"); }

struct Args {
    uint32_t threads = 1;
    int iters = 50;
};

static Args parse_args(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if ((s == "--threads" || s == "-t") && i + 1 < argc) {
            a.threads = (uint32_t)atoi(argv[++i]);
            if (a.threads == 0) a.threads = 1;
        } else if ((s == "--iters" || s == "-n") && i + 1 < argc) {
            a.iters = atoi(argv[++i]);
            if (a.iters < 1) a.iters = 50;
        } else if (s == "--help" || s == "-h") {
            printf("Usage: bench_mem [--threads N] [--iters N]\n");
            exit(0);
        }
    }
    return a;
}

// (1) DRAM sequential read.
static double bench_dram_seq(uint32_t tid, uint32_t nthreads, int iters) {
    const size_t N = 256 * 1024 * 1024;
    vector<uint8_t> buf(N);
    // Each thread uses a distinct buffer (different physical pages) so the
    // threads don't share cache lines.
    for (size_t i = 0; i < N; i++) buf[i] = (uint8_t)((i * 7 + tid) & 0xFF);
    uint64_t sink = 0;
    for (size_t i = 0; i < N; i += 4096) sink += buf[i];
    clobber(sink);
    double t0 = secs();
    for (int it = 0; it < iters; it++) {
        uint8x16x4_t acc;
        for (int r = 0; r < 4; r++) acc.val[r] = vdupq_n_u8(0);
        for (size_t i = 0; i + 64 <= N; i += 64) {
            acc.val[0] = veorq_u8(acc.val[0], vld1q_u8(&buf[i+0]));
            acc.val[1] = veorq_u8(acc.val[1], vld1q_u8(&buf[i+16]));
            acc.val[2] = veorq_u8(acc.val[2], vld1q_u8(&buf[i+32]));
            acc.val[3] = veorq_u8(acc.val[3], vld1q_u8(&buf[i+48]));
        }
        uint8x16_t x = veorq_u8(veorq_u8(acc.val[0], acc.val[1]),
                                veorq_u8(acc.val[2], acc.val[3]));
        sink += vaddvq_u8(x);
        clobber(sink);
    }
    return secs() - t0;
}

// (2) L1/L2-resident sequential read.
static double bench_l2_seq(uint32_t tid, uint32_t nthreads, int iters) {
    const size_t N = 32 * 1024;
    vector<uint8_t> buf(N);
    for (size_t i = 0; i < N; i++) buf[i] = (uint8_t)((i * 7 + tid) & 0xFF);
    uint64_t sink = 0;
    const uint64_t PASSES = (uint64_t)256 * 1024 * 1024 * iters / N;
    double t0 = secs();
    for (uint64_t p = 0; p < PASSES; p++) {
        uint8x16x4_t acc;
        for (int r = 0; r < 4; r++) acc.val[r] = vdupq_n_u8(0);
        for (size_t i = 0; i + 64 <= N; i += 64) {
            acc.val[0] = veorq_u8(acc.val[0], vld1q_u8(&buf[i+0]));
            acc.val[1] = veorq_u8(acc.val[1], vld1q_u8(&buf[i+16]));
            acc.val[2] = veorq_u8(acc.val[2], vld1q_u8(&buf[i+32]));
            acc.val[3] = veorq_u8(acc.val[3], vld1q_u8(&buf[i+48]));
        }
        uint8x16_t x = veorq_u8(veorq_u8(acc.val[0], acc.val[1]),
                                veorq_u8(acc.val[2], acc.val[3]));
        sink += vaddvq_u8(x);
        if ((p & 0xFFFF) == 0) clobber(sink);
    }
    clobber(sink);
    return secs() - t0;
}

// (3) Latency-bound random chase (linked list across 64B-aligned lines).
//
// The chase reads `chain[pos]` to get the next `pos`. The hardware can't
// prefetch (dependent load chain) so this measures pure memory latency.
//
// IMPORTANT: clang is *extremely* aggressive about this pattern. Even with
// `__asm__ volatile("ldr..." : "+r"(pos) : ... : "memory")` per-iteration,
// clang's MLIR pipeline analyzes the `+r` data flow and rewrites the loop
// — replacing the dependent load with `pos += stride` in some cases, or
// terminating the loop early when the random cycle returns to its start.
// (Confirmed via radare2 disassembly — the inner loop had NO load, just
// `add w8, w8, w10` with stride 0x1eef.)
//
// The fix: write the ENTIRE inner loop in asm. The compiler has no part of
// the loop body to optimize. We pass it the start pos, chain pointer, and
// step count; it returns the final pos. Each step is one dependent load.
static double bench_random_chase(uint32_t tid, uint32_t nthreads, int iters) {
    const size_t N_LINES = 64 * 1024 * 1024 / 64;  // 1M lines = 64 MB
    vector<uint32_t> chain(N_LINES);
    vector<uint32_t> perm(N_LINES);
    for (uint32_t i = 0; i < N_LINES; i++) perm[i] = i;
    unsigned int seed = 42 + tid;
    for (uint32_t i = N_LINES - 1; i > 0; i--) {
        uint32_t j = rand_r(&seed) % (i + 1);
        swap(perm[i], perm[j]);
    }
    for (uint32_t i = 0; i < N_LINES; i++)
        chain[perm[i]] = perm[(i+1) % N_LINES];
    const uint32_t* chain_p = chain.data();

    // Warmup (small chase).
    {
        uint32_t p = 0;
        uint64_t steps = 1024;
        __asm__ volatile(
            "1: ldr %w[pos], [%[base], %w[pos], uxtw #2]\n\t"
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [pos] "+r"(p), [n] "+r"(steps)
            : [base] "r"(chain_p)
            : "memory", "cc");
    }

    uint64_t sink = 0;
    const uint64_t STEPS = 50000000 / iters;
    double t0 = secs();
    for (int it = 0; it < iters; it++) {
        uint32_t pos = (it * 7919) & (N_LINES - 1);  // vary start across iters
        uint64_t steps = STEPS;
        // Entire inner loop in asm. The compiler cannot analyze this.
        // %w[pos] = 32-bit pos register; uses indexed addressing
        // [base, pos, uxtw #2] = base + pos*4 (uint32_t stride).
        __asm__ volatile(
            "1: ldr %w[pos], [%[base], %w[pos], uxtw #2]\n\t"
            "subs %[n], %[n], #1\n\t"
            "b.ne 1b\n\t"
            : [pos] "+r"(pos), [n] "+r"(steps)
            : [base] "r"(chain_p)
            : "memory", "cc");
        sink += pos;
        __asm__ volatile("" : "+r"(sink) : : "memory");
    }
    return secs() - t0;
}

// (4) Stream-triad (a = b*scalar + c), 256 MB × 3.
static double bench_stream_triad(uint32_t tid, uint32_t nthreads, int iters) {
    const size_t N = 64 * 1024 * 1024;  // 64M floats = 256 MB
    vector<float> a(N, 0), b(N), c(N);
    unsigned int seed = 42 + tid;
    for (size_t i = 0; i < N; i++) {
        b[i] = (rand_r(&seed) / float(RAND_MAX));
        c[i] = (rand_r(&seed) / float(RAND_MAX));
    }
    const float sc = 1.234f;
    float sk = 0;
    for (size_t i = 0; i < N; i += 64) sk += b[i] + c[i];
    clobber(sk);
    double t0 = secs();
    for (int it = 0; it < iters; it++) {
        for (size_t i = 0; i + 16 <= N; i += 16) {
            float32x4_t b0 = vld1q_f32(&b[i]); float32x4_t b1 = vld1q_f32(&b[i+4]);
            float32x4_t b2 = vld1q_f32(&b[i+8]); float32x4_t b3 = vld1q_f32(&b[i+12]);
            float32x4_t c0 = vld1q_f32(&c[i]); float32x4_t c1 = vld1q_f32(&c[i+4]);
            float32x4_t c2 = vld1q_f32(&c[i+8]); float32x4_t c3 = vld1q_f32(&c[i+12]);
            vst1q_f32(&a[i],   vmlaq_n_f32(c0, b0, sc));
            vst1q_f32(&a[i+4], vmlaq_n_f32(c1, b1, sc));
            vst1q_f32(&a[i+8], vmlaq_n_f32(c2, b2, sc));
            vst1q_f32(&a[i+12],vmlaq_n_f32(c3, b3, sc));
        }
        sk += a[it % N];
        clobber(sk);
    }
    return secs() - t0;
}

int main(int argc, char** argv) {
    Args args = parse_args(argc, argv);

    printf("=== Memory subsystem benchmark ===\n");
#ifdef __APPLE__
    printf("Platform: macOS (Apple Silicon)\n");
#else
    printf("Platform: Linux (c4a Axion V2)\n");
#endif
    printf("Logical CPUs: %u | threads under test: %u | iters: %d\n\n",
           thread::hardware_concurrency(), args.threads, args.iters);

    auto run_test = [&](const char* label, auto fn, double scale_gb, double per_call_count) {
        // IMPORTANT: each test fn() internally loops over `iters`, so its
        // returned wall-time already covers `iters` inner passes. Don't
        // multiply by iters here. Total work across all threads =
        // per_call_count × threads (NOT × iters — that double-counts).
        vector<double> times(args.threads, 0.0);
        vector<thread> ts;
        for (uint32_t t = 0; t < args.threads; t++) {
            ts.emplace_back([&, t]() {
                times[t] = fn(t, args.threads, args.iters);
            });
        }
        for (auto& th : ts) th.join();
        double max_t = *max_element(times.begin(), times.end());
        double total_gb = scale_gb * args.threads;
        double total_count = per_call_count * args.threads;
        if (scale_gb > 0) {
            printf("  %-30s %7.1f GB/s  (wall=%.2fs, max-thread=%.2fs)\n",
                   label, total_gb / max_t, max_t, max_t);
        } else {
            printf("  %-30s %7.1f M/s  (wall=%.2fs, %.2f ns/op)\n",
                   label, total_count / max_t / 1e6, max_t,
                   max_t * 1e9 / total_count);
        }
    };

    printf("--- per-thread results (each thread = independent workload) ---\n");
    // Per-call work (one fn() invocation = iters inner passes):
    //   (1) 256 MB × iters
    //   (2) 256 MB worth of L1/L2 passes (256MB total, distributed across iters)
    //   (3) 50M dependent loads total (STEPS=50M/iters, × iters outer)
    //   (4) 768 MB (3 × 256MB) × iters
    run_test("(1) DRAM seq read (256 MB)",  bench_dram_seq,      256.0 * args.iters * 1024 * 1024 / 1e9, 0);
    run_test("(2) L1/L2 seq read (32 KB)",  bench_l2_seq,        256.0 * args.iters * 1024 * 1024 / 1e9, 0);
    run_test("(3) Random chase (4MB chain)",bench_random_chase,  0, 50e6);
    run_test("(4) Stream-triad (768 MB)",   bench_stream_triad,  64e6 * 3 * 4 * args.iters / 1e9, 0);
    printf("\n");
    printf("Interpretation:\n");
    printf("  - If N-thread GB/s scales ~N× over 1-thread: bandwidth NOT saturated.\n");
    printf("  - If N-thread GB/s plateaus: bandwidth-saturated (memory throttling).\n");
    printf("  - Random chase is latency-bound: M/s should NOT scale with threads\n");
    printf("    until queue depth > 1 per core (NEON has no hardware prefetcher for it).\n");
    return 0;
}
