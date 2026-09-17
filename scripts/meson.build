# Microbenchmarks and exploratory scripts. Not installed; built on demand
# (ninja -C build bench_preprocess_query). Each is a standalone executable.

# NEON-intrinsic spikes/microbenches: ARM-only sources (bare #include
# <arm_neon.h>, no guards — they are archival measurement code). Excluded
# from x86 builds; port via the simd:: kernel API if ever needed again.
is_aarch64 = host_machine.cpu_family() == 'aarch64'

bench_preprocess_query = executable('bench_preprocess_query',
    'bench_preprocess_query.cpp',
    dependencies: sextant_dep,
    include_directories: numkong_inc,  # uses nk_sqeuclidean_f32/nk_dot_f32
    install: false,
)

if is_aarch64
bench_fastscan = executable('bench_fastscan',
    'bench_fastscan.cpp',
    dependencies: sextant_dep,
    install: false,
)
endif

if is_aarch64
spike_pq4_recall = executable('spike_pq4_recall',
    'spike_pq4_recall.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_hybrid_ivf = executable('spike_hybrid_ivf',
    'spike_hybrid_ivf.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_precision_cmp = executable('spike_precision_cmp',
    'spike_precision_cmp.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_large_k = executable('spike_large_k',
    'spike_large_k.cpp',
    dependencies: sextant_dep,
    install: false,
)

bench_mem = executable('bench_mem',
    'bench_mem.cpp',
    dependencies: sextant_dep,
    install: false,
)
endif

if is_aarch64
spike_margin_rerank = executable('spike_margin_rerank',
    'spike_margin_rerank.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_graph_coarse_router = executable('spike_graph_coarse_router',
    'spike_graph_coarse_router.cpp',
    dependencies: sextant_dep,
    install: false,
)
endif

spike_pq_bounds = executable('spike_pq_bounds',
    'spike_pq_bounds.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_panorama_rerank = executable('spike_panorama_rerank',
    'spike_panorama_rerank.cpp',
    dependencies: sextant_dep,
    install: false,
)

if is_aarch64
bench_kernel = executable('bench_kernel',
    'bench_kernel.cpp',
    dependencies: sextant_dep,
    install: false,
)
endif

spike_lm_recall = executable('spike_lm_recall',
    'spike_lm_recall.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

spike_pq2_tier = executable('spike_pq2_tier',
    'spike_pq2_tier.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

spike_local_shape = executable('spike_local_shape',
    'spike_local_shape.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

spike_routing = executable('spike_routing',
    'spike_routing.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: sextant_dep,
)

spike_routing_ceiling = executable('spike_routing_ceiling',
    'spike_routing_ceiling.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: sextant_dep,
)

plane_selftest = executable('plane_selftest',
    'plane_selftest.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: sextant_dep,
)

spike_slm_roundtrip = executable('spike_slm_roundtrip',
    'spike_slm_roundtrip.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

spike_slm_storage = executable('spike_slm_storage',
    'spike_slm_storage.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

capi_smoke = executable('capi_smoke',
    'capi_smoke.cpp',
    include_directories: [engine_inc, src_inc, sources_inc],
    link_with: [sextant_lib, sextant_sources_lib, sextant_c_lib],
)

spike_uniform_check = executable('spike_uniform_check',
    'spike_uniform_check.cpp',
    include_directories: [engine_inc, src_inc],
    dependencies: sextant_dep,
)

capi_qps = executable('capi_qps',
    'capi_qps.cpp',
    include_directories: [engine_inc, src_inc, sources_inc],
    link_with: [sextant_lib, sextant_sources_lib, sextant_c_lib],
)

spike_bits_elbow = executable('spike_bits_elbow',
    'spike_bits_elbow.cpp',
    dependencies: sextant_dep,
    install: false,
)

sched_transition = executable('sched_transition',
    'sched_transition.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: sextant_dep,
)

join_sim = executable('join_sim',
    'join_sim.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: sextant_dep,
    install: false,
)

probe_shape = executable('probe_shape',
    'probe_shape.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc, sources_inc],
    dependencies: [sextant_dep, sextant_sources_dep],
    cpp_args: ['-DSEXTANT_SOURCE_DIR="' + meson.source_root() + '"'],
    install: false,
)

probe_batch = executable('probe_batch',
    'probe_batch.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc, sources_inc],
    dependencies: [sextant_dep, sextant_sources_dep],
    install: false,
)
