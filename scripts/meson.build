# Microbenchmarks and exploratory scripts. Not installed; built on demand
# (ninja -C build bench_preprocess_query). Each is a standalone executable.

bench_preprocess_query = executable('bench_preprocess_query',
    'bench_preprocess_query.cpp',
    dependencies: sextant_dep,
    include_directories: numkong_inc,  # uses nk_sqeuclidean_f32/nk_dot_f32
    install: false,
)

bench_fastscan = executable('bench_fastscan',
    'bench_fastscan.cpp',
    dependencies: sextant_dep,
    install: false,
)

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

spike_colocated = executable('spike_colocated',
    'spike_colocated.cpp',
    dependencies: sextant_dep,
    install: false,
)

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

bench_kernel = executable('bench_kernel',
    'bench_kernel.cpp',
    dependencies: sextant_dep,
    install: false,
)

spike_lm_recall = executable('spike_lm_recall',
    'spike_lm_recall.cpp',
    include_directories: [engine_inc, src_inc],
    link_with: sextant_lib,
)

spike_routing = executable('spike_routing',
    'spike_routing.cpp',
    include_directories: [engine_inc, src_inc, third_party_inc],
    link_with: sextant_lib,
)

spike_slm_roundtrip = executable('spike_slm_roundtrip',
    'spike_slm_roundtrip.cpp',
    include_directories: [engine_inc, src_inc],
    link_with: sextant_lib,
)

spike_slm_storage = executable('spike_slm_storage',
    'spike_slm_storage.cpp',
    include_directories: [engine_inc, src_inc],
    link_with: sextant_lib,
)
