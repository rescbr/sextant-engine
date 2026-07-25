# Microbenchmarks and exploratory scripts. Not installed; built on demand
# (ninja -C build bench_preprocess_query). Each is a standalone executable.

bench_preprocess_query = executable('bench_preprocess_query',
    'bench_preprocess_query.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc, numkong_inc],
    install: false,
)

bench_fastscan = executable('bench_fastscan',
    'bench_fastscan.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc, numkong_inc],
    install: false,
)

spike_pq4_recall = executable('spike_pq4_recall',
    'spike_pq4_recall.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc, numkong_inc],
    install: false,
)
