sextant_cli = executable('sextant',
    ['sextant_cli.cpp', 'analyze.cpp'],
    dependencies: [sextant_dep, sextant_sources_dep],
    include_directories: [engine_inc, third_party_inc, version_inc],
    install: false,
)

fvecs_to_fbin = executable('fvecs_to_fbin',
    'fvecs_to_fbin.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc],
    install: false,
)

sextant_bench = executable('sextant_bench',
    'benchmark.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc, version_inc],
    install: false,
)

pq_explore = executable('pq_explore',
    'pq_explore.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc],
    install: false,
)

sextant_filtered_gt = executable('sextant_filtered_gt',
    'compute_filtered_gt.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc],
    install: false,
)

fbin_to_parquet = executable('fbin_to_parquet',
    'fbin_to_parquet.cpp',
    dependencies: [sextant_dep, carquet_dep],
    include_directories: [engine_inc, third_party_inc],
    install: false,
)
