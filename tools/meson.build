sextant_cli = executable('sextant',
    ['sextant_cli.cpp', 'analyze.cpp'],
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc],
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
    include_directories: [engine_inc, third_party_inc, numkong_inc],
    install: false,
)

pq_explore = executable('pq_explore',
    'pq_explore.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc],
    install: false,
)
