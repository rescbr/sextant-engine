# Microbenchmarks and exploratory scripts. Not installed; built on demand
# (ninja -C build bench_preprocess_query). Each is a standalone executable.

bench_preprocess_query = executable('bench_preprocess_query',
    'bench_preprocess_query.cpp',
    dependencies: sextant_dep,
    include_directories: [engine_inc, third_party_inc, numkong_inc],
    install: false,
)
