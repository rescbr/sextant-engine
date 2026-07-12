# gtest from WrapDB (meson wrap install gtest).
gtest_dep = dependency('gtest')
gtest_main_dep = dependency('gtest_main')

test_inc = include_directories('../include', '../src', '../third_party')

test_sources = [
    'test_storage.cpp',
    'test_block_cache.cpp',
    'test_wtinylfu.cpp',
    'test_pq_quantizer.cpp',
    'test_vamana_core.cpp',
    'test_fbin_source.cpp',
    'test_engine.cpp',
    'test_paged_search.cpp',
    'test_tl_cache.cpp',
    'test_memgraph.cpp',
    'test_cli.cpp',
    'test_partition.cpp',
]

foreach src : test_sources
    name = src.split('.')[0]
    exe = executable(name,
        src,
        dependencies: [sextant_dep, gtest_dep, gtest_main_dep],
        include_directories: test_inc,
    )
    test(name, exe)
endforeach
