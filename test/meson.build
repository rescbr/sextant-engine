# gtest from WrapDB (meson wrap install gtest).
gtest_dep = dependency('gtest')
gtest_main_dep = dependency('gtest_main')

# third_party as is_system so its headers (ctpl/cmdline) don't trip -Werror.
test_inc = include_directories('../include', '../src', '../third_party',
    is_system: true)

test_sources = [
    'test_storage.cpp',
    'test_block_cache.cpp',
    'test_wtinylfu.cpp',
    'test_pq_quantizer.cpp',
    'test_prq_quantizer.cpp',
    'test_rabitq_quantizer.cpp',
    'test_pq_fastscan.cpp',
    'test_pq4_kernel.cpp',
    'test_pq4_encoding.cpp',
    'test_vamana_core.cpp',
    'test_fbin_source.cpp',
    'test_engine.cpp',
    'test_paged_search.cpp',
    'test_tl_cache.cpp',
    'test_memgraph.cpp',
    'test_cli.cpp',
    'test_partition.cpp',
    'test_tree_phase0.cpp',
    'test_tree_phase1.cpp',
    'test_tree_estimator.cpp',
    'test_tree_filter.cpp',
    'test_fsck.cpp',
    'test_tree_insert_delete.cpp',
    'test_parquet_source.cpp',
]

foreach src : test_sources
    name = src.split('.')[0]
    exe = executable(name,
        src,
        dependencies: [sextant_dep, sextant_sources_dep, gtest_dep, gtest_main_dep],
        include_directories: [test_inc, sources_inc],
        cpp_args: ['-DSEXTANT_SOURCE_DIR="' + meson.source_root() + '"'],
    )
    test(name, exe, env: ['SEXTANT_TEST_DATA=' + meson.current_source_dir() / 'data'])
endforeach

# Install test data files (SIFTsmall etc.) so tests find them at runtime.
install_data('data/siftsmall_base.fbin', install_dir: 'sextant-test')
