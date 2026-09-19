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
    'test_pq_fastscan.cpp',
    'test_pq4_kernel.cpp',
    'test_pq4_encoding.cpp',
    'test_fbin_source.cpp',
    'test_tree_phase0.cpp',
    'test_tree_phase1.cpp',
    'test_tree_estimator.cpp',
    'test_tree_filter.cpp',
    'test_fsck.cpp',
    'test_tree_insert_delete.cpp',
    'test_parquet_source.cpp',
    'test_vacuum_defrag.cpp',
    'test_quantizer_families.cpp',
    'test_simd_x86.cpp',
    'test_leaf_cache.cpp',
    'test_batch_search.cpp',
    'test_cluster_stage.cpp',
    'test_dots4_kernel.cpp',

]

foreach src : test_sources
    name = src.split('.')[0]
    exe = executable(name,
        src,
        dependencies: [sextant_dep, sextant_sources_dep, gtest_dep, gtest_main_dep],
        include_directories: [test_inc, sources_inc],
        cpp_args: ['-DSEXTANT_SOURCE_DIR="' + meson.source_root() + '"'],
    )
    # Slow suites under the sanitizer build (ASAN+UBSAN with
    # MALLOC_PERTURB_): test_tree_insert_delete needs ~240s and
    # test_quantizer_families ~533s on the x86 dev box (they pass well
    # under 180s in release). Per-test overrides so the full sanitizer
    # suite passes in one pass; everything else keeps the 180s bound
    # that also covers slow ARM hardware.
    slow_timeout = {
        'test_tree_insert_delete': 600,
        'test_quantizer_families': 900,
    }
    test(name, exe, timeout: slow_timeout.get(name, 180),
         env: ['SEXTANT_TEST_DATA=' + meson.current_source_dir() / 'data',
               'SEXTANT_DATASETS=' + meson.current_source_dir() / '../datasets'])
endforeach

# test_capi links the C ABI shim library (sextant_c) on top of the engine.
test_capi_exe = executable('test_capi', 'test_capi.cpp',
    dependencies: [sextant_c_dep, sextant_sources_dep, gtest_dep, gtest_main_dep],
    include_directories: [test_inc, sources_inc],
    cpp_args: ['-DSEXTANT_SOURCE_DIR="' + meson.source_root() + '"'],
)
test('test_capi', test_capi_exe,
     timeout: 300,  # the forced-spill suite includes a 33k-row multi-chunk build
     env: ['SEXTANT_TEST_DATA=' + meson.current_source_dir() / 'data',
           'SEXTANT_DATASETS=' + meson.current_source_dir() / '../datasets'])

# Install test data files (SIFTsmall etc.) so tests find them at runtime.
install_data('data/siftsmall_base.fbin', install_dir: 'sextant-test')
