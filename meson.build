project('sextant-engine', 'cpp',
    version: '0.1.0',
    default_options: [
        'cpp_std=c++23',
        'default_library=static',
        'warning_level=2',
        'b_ndebug=if-release',
        'buildtype=release',
    ],
)

cpp = meson.get_compiler('cpp')

# --- Build version (git commit + dirty flag) ----------------------------------
# Captured at meson setup / reconfigure time and baked into build/version.h as
# SEXTANT_GIT_SHA / SEXTANT_GIT_DIRTY. Lets `sextant --version` and `sextant_bench
# --version` report exactly which source produced the binary — critical for
# matching a dev-VM's prebuilt binary back to a local change set.
#
# The SHA shortens to 7 chars (git's default abbreviated length). "dirty" is set
# when the working tree or index has uncommitted changes; binaries built from a
# dirty tree are flagged so a stale/dirty build is never mistaken for a clean one.
git = find_program('git', required: false)
if git.found()
    git_sha = run_command(git, 'rev-parse', '--short=7', 'HEAD',
                          check: false, capture: true).stdout().strip()
    # `git status --porcelain` prints one line per uncommitted entry. We use
    # --untracked-files=no so the dirty flag reflects only tracked-source
    # changes — an untracked build/ dir (or scratch files) would otherwise make
    # every in-tree build look dirty. Any non-empty output means dirty.
    git_dirty_out = run_command(git, 'status', '--porcelain', '--untracked-files=no',
                                check: false, capture: true).stdout().strip()
    git_dirty = (git_dirty_out != '') ? '1' : '0'
    if git_sha == ''
        git_sha = 'unknown'
    endif
else
    git_sha = 'unknown'
    git_dirty = '0'
endif
version_data = configuration_data()
version_data.set('SEXTANT_VERSION', meson.project_version())
version_data.set('SEXTANT_GIT_SHA', git_sha)
version_data.set('SEXTANT_GIT_DIRTY', git_dirty)
# Generated into the build dir (not committed); included via version_inc below.
version_h = configure_file(
    input: 'version.h.in',
    output: 'version.h',
    configuration: version_data,
)
version_inc = include_directories('.')

# --- Compiler flags -----------------------------------------------------------
# Warnings are errors for engine code. Third-party headers (numkong, ctpl,
# cmdline, nsync) are added as -isystem below so their warnings are suppressed
# and -Werror doesn't fire on code we don't own. Each subproject is its own
# meson project (own project() call), so these add_project_arguments don't leak
# into nsync/spdlog/cpptoml/cpptrace/gtest builds.
warning_flags = ['-Wall', '-Wextra', '-Wpedantic', '-Wno-unused-parameter', '-Werror']
foreach flag : warning_flags
    if cpp.has_argument(flag)
        add_project_arguments(flag, language: 'cpp')
    endif
endforeach

# Release: -O3 -g (debug symbols are free — they don't affect codegen, only
# the symbol table, and they make profiling/lldb usable in production builds).
# Debug: -O0 -g3 + sanitizers (for catching memory errors in tests).
release_flags = ['-O3', '-g']
debug_flags = ['-O0', '-g3', '-fsanitize=address,undefined', '-fno-omit-frame-pointer']

is_debug = get_option('debug')
if is_debug
    foreach flag : debug_flags
        if cpp.has_argument(flag)
            add_project_arguments(flag, language: 'cpp')
        endif
    endforeach
else
    foreach flag : release_flags
        if cpp.has_argument(flag)
            add_project_arguments(flag, language: 'cpp')
        endif
    endforeach
    march_native = cpp.has_argument('-march=native')
    if march_native
        add_project_arguments('-march=native', language: 'cpp')
    endif
endif

# --- FP16 support (required for FP16 prune) -----------------------------------
# Sextant requires hardware FP16 support for the build-time prune occlusion check.
# On ARM: NEON (any AArch64). On x86: F16C (Haswell 2013+).
#
# We DETECT support but do NOT force-add -mf16c. If the user passes -march=native
# or -mcpu=neoverse-v2 (which enable F16C implicitly on capable hardware), the
# detection passes. Forcing -mf16c would conflict with architecture-specific flags.
fp16_ok = cpp.compiles('''
    #if defined(__x86_64__) || defined(__i386__)
    #include <immintrin.h>
    int main() {
        __m128i h = _mm_setzero_si128();
        __m256 f = _mm256_cvtph_ps(h);
        (void)f;
        return 0;
    }
    #elif defined(__aarch64__) || defined(__ARM_NEON)
    #include <arm_neon.h>
    int main() {
        float16x4_t h = vdup_n_f16(0);
        float32x4_t f = vcvt_f32_f16(h);
        (void)f;
        return 0;
    }
    #else
    #error unsupported architecture
    #endif
''')
if not fp16_ok
    error('Sextant requires hardware FP16 support (NEON on ARM, F16C on x86). ' +
          'Ensure -march=native or -mf16c is enabled, or build on AArch64.')
endif

# --- Third-party dependencies -------------------------------------------------

# nsync: built as a native Meson subproject (Meson's cmake converter doesn't
# handle nsync's copy_if_different + LANGUAGE CXX trick). See
# subprojects/nsync/meson.build. Symbols are in namespace nsync.
nsync_proj = subproject('nsync')
nsync_dep = nsync_proj.get_variable('nsync_dep')

# spdlog: from WrapDB (header-only mode). meson wrap install spdlog.
spdlog_dep = dependency('spdlog')

# NumKong: header-only. Only used by scripts/bench_preprocess_query.cpp
# (an old microbench). NOT used by the engine itself. Add the include path
# only to scripts that actually need it. is_system silences numkong's own
# -Wmissing-braces noise under -Werror.
numkong_inc = include_directories('third_party/numkong/include', is_system: true)

# CTPL and cmdline: committed headers, include path only. is_system silences
# any third-party warnings under -Werror.
third_party_inc = include_directories('third_party', is_system: true)

# cpptoml: header-only TOML parser (for tree manifest serialization).
cpptoml_proj = subproject('cpptoml')
cpptoml_dep = cpptoml_proj.get_variable('cpptoml_dep')

# cpptrace: stack traces on crash (signal handler in main).
cpptrace_proj = subproject('cpptrace')
cpptrace_dep = cpptrace_proj.get_variable('cpptrace_dep')

# --- Engine library -----------------------------------------------------------
engine_inc = include_directories('include')
src_inc = include_directories('src')

engine_sources = [
    'src/storage/direct_io.cpp',
    'src/storage/buffered_io.cpp',
    'src/storage/block_allocator.cpp',
    'src/storage/block_buffer_pool.cpp',
    'src/storage/block_cache.cpp',
    'src/storage/node_store.cpp',
    'src/storage/memgraph.cpp',
    'src/quant/pq_quantizer.cpp',
    'src/quant/anisotropic_pq_quantizer.cpp',
    'src/quant/product_residual_quantizer.cpp',
    'src/quant/rabitq_quantizer.cpp',
    'src/algo/vamana_core.cpp',
    'src/engine/resolve_params.cpp',
    'src/engine/index.cpp',
    'src/storage/code_stream.cpp',
    'src/engine/probe.cpp',
    'src/engine/builder.cpp',
    'src/engine/searcher.cpp',
    'src/engine/estimator.cpp',
    'src/engine/fbin_source.cpp',
    'src/engine/mem_source.cpp',
    'src/engine/partition.cpp',
    'src/engine/manifest_io.cpp',
    # Hierarchical IVF tree (Phase 0: infrastructure)
    'src/tree/page_file.cpp',
    'src/tree/page_allocator.cpp',
    'src/tree/superblock.cpp',
    'src/tree/tree_manifest.cpp',
    'src/tree/ivf_tree_index.cpp',
    'src/tree/tree_estimator.cpp',
    'src/tree/filter_scan.cpp',
    'src/tree/cardinality.cpp',
    'src/tree/filter_data_io.cpp',
    'src/tree/fsck.cpp',
    'src/logging.cpp',
    'src/system.cpp',
]

sextant_lib = static_library('sextant',
    engine_sources,
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: [nsync_dep, spdlog_dep, cpptoml_dep, cpptrace_dep],
    install: false,
)

sextant_dep = declare_dependency(
    link_with: sextant_lib,
    include_directories: [engine_inc, src_inc],
    dependencies: [nsync_dep, spdlog_dep, cpptoml_dep, cpptrace_dep],
)

# --- CLI tool -----------------------------------------------------------------
subdir('tools')

# --- Tests --------------------------------------------------------------------
subdir('test')

# --- Microbenchmarks / exploratory scripts -----------------------------------
subdir('scripts')
