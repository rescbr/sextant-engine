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

# Force clang: the engine does not build with GCC on x86 (float16_t is
# __fp16, clang-only syntax there; ARM GCC also diverges on NEON typing).
# Fail FAST with a clear message instead of a confusing types.hpp error.
# To set up a build dir: `meson setup build --native-file clang.ini`
# (clang.ini pins c/cpp to clang/clang++).
if host_machine.cpu_family() == 'x86' or host_machine.cpu_family() == 'x86_64'
    if meson.get_compiler('cpp').get_id() != 'clang'
        error('\n' +
              'sextant-engine requires clang on x86 (float16_t = __fp16).\n' +
              'Your compiler is GCC. Configure with:\n' +
              '  meson setup <builddir> --native-file clang.ini\n' +
              'or export CC=clang CXX=clang++ before running meson setup.')
    endif
endif

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

# --- SIMD target detection ---------------------------------------------------
# Determines which scalar-dot kernel tier is compiled in. Stored as
# SEXTANT_SIMD_TARGET in version.h alongside the git SHA.
# Based on compiler-predefined __ARM_FEATURE_* / __AVX* macros (set by
# -march). Probes always use -march=native to match the build flags below.
# On M4: -march=native → neon+i8mm (+ sme via __ARM_FEATURE_SME).
# On c4a: -march=armv8-a+sve2 → sve2.
# On Zen5 (9950X): -march=native → avx512 (+vnni+vbmi, the x86 analog of
# SVE2 SDOT gather-dot for the i8/u4 scan kernels).
if cpp.compiles('''
    #ifndef __ARM_FEATURE_SVE
    #error no sve
    #endif
    int main() { return 0; }
''', args: ['-march=native'])
    simd_target = 'sve2'
elif cpp.compiles('''
    #if !(defined(__AVX512F__) && defined(__AVX512VNNI__) && defined(__AVX512VBMI__))
    #error no avx512 suite
    #endif
    int main() { return 0; }
''', args: ['-march=native'])
    simd_target = 'avx512'
elif cpp.compiles('''
    #ifndef __AVX2__
    #error no avx2
    #endif
    int main() { return 0; }
''', args: ['-march=native'])
    simd_target = 'avx2'
elif cpp.compiles('''
    #ifndef __ARM_FEATURE_MATMUL_INT8
    #error no i8mm
    #endif
    int main() { return 0; }
''', args: ['-march=native'])
    simd_target = 'neon+i8mm'
else
    simd_target = 'neon'
endif
message('SIMD target: ' + simd_target)

# Expose SVE2 availability to simd_kernels.hpp as a compile definition so the
# scalar_dot_u4_sve2 dispatcher calls the real SVE2 impl when the TU is built.
# Only on real SVE2 hardware (Neoverse-V2 / c4a). On M4 (neon+i8mm) the
# dispatcher falls back to the NEON kernel.
if simd_target == 'sve2'
    add_project_arguments('-DSEXTANT_HAS_SVE2_KERNEL=1', language: 'cpp')
endif

version_data = configuration_data()
version_data.set('SEXTANT_VERSION', meson.project_version())
version_data.set('SEXTANT_GIT_SHA', git_sha)
version_data.set('SEXTANT_GIT_DIRTY', git_dirty)
version_data.set('SEXTANT_SIMD_TARGET', simd_target)
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
    # Debug builds also get -march=native (user call): sanitizers don't change
    # the ISA requirement, and x86 needs F16C for util/fp16.hpp regardless.
    march_native = cpp.has_argument('-march=native')
    if march_native
        add_project_arguments('-march=native', language: 'cpp')
    endif
    # Sanitizer flags must ALSO be on the link line (Gentoo ld.bfd does not
    # pull the runtime in otherwise; the macOS linker did).
    foreach flag : debug_flags
        if cpp.has_link_argument(flag)
            add_project_link_arguments(flag, language: 'cpp')
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
# Note the probe uses -march=native to match the build flags: on x86 the
# compiler baseline has no F16C, so probing without it fails on every
# Haswell+ machine (caught on the 9950X port).
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
''', args: ['-march=native'])
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

# carquet: pure-C Parquet reader/writer (for ParquetSource).
carquet_proj = subproject('carquet')
carquet_dep = carquet_proj.get_variable('carquet_dep')

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
    'src/quant/scalar_lloydmax_quantizer.cpp',
    'src/algo/vamana_core.cpp',
    'src/engine/resolve_params.cpp',
    'src/engine/index.cpp',
    'src/storage/code_stream.cpp',
    'src/engine/probe.cpp',
    'src/engine/builder.cpp',
    'src/engine/searcher.cpp',
    'src/engine/estimator.cpp',
    'src/engine/partition.cpp',
    'src/engine/manifest_io.cpp',
    # Hierarchical IVF tree (Phase 0: infrastructure)
    'src/tree/page_file.cpp',
    'src/tree/page_allocator.cpp',
    'src/tree/superblock.cpp',
    'src/tree/tree_manifest.cpp',
    'src/tree/ivf_tree_index.cpp',
    'src/tree/leaf_extent_cache.cpp',
    'src/tree/batch_scheduler.cpp',
    'src/tree/coders/coder_factory.cpp',
    'src/tree/coders/global_pq_coder.cpp',
    'src/tree/coders/local_pq_coder.cpp',
    'src/tree/coders/scalar_lm_coder.cpp',
    'src/tree/coders/local_scalar_coder.cpp',
    'src/tree/tree_estimator.cpp',
    'src/tree/filter_scan.cpp',
    'src/tree/cardinality.cpp',
    'src/tree/plane.cpp',
    'src/tree/filter_data_io.cpp',
    'src/tree/fsck.cpp',
    'src/util/engine_trace.cpp',
    'src/logging.cpp',
    'src/system.cpp',
]

# --- SVE2 decode-dot kernel (separate compilation unit) -----------------------
# arm_sve.h must be included at file scope (clang 18 fails on function-scope
# SVE intrinsics), and the TU needs -march=armv8-a+sve2 while the rest of the
# engine builds with -march=native. So we compile it as its own static_library
# with the SVE2 flag, then fold the object into sextant_lib via extract_objects.
# Only built when SVE2 is detected; on non-SVE2 hardware (Apple M4) this TU is
# skipped entirely and the dispatcher falls back to NEON.
# Pattern follows subprojects/carquet/meson.build (per-ISA TU + extract_objects).
sve2_objects = []
if simd_target == 'sve2'
    sve2_tu = static_library('sextant_sve2_tu',
        'src/simd/sve2_kernels.cpp',
        cpp_args: ['-march=armv8-a+sve2', '-fno-lto'],
        include_directories: [engine_inc, src_inc],
        install: false,
    )
    sve2_objects = sve2_tu.extract_objects('src/simd/sve2_kernels.cpp')
endif

sextant_lib = static_library('sextant',
    engine_sources,
    objects: sve2_objects,
    include_directories: [engine_inc, src_inc, third_party_inc],
    dependencies: [nsync_dep, spdlog_dep, cpptoml_dep, cpptrace_dep],
    install: false,
)

sextant_dep = declare_dependency(
    link_with: sextant_lib,
    include_directories: [engine_inc, src_inc],
    dependencies: [nsync_dep, spdlog_dep, cpptoml_dep, cpptrace_dep],
)

# --- Source adapters (FbinSource, MemSource) ---------------------------------
# These are NOT part of libsextant — they're file-format adapters used by the
# CLI and tests. The engine depends only on the VectorSource interface.
sources_inc = include_directories('sources')

sextant_sources_lib = static_library('sextant_sources',
    ['sources/fbin_source.cpp', 'sources/mem_source.cpp',
     'sources/parquet_source.cpp',
     'sources/parquet_glob_source.cpp'],
    include_directories: [sources_inc, engine_inc, src_inc, third_party_inc],
    dependencies: [sextant_dep, carquet_dep],
    install: false,
)

sextant_sources_dep = declare_dependency(
    link_with: sextant_sources_lib,
    include_directories: sources_inc,
    dependencies: [sextant_dep, carquet_dep],
)

# --- C ABI shim (task 5: external benchmark harnesses) ------------------------
# Header-only install surface: include/sextant/sextant_c.h. Links the engine
# plus the fbin source adapter so a harness can build straight from .fbin.
sextant_c_lib = static_library('sextant_c',
    ['src/capi/sextant_c.cpp'],
    include_directories: [engine_inc, src_inc, sources_inc, third_party_inc],
    dependencies: [sextant_dep, sextant_sources_dep],
    install: false,
)

# Shared variant for external harnesses (task 5: Rust FFI links one .so
# instead of a thin-archive merge). -Dcapi_shared=true.
if get_option('capi_shared')
    sextant_c_shared = shared_library('sextant_c',
        ['src/capi/sextant_c.cpp'],
        include_directories: [engine_inc, src_inc, sources_inc, third_party_inc],
        dependencies: [sextant_dep, sextant_sources_dep],
        install: false,
    )
endif

# --- CLI tool -----------------------------------------------------------------
subdir('tools')

# --- Tests --------------------------------------------------------------------
subdir('test')

# --- Microbenchmarks / exploratory scripts -----------------------------------
subdir('scripts')
