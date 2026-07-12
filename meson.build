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

# --- Compiler flags -----------------------------------------------------------
warning_flags = ['-Wall', '-Wextra', '-Wpedantic', '-Wno-unused-parameter']
foreach flag : warning_flags
    if cpp.has_argument(flag)
        add_project_arguments(flag, language: 'cpp')
    endif
endforeach

release_flags = ['-O3']
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

# --- Third-party dependencies -------------------------------------------------

# nsync: built as a native Meson subproject (Meson's cmake converter doesn't
# handle nsync's copy_if_different + LANGUAGE CXX trick). See
# subprojects/nsync/meson.build. Symbols are in namespace nsync.
nsync_proj = subproject('nsync')
nsync_dep = nsync_proj.get_variable('nsync_dep')

# spdlog: from WrapDB (header-only mode). meson wrap install spdlog.
spdlog_dep = dependency('spdlog')

# NumKong: header-only, just add include path.
numkong_inc = include_directories('third_party/numkong/include')

# CTPL and cmdline: committed headers, include path only.
third_party_inc = include_directories('third_party')

# --- Engine library -----------------------------------------------------------
engine_inc = include_directories('include')
src_inc = include_directories('src')

engine_sources = [
    'src/storage/direct_io.cpp',
    'src/storage/block_allocator.cpp',
    'src/storage/sharded_lru.cpp',
    'src/quant/pq_quantizer.cpp',
    'src/algo/vamana_core.cpp',
    'src/engine/build.cpp',
    'src/engine/search.cpp',
    'src/logging.cpp',
]

sextant_lib = static_library('sextant',
    engine_sources,
    include_directories: [engine_inc, src_inc, numkong_inc, third_party_inc],
    dependencies: [nsync_dep, spdlog_dep],
    install: false,
)

sextant_dep = declare_dependency(
    link_with: sextant_lib,
    include_directories: [engine_inc, src_inc],
    dependencies: [nsync_dep, spdlog_dep],
)

# --- CLI tool -----------------------------------------------------------------
subdir('tools')

# --- Tests --------------------------------------------------------------------
subdir('test')
