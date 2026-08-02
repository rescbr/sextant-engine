# Meson build wrapper for cpptrace (jeremy-rifkin/cpptrace).
#
# We compile cpptrace natively (not via cmake.subproject) to avoid its
# FetchContent of libdwarf + zstd at configure time. Instead we select the
# "addr2line" symbol backend (external tool: addr2line on Linux, atos on
# macOS) which needs zero bundled dependencies.
#
# Backends selected:
#   Unwind:    execinfo (macOS + Linux glibc) — `<execinfo.h>` backtrace().
#   Symbols:   addr2line (external tool invocation).
#   Demangle:  cxxabi (libc++abi / libstdc++ __cxa_demangle).
#
# These are the most portable backends with zero external link deps.

project('cpptrace', 'cpp', version: '1.0.4')

cpptrace_root = meson.current_source_dir()

src_files = [
    'src/cpptrace.cpp',
    'src/ctrace.cpp',
    'src/exceptions.cpp',
    'src/formatting.cpp',
    'src/from_current.cpp',
    'src/logging.cpp',
    'src/options.cpp',
    'src/prune_symbol.cpp',
    'src/utils.cpp',
    # demangle
    'src/demangle/demangle_with_cxxabi.cpp',
    # symbols
    'src/symbols/symbols_core.cpp',
    'src/symbols/symbols_with_addr2line.cpp',
    # unwind
    'src/unwind/unwind_with_execinfo.cpp',
    'src/unwind/unwind_with_nothing.cpp',
    # snippets
    'src/snippets/snippet.cpp',
    # jit (JIT-compiled code object tracking)
    'src/jit/jit_objects.cpp',
    # binary (object file parsing for snippets)
    'src/binary/elf.cpp',
    'src/binary/mach-o.cpp',
    'src/binary/module_base.cpp',
    'src/binary/object.cpp',
    'src/binary/pe.cpp',
    'src/binary/safe_dl.cpp',
    # platform
    'src/platform/dbghelp_utils.cpp',
    'src/platform/memory_mapping.cpp',
    # utils
    'src/utils/error.cpp',
    'src/utils/microfmt.cpp',
    'src/utils/replace_all.cpp',
    'src/utils/string_view.cpp',
    'src/utils/utils.cpp',
    # utils/io (file abstraction used by binary/mach-o, binary/elf for snippets)
    'src/utils/io/file.cpp',
    'src/utils/io/memory_file_view.cpp',
]

# cpptrace's internal sources use relative includes like "utils/common.hpp"
# and "platform/platform.hpp" relative to src/. We must add both include/ and
# src/ to the include path.
cpptrace_inc = include_directories('include', 'src')

# Backend selection defines. cpptrace uses preprocessor conditionals to
# choose unwind/symbol/demangle backends. Without these, the dispatch
# functions compile to empty stubs.
#
#   Unwind:   CPPTRACE_UNWIND_WITH_EXECINIT (backtrace() from <execinfo.h>)
#             Also need CPPTRACE_UNWIND_WITH_NOTHING for the fallback path.
#   Symbols:  CPPTRACE_GET_SYMBOLS_WITH_ADDR2LINE (external addr2line/atos)
#   Demangle: CPPTRACE_DEMANGLE_WITH_CXXABI (__cxa_demangle)
backend_defines = [
    '-DCPPTRACE_STATIC_DEFINE',
    '-DCPPTRACE_UNWIND_WITH_EXECINFO',
    '-DCPPTRACE_UNWIND_WITH_NOTHING',
    '-DCPPTRACE_GET_SYMBOLS_WITH_ADDR2LINE',
    '-DCPPTRACE_DEMANGLE_WITH_CXXABI',
    # Search PATH for addr2line (Linux) / atos (macOS) at runtime rather
    # than hardcoding a binary path.
    '-DCPPTRACE_ADDR2LINE_SEARCH_SYSTEM_PATH',
]

cpptrace_lib = static_library('cpptrace',
    src_files,
    include_directories: cpptrace_inc,
    install: false,
    cpp_args: backend_defines,
    # cpptrace targets C++17; our project sets C++23 at the top level.
    # We do NOT override the std here (meson's cpp_std belongs to project(),
    # not static_library). cpptrace compiles fine under C++23.
)

cpptrace_dep = declare_dependency(
    link_with: cpptrace_lib,
    include_directories: cpptrace_inc,
)
