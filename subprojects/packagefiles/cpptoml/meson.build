# Meson build wrapper for cpptoml (skystrife/cpptoml).
# Header-only TOML parser. We expose a declare_dependency with the include
# path + a forced <limits> include (cpptoml uses std::numeric_limits without
# including <limits>; breaks on clang 18 / libstdc++ 13 on ARM64 Linux).

project('cpptoml', 'cpp', version: '0.5.0')

cpptoml_dep = declare_dependency(
    include_directories: include_directories('include'),
    compile_args: ['-include', 'limits'],
)
