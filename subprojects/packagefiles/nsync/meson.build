# Meson build wrapper for nsync (google/nsync).
#
# nsync's CMakeLists builds nsync_cpp by copying the .c sources into a build/
# subdir and compiling them as C++. Meson's cmake.subproject doesn't handle
# this copy_if_different + LANGUAGE override pattern. So we do it natively:
# use a generator to copy .c → .cpp, then compile as C++.
#
# nsync ships a lowercase `version` file in its root. On case-insensitive macOS,
# this collides with the C++23 system header <version>. We delete it via a
# wrap diff_files patch (subprojects/packagefiles/nsync/delete-version-file.patch)
# rather than mutating the source tree at configure time. The file only contains
# the version string ("1.30.0"), which is hardcoded in project() below and not
# read by any C source.

project('nsync', 'cpp', version: '1.30.0')

nsync_root = meson.current_source_dir()

# --- Sources compiled as C++ (the nsync_cpp equivalent) -----------------------
common_src = [
    'internal/common.c',
    'internal/counter.c',
    'internal/cv.c',
    'internal/debug.c',
    'internal/dll.c',
    'internal/mu.c',
    'internal/mu_wait.c',
    'internal/note.c',
    'internal/once.c',
    'internal/sem_wait.c',
    'internal/time_internal.c',
    'internal/wait.c',
]

host_system = host_machine.system()
host_cpu = host_machine.cpu_family()

# Compiler-specific platform dir (clang or gcc).
cpp = meson.get_compiler('cpp')
if cpp.get_id() == 'clang' or cpp.get_id() == 'apple-clang'
    compiler_dir = 'clang'
elif cpp.get_id() == 'gcc'
    compiler_dir = 'gcc'
else
    compiler_dir = 'gcc'  # fallback
endif

# CPU-specific platform dir.
if host_cpu == 'x86_64'
    cpu_dir = 'x86_64'
elif host_cpu == 'x86'
    cpu_dir = 'x86_32'
elif host_cpu == 'aarch64'
    cpu_dir = 'aarch64'
elif host_cpu == 'arm'
    cpu_dir = 'arm'
else
    cpu_dir = 'x86_64'  # fallback
endif

if host_system == 'darwin'
    os_src_c = [
        'platform/posix/src/per_thread_waiter.c',
        'platform/posix/src/clock_gettime.c',
    ]
    os_src_cc = [
        'platform/c++11/src/yield.cc',
        'platform/c++11/src/time_rep_timespec.cc',
        'platform/c++11/src/nsync_panic.cc',
        'platform/c++11/src/nsync_semaphore_mutex.cc',
    ]
    platform_dirs = ['public', 'internal', 'platform/c++11', 'platform/macos',
                     'platform/posix', 'platform/' + compiler_dir,
                     'platform/' + cpu_dir]
    platform_args = ['-DNSYNC_ATOMIC_CPP11', '-DNSYNC_USE_CPP11_TIMEPOINT', '-D_DARWIN_C_SOURCE']
elif host_system == 'linux'
    os_src_c = [
        'platform/posix/src/per_thread_waiter.c',
        'platform/posix/src/clock_gettime.c',
    ]
    os_src_cc = [
        'platform/c++11/src/yield.cc',
        'platform/c++11/src/time_rep_timespec.cc',
        'platform/c++11/src/nsync_panic.cc',
        'platform/c++11/src/nsync_semaphore_mutex.cc',
    ]
    platform_dirs = ['public', 'internal', 'platform/c++11', 'platform/c++11.futex',
                     'platform/linux', 'platform/posix',
                     'platform/' + compiler_dir, 'platform/' + cpu_dir]
    platform_args = ['-DNSYNC_ATOMIC_CPP11', '-DNSYNC_USE_CPP11_TIMEPOINT']
else
    error('Unsupported host system: @0@'.format(host_system))
endif

# Use -isystem to avoid shadowing system headers.
inc_args = []
foreach dir : platform_dirs
    inc_args += ['-isystem', nsync_root / dir]
endforeach

all_args = inc_args + platform_args

# Copy .c → .cpp so Meson compiles them as C++ (nsync's nsync_cpp pattern).
copy_gen = generator(find_program('python3'),
    output: '@PLAINNAME@.cpp',
    arguments: ['-c', 'import shutil, sys; shutil.copy(sys.argv[1], sys.argv[2])', '@INPUT@', '@OUTPUT@'])

common_src_cpp = copy_gen.process(common_src)
os_src_c_cpp = copy_gen.process(os_src_c)

nsync_lib = static_library('nsync',
    [common_src_cpp, os_src_c_cpp, os_src_cc],
    cpp_args: all_args,
    install: false,
)

# Public consumers need public/ for nsync_cpp.h. Also use -isystem to avoid
# shadowing system headers.
nsync_dep = declare_dependency(
    link_with: nsync_lib,
    compile_args: ['-isystem', nsync_root / 'public'] + platform_args,
)
