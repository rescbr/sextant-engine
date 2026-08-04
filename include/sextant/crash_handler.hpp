#pragma once

/// @file crash_handler.hpp
/// Three-layer crash debugging infrastructure:
///
/// 1. **Throw-site trace interception** (`__cxa_throw`):
///    Intercepts every C++ throw (including STL exceptions) and captures the
///    stack trace BEFORE stack unwinding. Stored in a thread-local. Any catch
///    block can read it via `sextant::last_throw_trace()`. This is the Unix
///    equivalent of Windows SEH — the throw-site frames are captured while
///    they still exist on the stack.
///
/// 2. **Signal handler** (SIGSEGV/SIGABRT/etc):
///    Prints a cpptrace stack trace, restores the default handler, and
///    re-raises. The OS produces a core dump for post-mortem in lldb/gdb.
///    On Linux, core dumps are filtered to exclude file-backed mmap pages
///    (the tree index) — keeping them small.
///
/// 3. **Terminate handler** (uncaught exceptions):
///    Prints the exception message + the intercepted throw-site trace, then
///    aborts for a core dump.
///
/// Call `install_crash_handler()` once at program start.
/// Exceptions in CLI catch blocks should call `__builtin_trap()` after
/// printing the trace — this drops into lldb (if attached) or produces a
/// core dump (if not).
///
/// Uses cpptrace with the execinfo unwind backend + addr2line symbol resolver.

#include <cpptrace/cpptrace.hpp>

#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <dlfcn.h>
#endif

#ifdef __APPLE__
#include <dlfcn.h>
#endif

namespace sextant {

// ---------------------------------------------------------------------------
// Thread-local throw-site trace (populated by __cxa_throw interception).
// ---------------------------------------------------------------------------
inline thread_local std::optional<cpptrace::stacktrace>
    tl_throw_trace;

/// Read the trace captured at the most recent throw site on this thread.
/// Empty if no throw occurred or interception is disabled (release builds).
inline const std::optional<cpptrace::stacktrace>&
last_throw_trace() {
    return tl_throw_trace;
}

// ---------------------------------------------------------------------------
// __cxa_throw interception — capture throw-site trace before unwinding.
// ---------------------------------------------------------------------------
// This runs BEFORE stack unwinding begins, so the full throw-site call
// stack is still alive. We store it in a thread-local; any catch block
// on the same thread can retrieve it.
//
// ---------------------------------------------------------------------------
// __cxa_throw interception — capture throw-site trace before unwinding.
// ---------------------------------------------------------------------------
// This runs BEFORE stack unwinding begins, so the full throw-site call
// stack is still alive. We store it in a thread-local; any catch block
// on the same thread can retrieve it.
//
// Enabled unconditionally (debug + release). The only throws in the engine
// are fatal errors (I/O failure, corrupt index, invalid params) or the
// one std::stod parse-fallback outside the hot path — never hot-loop flow
// control. The ~50-100μs overhead per throw is negligible at this frequency.
extern "C" __attribute__((visibility("default"))) inline
void __cxa_throw(void* thrown_exception, std::type_info* tinfo,
                 void (*dest)(void*)) {
    // Capture the throw-site stack trace.
    tl_throw_trace = cpptrace::generate_trace();

    // Forward to the real __cxa_throw.
    static void (*real_cxa_throw)(void*, std::type_info*, void(*)(void*))
        = reinterpret_cast<void(*)(void*, std::type_info*, void(*)(void*))>(
            dlsym(RTLD_NEXT, "__cxa_throw"));
    if (real_cxa_throw) {
        real_cxa_throw(thrown_exception, tinfo, dest);
    } else {
        // Fallback: shouldn't happen, but if dlsym fails we must still throw.
        std::fprintf(stderr,
            "sextant: __cxa_throw interception failed (dlsym returned null)\n");
        std::abort();
    }
    __builtin_unreachable();  // __cxa_throw never returns
}

// ---------------------------------------------------------------------------
// Signal handler — print trace + core dump.
// ---------------------------------------------------------------------------
inline void crash_signal_handler(int sig) {
    const char* name = "unknown";
    switch (sig) {
        case SIGSEGV: name = "SIGSEGV"; break;
        case SIGABRT: name = "SIGABRT"; break;
        case SIGILL:  name = "SIGILL";  break;
        case SIGFPE:  name = "SIGFPE";  break;
        case SIGBUS:  name = "SIGBUS";  break;
    }
    std::fprintf(stderr,
        "\n=== Sextant crashed (signal %d: %s) ===\n", sig, name);
    cpptrace::generate_trace();
    std::fflush(stderr);
    // Restore default handler and re-raise — produces a core dump.
    std::signal(sig, SIG_DFL);
    std::raise(sig);
}

// ---------------------------------------------------------------------------
// Install all crash handlers.
// ---------------------------------------------------------------------------
inline void install_crash_handler() {
#ifdef __linux__
    // Exclude file-backed mmap'd pages from core dumps. We only need
    // anonymous memory (heap, stacks) for post-mortem debugging.
    // 0x3 = ANON_PRIVATE | ANON_SHARED (bits 0-1).
    int cdf = ::open("/proc/self/coredump_filter", O_WRONLY);
    if (cdf >= 0) {
        const char filter[] = "0x3";
        ::write(cdf, filter, sizeof(filter) - 1);
        ::close(cdf);
    }
#endif

    std::signal(SIGSEGV, crash_signal_handler);
    std::signal(SIGABRT, crash_signal_handler);
    std::signal(SIGILL,  crash_signal_handler);
    std::signal(SIGFPE,  crash_signal_handler);
    std::signal(SIGBUS,  crash_signal_handler);

    // Terminate handler — fires for uncaught exceptions.
    std::set_terminate([]() {
        std::fprintf(stderr,
            "\n=== Sextant terminated (uncaught exception) ===\n");
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Exception: %s\n", e.what());
        } catch (...) {
            std::fprintf(stderr, "Exception: (unknown type)\n");
        }
        // Print the throw-site trace (captured by __cxa_throw interception).
        if (tl_throw_trace) {
            std::fprintf(stderr, "Throw-site trace:\n%s\n",
                         tl_throw_trace->to_string().c_str());
        }
        std::fflush(stderr);
        std::abort();  // core dump
    });
}

}  // namespace sextant
