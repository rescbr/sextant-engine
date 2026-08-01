#pragma once

/// @file crash_handler.hpp
/// Signal handler that prints a stack trace on crash.
///
/// Call `install_crash_handler()` once at program start (before any threads
/// that might crash). On SIGSEGV/SIGABRT/SIGILL/SIGFPE/SIGBUS, it prints a
/// stack trace to stderr and exits.
///
/// Uses cpptrace with the execinfo unwind backend + addr2line symbol resolver.

#include <cpptrace/cpptrace.hpp>

#include <csignal>
#include <cstdio>

namespace sextant {

inline void crash_signal_handler(int sig) {
    const char* name = "unknown";
    switch (sig) {
        case SIGSEGV: name = "SIGSEGV"; break;
        case SIGABRT: name = "SIGABRT"; break;
        case SIGILL:  name = "SIGILL";  break;
        case SIGFPE:  name = "SIGFPE";  break;
        case SIGBUS:  name = "SIGBUS";  break;
    }
    std::fprintf(stderr, "\n=== Sextant crashed (signal %d: %s) ===\n", sig, name);
    cpptrace::generate_trace();
    std::_Exit(128 + sig);
}

inline void install_crash_handler() {
    std::signal(SIGSEGV, crash_signal_handler);
    std::signal(SIGABRT, crash_signal_handler);
    std::signal(SIGILL,  crash_signal_handler);
    std::signal(SIGFPE,  crash_signal_handler);
    std::signal(SIGBUS,  crash_signal_handler);

    // Also install a terminate handler for uncaught C++ exceptions.
    std::set_terminate([]() {
        std::fprintf(stderr, "\n=== Sextant terminated (uncaught exception) ===\n");
        try {
            std::rethrow_exception(std::current_exception());
        } catch (const std::exception& e) {
            std::fprintf(stderr, "Exception: %s\n", e.what());
        }
        cpptrace::generate_trace();
        std::_Exit(1);
    });
}

}  // namespace sextant
