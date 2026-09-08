#pragma once

/// @file sync.hpp
/// Thin C++ RAII wrappers over nsync reader-writer locks.
///
/// nsync provides `nsync_mu` — a reader-writer lock as fast as a mutex, with
/// native futex (Linux) and ulock/GCD (macOS) support. When nsync's headers
/// are included from C++, all symbols land in `namespace nsync` via the
/// NSYNC_CPP_START_/NSYNC_CPP_END_ macros.
///
/// `Mutex` encapsulates an `nsync_mu` (zero-initialized, matching NSYNC_MU_INIT)
/// and exposes both exclusive and shared lock/unlock methods. The scoped guard
/// classes (`ScopedReadLock`, `ScopedWriteLock`) are the primary interface —
/// all locking should go through them, never raw nsync calls.

#include <nsync_cpp.h>
#include <nsync.h>

#include <cstring>

namespace sextant {

/// Owning wrapper around an nsync reader-writer lock.
/// Default-constructed state is zero-initialized, matching NSYNC_MU_INIT.
/// Non-copyable, non-movable (contains an atomics-backed struct).
class Mutex {
public:
    Mutex() { std::memset(&mu_, 0, sizeof(mu_)); }
    ~Mutex() = default;

    Mutex(const Mutex&) = delete;
    Mutex& operator=(const Mutex&) = delete;
    Mutex(Mutex&&) = delete;
    Mutex& operator=(Mutex&&) = delete;

    /// Exclusive (write) lock.
    void lock() { nsync::nsync_mu_lock(&mu_); }
    void unlock() { nsync::nsync_mu_unlock(&mu_); }
    /// Non-blocking try-acquire of the exclusive lock.
    bool try_lock() { return nsync::nsync_mu_trylock(&mu_) != 0; }

    /// Shared (read) lock.
    void lock_shared() { nsync::nsync_mu_rlock(&mu_); }
    void unlock_shared() { nsync::nsync_mu_runlock(&mu_); }

    /// Underlying nsync_mu (for use by the scoped guard classes or
    /// nsync condition variables).
    nsync::nsync_mu* native_handle() { return &mu_; }

private:
    nsync::nsync_mu mu_;
};

/// RAII guard for a shared (read) lock.
class ScopedReadLock {
public:
    explicit ScopedReadLock(nsync::nsync_mu* mu) : mu_(mu) {
        nsync::nsync_mu_rlock(mu_);
    }
    explicit ScopedReadLock(Mutex& m) : mu_(m.native_handle()) {
        nsync::nsync_mu_rlock(mu_);
    }
    ~ScopedReadLock() { nsync::nsync_mu_runlock(mu_); }

    ScopedReadLock(const ScopedReadLock&) = delete;
    ScopedReadLock& operator=(const ScopedReadLock&) = delete;

private:
    nsync::nsync_mu* mu_;
};

/// RAII guard for an exclusive (write) lock.
class ScopedWriteLock {
public:
    explicit ScopedWriteLock(nsync::nsync_mu* mu) : mu_(mu) {
        nsync::nsync_mu_lock(mu_);
    }
    explicit ScopedWriteLock(Mutex& m) : mu_(m.native_handle()) {
        nsync::nsync_mu_lock(mu_);
    }
    ~ScopedWriteLock() { nsync::nsync_mu_unlock(mu_); }

    ScopedWriteLock(const ScopedWriteLock&) = delete;
    ScopedWriteLock& operator=(const ScopedWriteLock&) = delete;

private:
    nsync::nsync_mu* mu_;
};

}  // namespace sextant
