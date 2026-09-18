#include "tree/scan_pool.hpp"

#include <spdlog/spdlog.h>

#include <thread>

namespace sextant::tree {

namespace {
constexpr uint32_t kMinThreads = 1;

ctpl::thread_pool_tls<ScanWorkerTag>& pool_instance() {
    // Intentionally leaked (see header): no static-destruction ordering
    // hazards, parked threads are free.
    static ctpl::thread_pool_tls<ScanWorkerTag>* pool =
        new ctpl::thread_pool_tls<ScanWorkerTag>(std::thread::hardware_concurrency());
    return *pool;
}
}  // namespace

ctpl::thread_pool_tls<ScanWorkerTag>& scan_pool() {
    return pool_instance();
}

uint32_t scan_pool_threads() {
    return static_cast<uint32_t>(pool_instance().size());
}

uint32_t scan_pool_set_threads(uint32_t n) {
    const uint32_t target =
        n == 0 ? std::max<uint32_t>(kMinThreads, std::thread::hardware_concurrency())
               : std::max<uint32_t>(kMinThreads, n);
    if (target != scan_pool_threads()) {
        pool_instance().resize(target);
        spdlog::info("[sextant] scan pool resized to {} threads", target);
    }
    return target;
}

}  // namespace sextant::tree
