#include "sextant/system.hpp"

#include <unistd.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
#endif

namespace sextant {

uint64_t physical_ram_bytes() {
#if defined(__APPLE__)
    uint64_t membytes = 0;
    size_t len = sizeof(membytes);
    if (::sysctlbyname("hw.memsize", &membytes, &len, nullptr, 0) == 0 &&
        membytes > 0) {
        return membytes;
    }
#elif defined(__linux__)
    // /proc/meminfo is more reliable than sysconf on Linux, but sysconf is a
    // fine portable fallback. Try sysconf first (no file parse needed).
#endif
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGE_SIZE);
    if (pages > 0 && page_size > 0) {
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    }
    return 0;
}

}  // namespace sextant
