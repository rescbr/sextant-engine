#include "sextant/logging.hpp"

#include <spdlog/spdlog.h>

namespace sextant {

void init_logging(LogLevel level) {
    static bool initialized = false;
    if (initialized) {
        set_log_level(level);
        return;
    }
    initialized = true;

    spdlog::set_pattern("[sextant] [%^%l%$] %v");
    set_log_level(level);
}

void set_log_level(LogLevel level) {
    switch (level) {
        case LogLevel::Debug:
            spdlog::set_level(spdlog::level::debug);
            break;
        case LogLevel::Info:
            spdlog::set_level(spdlog::level::info);
            break;
        case LogLevel::Warn:
            spdlog::set_level(spdlog::level::warn);
            break;
        case LogLevel::Error:
            spdlog::set_level(spdlog::level::err);
            break;
    }
}

}  // namespace sextant
