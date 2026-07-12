// sextant CLI — stub.
// Full implementation with cmdline.h dispatched separately.

#include "sextant/logging.hpp"

#include <spdlog/spdlog.h>

#include <iostream>

int main(int argc, char* argv[]) {
    sextant::init_logging();

    if (argc < 2) {
        std::cerr << "Usage: sextant <command> [options]\n"
                  << "Commands: build, search, insert\n";
        return 1;
    }

    spdlog::info("sextant CLI stub — command: {}", argv[1]);
    return 0;
}
