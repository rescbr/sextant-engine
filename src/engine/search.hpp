#pragma once

/// @file search.hpp
/// Search pipeline.

#include <sextant/types.hpp>
#include <vector>
#include <string>

namespace sextant {

struct SearchConfig {
    uint32_t k = 10;
    uint32_t L_search = 200;
    uint32_t rerank_factor = 10;
    uint32_t io_limit = 64;
};

}  // namespace sextant
