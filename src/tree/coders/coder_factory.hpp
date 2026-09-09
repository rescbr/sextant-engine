#pragma once

/// @file coder_factory.hpp
/// make_leaf_coder(): the ONLY place quantizer_type strings are interpreted.

#include "../leaf_coder.hpp"

namespace sextant::tree {

/// See leaf_coder.hpp for the contract. `params` is IN/OUT (scalar families
/// normalize m4 = dim; the scan_i8 override (-1=auto, via
/// set_scan_i8_override) is resolved into
/// scan_i8_mode once).
std::unique_ptr<LeafCoder> make_leaf_coder(const std::string& quantizer_type,
                                           CoderParams& params, bool for_open);

}  // namespace sextant::tree
