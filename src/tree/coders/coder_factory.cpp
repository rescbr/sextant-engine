#include "coder_factory.hpp"

#include "global_pq_coder.hpp"
#include "local_pq_coder.hpp"
#include "local_scalar_coder.hpp"
#include "scalar_lm_coder.hpp"
#include "spdlog/spdlog.h"

#include <cstdlib>
#include <stdexcept>

namespace sextant::tree {

namespace scan_detail {
thread_local int g_scan_i8_override = -1;
int scan_i8_override() { return g_scan_i8_override; }
void set_override(int v) { g_scan_i8_override = v; }
}  // namespace scan_detail

std::unique_ptr<LeafCoder> make_leaf_coder(const std::string& quantizer_type,
                                           CoderParams& params,
                                           bool for_open) {
    // Resolve the thread-local kernel override ONCE, on the opening
    // thread: scan setups are built on sweep worker threads where a
    // thread-local reads -1 (the CLI flag must reach the workers).
    // Per-search overrides (C API) set AFTER open keep the thread-local
    // route on caller threads.
    if (for_open && scan_detail::scan_i8_override() > 0)
        params.scan_i8_mode =
            static_cast<uint8_t>(scan_detail::scan_i8_override());
    if (quantizer_type == "pq") {
        return std::make_unique<GlobalPqCoder>(GlobalPqCoder::Kind::Plain,
                                               params);
    }
    if (quantizer_type == "anisotropic_pq" ||
        quantizer_type == "anisotropic-pq") {
        // Anisotropic training objective only; the codebook deserializes
        // through the PqQuantizer base — at open time construct the plain
        // base so search is identical.
        if (for_open)
            return std::make_unique<GlobalPqCoder>(GlobalPqCoder::Kind::Plain,
                                                   params);
        return std::make_unique<GlobalPqCoder>(
            GlobalPqCoder::Kind::Anisotropic, params);
    }
    if (quantizer_type == "prq") {
        // Clamp nsplits to a divisor of both m and dim (build-time resolve).
        if (!for_open && params.m4 > 0 && params.dim > 0) {
            uint32_t nsplits = (params.prq_nsplits > 0)
                ? params.prq_nsplits
                : static_cast<uint32_t>(params.dim) / 8;
            if (params.m4 % nsplits != 0 || params.dim % nsplits != 0) {
                uint32_t best = 1;
                for (uint32_t ns = nsplits; ns >= 1; --ns) {
                    if (params.m4 % ns == 0 && params.dim % ns == 0) {
                        best = ns;
                        break;
                    }
                }
                spdlog::warn(
                    "[sextant] PRQ: nsplits {} incompatible with m={} dim={}, "
                    "clamped to {}", nsplits, params.m4, params.dim, best);
                nsplits = best;
            }
            params.prq_nsplits = nsplits;
        }
        return std::make_unique<GlobalPqCoder>(GlobalPqCoder::Kind::Prq,
                                               params);
    }
    if (quantizer_type == "local_pq") {
        return std::make_unique<LocalPqCoder>(params);
    }
    if (quantizer_type == "local_scalar") {
        if (params.pq_bits != 4) {
            throw std::invalid_argument(
                "local_scalar currently supports only --pq-bits 4");
        }
        params.m4 = static_cast<uint16_t>(params.dim);
        params.has_ip_bias = params.metric == MetricKind::InnerProduct;
        return std::make_unique<LocalScalarCoder>(params);
    }
    if (quantizer_type == "scalar_lloydmax" ||
        quantizer_type == "scalar_uniform" ||
        quantizer_type == "scalar_shape") {
        if (params.pq_bits != 4) {
            throw std::invalid_argument(
                "scalar_lloydmax/scalar_uniform/scalar_shape currently "
                "support only --pq-bits 4");
        }
        // Scalar quantization is sub_dim=1: m = dim subquantizers.
        params.m4 = static_cast<uint16_t>(params.dim);
        params.has_ip_bias = params.metric == MetricKind::InnerProduct;
        ScalarLmCoder::LevelPolicy policy =
            ScalarLmCoder::LevelPolicy::LloydMax;
        if (quantizer_type == "scalar_uniform")
            policy = ScalarLmCoder::LevelPolicy::Uniform;
        else if (quantizer_type == "scalar_shape")
            policy = ScalarLmCoder::LevelPolicy::Shape;
        return std::make_unique<ScalarLmCoder>(policy, params);
    }
    throw std::runtime_error("IVFTreeIndex::open: unknown quantizer_type '" +
                             quantizer_type + "'");
}

}  // namespace sextant::tree
