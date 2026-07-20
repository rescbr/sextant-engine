#include "sextant/index.hpp"

#include "algo/vamana_core.hpp"
#include "quant/pq_quantizer.hpp"
#include "storage/direct_io.hpp"  // aligned_free
#include "storage/memgraph.hpp"
#include "storage/node_store.hpp"

namespace sextant {

Index::~Index() {
    if (codes_buffer) aligned_free(codes_buffer);
    if (nodes_buffer) aligned_free(nodes_buffer);
    if (raw_vecs_buffer) aligned_free(raw_vecs_buffer);
}

NodeStore* Index::top_store() const {
    if (memgraph) return memgraph.get();
    if (paged_store) return paged_store.get();
    return flat_store.get();
}

}  // namespace sextant
