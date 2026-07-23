#pragma once

/// @file bucket_queue.hpp
/// Bucket queue for beam_search frontier/W heaps.
///
/// Replaces std::push_heap / std::pop_heap (O(log L) comparisons) with O(1)
/// bucket-array insertion. The bucket queue trades comparisons for a small
/// flat array indexed by quantized distance. At L=400, the binary heap does
/// ~9 comparisons per push/pop, each touching a cache line in the ~4.8KB heap.
/// The bucket queue does 1 bucket-index computation + 1 append per push, and
/// a scan-from-lowest-bucket per pop (amortized O(1) since most pops hit the
/// current lowest bucket).
///
/// Design:
/// - Buckets are linear in distance, width = d_ref / kNumBuckets.
/// - d_ref is calibrated from the first push (the first entry-point distance).
/// - Distances below d_ref map to bucket 0; above d_ref × kOverflowScale map
///   to the last bucket (clamped). Re-calibration is NOT done (the search
///   converges to a narrow range; the first entry point is a reasonable ref).
/// - Each bucket is a small vector of (dist, id) pairs. Pop-min scans from
///   the lowest non-empty bucket; within a bucket, linear scan for the min.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

namespace sextant {

struct BucketItem {
    float dist;
    uint32_t id;
};

class BucketQueue {
public:
    static constexpr uint32_t kNumBuckets = 128;
    static constexpr float kOverflowScale = 2.0f;  // max dist = d_ref × 2

    BucketQueue() : buckets_(kNumBuckets) {}

    void clear() {
        for (auto& b : buckets_) b.clear();
        count_ = 0;
        d_min_ = -1.0f;  // sentinel: not calibrated
        bucket_width_ = 0.0f;
        min_bucket_ = kNumBuckets;  // no buckets have data
    }

    bool empty() const { return count_ == 0; }
    size_t size() const { return count_; }

    void push(float dist, uint32_t id) {
        if (d_min_ < 0.0f) {
            // First push: calibrate.
            d_min_ = dist * 0.5f;  // allow some headroom below
            bucket_width_ = dist / static_cast<float>(kNumBuckets);
            if (bucket_width_ <= 0.0f) bucket_width_ = 1.0f;  // guard
            min_bucket_ = kNumBuckets;
        }
        const uint32_t b = bucket_index(dist);
        buckets_[b].push_back({dist, id});
        if (b < min_bucket_) min_bucket_ = b;
        count_++;
    }

    /// Pop the minimum-distance item. Returns false if empty.
    bool pop_min(BucketItem& out) {
        if (count_ == 0) return false;
        // Scan from min_bucket_ for the first non-empty bucket.
        while (min_bucket_ < kNumBuckets && buckets_[min_bucket_].empty()) {
            min_bucket_++;
        }
        if (min_bucket_ >= kNumBuckets) return false;  // shouldn't happen
        // Within the bucket, find the min-distance item (small buckets → linear scan).
        auto& bucket = buckets_[min_bucket_];
        uint32_t min_idx = 0;
        float min_d = bucket[0].dist;
        for (uint32_t i = 1; i < bucket.size(); i++) {
            if (bucket[i].dist < min_d) {
                min_d = bucket[i].dist;
                min_idx = i;
            }
        }
        out = bucket[min_idx];
        // Swap-remove (order doesn't matter within a bucket).
        bucket[min_idx] = bucket.back();
        bucket.pop_back();
        count_--;
        return true;
    }

    /// Peek at the minimum distance (without popping).
    float peek_min_dist() const {
        if (count_ == 0) return std::numeric_limits<float>::max();
        // Scan from min_bucket_ for the first non-empty bucket.
        for (uint32_t b = min_bucket_; b < kNumBuckets; b++) {
            if (!buckets_[b].empty()) {
                float min_d = buckets_[b][0].dist;
                for (size_t i = 1; i < buckets_[b].size(); i++) {
                    if (buckets_[b][i].dist < min_d) min_d = buckets_[b][i].dist;
                }
                return min_d;
            }
        }
        return std::numeric_limits<float>::max();
    }

    /// Drain all items sorted ascending by distance. Clears the queue.
    void drain_sorted(std::vector<BucketItem>& out) {
        out.reserve(out.size() + count_);
        for (uint32_t b = 0; b < kNumBuckets; b++) {
            // Within a bucket, sort by distance (buckets are narrow → small sorts).
            auto& bucket = buckets_[b];
            if (bucket.size() > 1) {
                std::sort(bucket.begin(), bucket.end(),
                          [](const BucketItem& a, const BucketItem& b) {
                              return a.dist < b.dist;
                          });
            }
            for (const auto& item : bucket) {
                out.push_back(item);
            }
            bucket.clear();
        }
        count_ = 0;
        min_bucket_ = kNumBuckets;
    }

private:
    uint32_t bucket_index(float dist) const {
        const float max_dist = d_min_ + bucket_width_ * kNumBuckets * kOverflowScale;
        if (dist <= d_min_) return 0;
        if (dist >= max_dist) return kNumBuckets - 1;
        const float idx = (dist - d_min_) / bucket_width_;
        return std::min(static_cast<uint32_t>(idx), kNumBuckets - 1);
    }

    std::vector<std::vector<BucketItem>> buckets_;
    size_t count_ = 0;
    float d_min_ = -1.0f;        // calibration base (negative = not calibrated)
    float bucket_width_ = 0.0f;  // distance per bucket
    uint32_t min_bucket_ = kNumBuckets;  // lowest non-empty bucket (kNumBuckets = none)
};

}  // namespace sextant
