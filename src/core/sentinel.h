#pragma once
#include <cstddef>

struct SentinelResult {
    bool passed;              // true if all sentinels survived
    int corrupted_count;      // number of corrupted sentinel positions
    std::size_t first_offset; // byte offset of first corruption (0 if passed)
};

// Fill buffer with sentinel pattern (deterministic from seed)
void fill_sentinel(void *buf, std::size_t total_bytes, unsigned seed);

// Check that all non-active positions still have sentinel values.
// active_mask[i] = true means element i is active (allowed to change).
// Returns SentinelResult.
SentinelResult check_sentinels(const void *buf, int num_elements,
                                std::size_t typesize,
                                const bool *active_mask, unsigned seed);

// Convenience: check matrix ld-padding sentinels
// Active positions: (i,j) where 0 <= i < rows, 0 <= j < cols
// Sentinel positions: (i,j) where rows <= i < ld, 0 <= j < cols
SentinelResult check_matrix_sentinels(const void *buf, int rows, int cols,
                                       int ld, std::size_t typesize,
                                       unsigned seed);

// Convenience: check vector stride-gap sentinels
// Active positions: k * |inc| for k = 0..n-1
// Sentinel positions: all other positions in [0, 1+(n-1)*|inc|)
SentinelResult check_vector_sentinels(const void *buf, int n, int inc,
                                       std::size_t typesize, unsigned seed);
