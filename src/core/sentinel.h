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

// Allocate a buffer of num_elements * typesize bytes, filled with sentinel pattern.
// Returns the allocated buffer (caller must std::free it).
void *alloc_with_sentinel(int num_elements, std::size_t typesize, unsigned sentinel_seed);

// Copy only active matrix positions (rows 0..rows-1 of each column) from src to dst.
// Both src and dst have leading dimension ld. Only the active rows are copied;
// sentinel positions in the padding (rows..ld-1) are left untouched in dst.
void copy_matrix_active(void *dst, const void *src, int rows, int cols,
                         int ld, std::size_t typesize);

// Copy only active vector positions (k * |inc| for k = 0..n-1) from src to dst.
// Both src and dst have total length 1 + (n-1)*|inc|. Only the strided active
// elements are copied; gap positions are left untouched in dst.
void copy_vector_active(void *dst, const void *src, int n, int inc,
                         std::size_t typesize);
