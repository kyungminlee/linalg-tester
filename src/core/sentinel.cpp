#include "sentinel.h"

#include <cstdlib>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

void fill_sentinel(void *buf, std::size_t total_bytes, unsigned seed)
{
    std::mt19937 rng(seed);
    auto *p = static_cast<unsigned char *>(buf);
    for (std::size_t i = 0; i < total_bytes; ++i) {
        p[i] = static_cast<unsigned char>(rng() & 0xFF);
    }
}

SentinelResult check_sentinels(const void *buf, int num_elements,
                                std::size_t typesize,
                                const bool *active_mask, unsigned seed)
{
    SentinelResult result{true, 0, 0};

    std::size_t total_bytes = static_cast<std::size_t>(num_elements) * typesize;

    // Regenerate expected sentinel pattern
    std::vector<unsigned char> expected(total_bytes);
    fill_sentinel(expected.data(), total_bytes, seed);

    auto *actual = static_cast<const unsigned char *>(buf);

    for (int i = 0; i < num_elements; ++i) {
        if (active_mask[i]) {
            continue; // active element, allowed to differ
        }
        std::size_t offset = static_cast<std::size_t>(i) * typesize;
        if (std::memcmp(actual + offset, expected.data() + offset, typesize) != 0) {
            ++result.corrupted_count;
            if (result.passed) {
                result.passed = false;
                result.first_offset = offset;
            }
        }
    }

    return result;
}

SentinelResult check_matrix_sentinels(const void *buf, int rows, int cols,
                                       int ld, std::size_t typesize,
                                       unsigned seed)
{
    int num_elements = ld * cols;
    auto active_mask = std::make_unique<bool[]>(num_elements);
    std::memset(active_mask.get(), 0, num_elements * sizeof(bool));

    // Column-major: element (i,j) is at index j*ld + i
    for (int j = 0; j < cols; ++j) {
        for (int i = 0; i < rows; ++i) {
            active_mask[j * ld + i] = true;
        }
        // positions i = rows..ld-1 in column j remain false (sentinel)
    }

    return check_sentinels(buf, num_elements, typesize, active_mask.get(), seed);
}

SentinelResult check_vector_sentinels(const void *buf, int n, int inc,
                                       std::size_t typesize, unsigned seed)
{
    int abs_inc = std::abs(inc);
    if (abs_inc == 0) {
        abs_inc = 1;
    }

    int num_elements = 1 + (n - 1) * abs_inc;
    auto active_mask = std::make_unique<bool[]>(num_elements);
    std::memset(active_mask.get(), 0, num_elements * sizeof(bool));

    for (int k = 0; k < n; ++k) {
        active_mask[k * abs_inc] = true;
    }

    return check_sentinels(buf, num_elements, typesize, active_mask.get(), seed);
}
