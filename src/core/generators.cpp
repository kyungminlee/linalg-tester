#include "generators.h"

#include <algorithm>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <random>

/* ------------------------------------------------------------------ */
/* Internal helper: create an mt19937 from seed, update seed after use */
/* ------------------------------------------------------------------ */

static std::mt19937 make_rng(unsigned *seed)
{
    return std::mt19937(*seed);
}

static void update_seed(std::mt19937 &gen, unsigned *seed)
{
    *seed = gen();
}

static double rand_val(std::mt19937 &gen,
                        std::uniform_real_distribution<double> &dist)
{
    return dist(gen);
}

/* ------------------------------------------------------------------ */
/* gen_random_array                                                     */
/* ------------------------------------------------------------------ */

void *gen_random_array(int count, std::size_t typesize,
                        mpfr_to_custom_fn from_mpfr, mpfr_prec_t prec,
                        unsigned *seed)
{
    void *arr = std::malloc(static_cast<std::size_t>(count) * typesize);
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    char *p = static_cast<char *>(arr);
    for (int i = 0; i < count; ++i) {
        mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
        from_mpfr(p, val, MPFR_RNDN);
        p += typesize;
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return arr;
}

/* ------------------------------------------------------------------ */
/* gen_triangular_array                                                 */
/* ------------------------------------------------------------------ */

void *gen_triangular_array(int k, char uplo, char diag,
                            std::size_t typesize,
                            mpfr_to_custom_fn from_mpfr,
                            mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(k) * k, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            char *elem = arr + (static_cast<std::size_t>(j) * k + i) * typesize;
            if (i == j) {
                if (diag == 'U') {
                    mpfr_set_d(val, 1.0, MPFR_RNDN);
                } else {
                    double d = static_cast<double>(k) + std::abs(rand_val(gen, dist));
                    if (gen() & 1) d = -d;
                    mpfr_set_d(val, d, MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            } else if ((uplo == 'U' && j > i) || (uplo == 'L' && i > j)) {
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
            /* else: zero (from calloc) */
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_symmetric_array                                                  */
/* ------------------------------------------------------------------ */

void *gen_symmetric_array(int n, char uplo,
                           std::size_t typesize,
                           mpfr_to_custom_fn from_mpfr,
                           mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(n) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            char *elem_ij = arr + (static_cast<std::size_t>(j) * n + i) * typesize;
            char *elem_ji = arr + (static_cast<std::size_t>(i) * n + j) * typesize;

            if (i == j) {
                /* Diagonal: random value */
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem_ij, val, MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                /* Generate in the stored triangle, mirror to the other */
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem_ij, val, MPFR_RNDN);
                from_mpfr(elem_ji, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_band_array (LAPACK general band storage)                         */
/* ------------------------------------------------------------------ */

void *gen_band_array(int m, int n, int kl, int ku,
                      std::size_t typesize,
                      mpfr_to_custom_fn from_mpfr,
                      mpfr_prec_t prec, unsigned *seed)
{
    int ldab = kl + ku + 1;
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ldab) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        int i_min = std::max(0, j - ku);
        int i_max = std::min(m - 1, j + kl);
        for (int i = i_min; i <= i_max; ++i) {
            /* Band storage: row index in AB is ku + i - j */
            int ab_row = ku + i - j;
            char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
            mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
            from_mpfr(elem, val, MPFR_RNDN);
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_symmetric_band_array                                             */
/* ------------------------------------------------------------------ */

void *gen_symmetric_band_array(int n, int k, char uplo,
                                std::size_t typesize,
                                mpfr_to_custom_fn from_mpfr,
                                mpfr_prec_t prec, unsigned *seed)
{
    int ldab = k + 1;
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ldab) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            /* A(i,j) stored at AB[k+i-j + j*ldab] for max(0,j-k) <= i <= j */
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
        } else {
            /* uplo == 'L': A(i,j) stored at AB[i-j + j*ldab] for j <= i <= min(n-1,j+k) */
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_triangular_band_array                                            */
/* ------------------------------------------------------------------ */

void *gen_triangular_band_array(int n, int k, char uplo, char diag,
                                 std::size_t typesize,
                                 mpfr_to_custom_fn from_mpfr,
                                 mpfr_prec_t prec, unsigned *seed)
{
    int ldab = k + 1;
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ldab) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            /* Upper triangular band: rows max(0,j-k)..j stored at AB[k+i-j + j*ldab] */
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(val, 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(val, d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            }
        } else {
            /* Lower triangular band: rows j..min(n-1,j+k) stored at AB[i-j + j*ldab] */
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(val, 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(val, d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_packed_symmetric_array                                           */
/* ------------------------------------------------------------------ */

void *gen_packed_symmetric_array(int n, char uplo,
                                  std::size_t typesize,
                                  mpfr_to_custom_fn from_mpfr,
                                  mpfr_prec_t prec, unsigned *seed)
{
    std::size_t count = static_cast<std::size_t>(n) * (n + 1) / 2;
    char *arr = static_cast<char *>(std::malloc(count * typesize));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        /* Column-major packed upper: A(i,j) at index i + j*(j+1)/2 for i <= j */
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                char *elem = arr + idx * typesize;
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    } else {
        /* Column-major packed lower: A(i,j) at index i + j*(2n-j-1)/2 for i >= j */
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                char *elem = arr + idx * typesize;
                mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_packed_triangular_array                                          */
/* ------------------------------------------------------------------ */

void *gen_packed_triangular_array(int n, char uplo, char diag,
                                   std::size_t typesize,
                                   mpfr_to_custom_fn from_mpfr,
                                   mpfr_prec_t prec, unsigned *seed)
{
    std::size_t count = static_cast<std::size_t>(n) * (n + 1) / 2;
    char *arr = static_cast<char *>(std::calloc(count, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t val;
    mpfr_init2(val, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        /* Column-major packed upper: A(i,j) at index i + j*(j+1)/2 for i <= j */
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                char *elem = arr + idx * typesize;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(val, 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(val, d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    } else {
        /* Column-major packed lower: A(i,j) at index i + j*(2n-j-1)/2 for i >= j */
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                char *elem = arr + idx * typesize;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(val, 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(val, d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(val, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr(elem, val, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(val);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}
