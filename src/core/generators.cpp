#include "generators.h"
#include "mpfr_lapack_utils.h"
#include "mpfr_lapack_complex_utils.h"

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

/* ------------------------------------------------------------------ */
/* gen_random_complex_array                                             */
/* ------------------------------------------------------------------ */

void *gen_random_complex_array(int count, std::size_t typesize,
                                mpfr_to_custom_complex_fn from_mpfr_complex,
                                mpfr_prec_t prec, unsigned *seed)
{
    void *arr = std::malloc(static_cast<std::size_t>(count) * typesize);
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    char *p = static_cast<char *>(arr);
    for (int i = 0; i < count; ++i) {
        mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
        from_mpfr_complex(p, re, im, MPFR_RNDN);
        p += typesize;
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return arr;
}

/* ------------------------------------------------------------------ */
/* gen_hermitian_array  (A = A^H: real diagonal, conjugated off-diag)  */
/* ------------------------------------------------------------------ */

void *gen_hermitian_array(int n, char uplo,
                           std::size_t typesize,
                           mpfr_to_custom_complex_fn from_mpfr_complex,
                           mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(n) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            char *elem_ij = arr + (static_cast<std::size_t>(j) * n + i) * typesize;
            char *elem_ji = arr + (static_cast<std::size_t>(i) * n + j) * typesize;

            if (i == j) {
                /* Diagonal: real only */
                mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(im, 0.0, MPFR_RNDN);
                from_mpfr_complex(elem_ij, re, im, MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                /* Stored triangle: generate (re, im), mirror as conjugate */
                double rv = rand_val(gen, dist);
                double iv = rand_val(gen, dist);
                mpfr_set_d(re, rv, MPFR_RNDN);
                mpfr_set_d(im, iv, MPFR_RNDN);
                from_mpfr_complex(elem_ij, re, im, MPFR_RNDN);
                /* Conjugate in mirrored position */
                mpfr_set_d(im, -iv, MPFR_RNDN);
                from_mpfr_complex(elem_ji, re, im, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_complex_symmetric_array  (A = A^T: no conjugation on mirror)    */
/* ------------------------------------------------------------------ */

void *gen_complex_symmetric_array(int n, char uplo,
                                   std::size_t typesize,
                                   mpfr_to_custom_complex_fn from_mpfr_complex,
                                   mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(n) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            char *elem_ij = arr + (static_cast<std::size_t>(j) * n + i) * typesize;
            char *elem_ji = arr + (static_cast<std::size_t>(i) * n + j) * typesize;

            if (i == j) {
                /* Diagonal: complex (symmetric allows complex diagonal) */
                mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr_complex(elem_ij, re, im, MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                /* Stored triangle: generate (re, im), mirror without conjugation */
                double rv = rand_val(gen, dist);
                double iv = rand_val(gen, dist);
                mpfr_set_d(re, rv, MPFR_RNDN);
                mpfr_set_d(im, iv, MPFR_RNDN);
                from_mpfr_complex(elem_ij, re, im, MPFR_RNDN);
                /* Same value in mirrored position (no conjugation) */
                from_mpfr_complex(elem_ji, re, im, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_hermitian_band_array                                             */
/* ------------------------------------------------------------------ */

void *gen_hermitian_band_array(int n, int k, char uplo,
                                std::size_t typesize,
                                mpfr_to_custom_complex_fn from_mpfr_complex,
                                mpfr_prec_t prec, unsigned *seed)
{
    int ldab = k + 1;
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ldab) * n, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                if (i == j) {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            }
        } else {
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                char *elem = arr + (static_cast<std::size_t>(j) * ldab + ab_row) * typesize;
                if (i == j) {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_packed_hermitian_array                                           */
/* ------------------------------------------------------------------ */

void *gen_packed_hermitian_array(int n, char uplo,
                                  std::size_t typesize,
                                  mpfr_to_custom_complex_fn from_mpfr_complex,
                                  mpfr_prec_t prec, unsigned *seed)
{
    std::size_t count = static_cast<std::size_t>(n) * (n + 1) / 2;
    char *arr = static_cast<char *>(std::malloc(count * typesize));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                char *elem = arr + idx * typesize;
                if (i == j) {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                char *elem = arr + idx * typesize;
                if (i == j) {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                }
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            }
        }
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_triangular_complex_array                                         */
/* ------------------------------------------------------------------ */

void *gen_triangular_complex_array(int k, char uplo, char diag,
                                    std::size_t typesize,
                                    mpfr_to_custom_complex_fn from_mpfr_complex,
                                    mpfr_prec_t prec, unsigned *seed)
{
    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(k) * k, typesize));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    mpfr_t re, im;
    mpfr_init2(re, prec);
    mpfr_init2(im, prec);

    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            char *elem = arr + (static_cast<std::size_t>(j) * k + i) * typesize;
            if (i == j) {
                if (diag == 'U') {
                    mpfr_set_d(re, 1.0, MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                } else {
                    double d = static_cast<double>(k) + std::abs(rand_val(gen, dist));
                    if (gen() & 1) d = -d;
                    mpfr_set_d(re, d, MPFR_RNDN);
                    mpfr_set_d(im, 0.0, MPFR_RNDN);
                }
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            } else if ((uplo == 'U' && j > i) || (uplo == 'L' && i > j)) {
                mpfr_set_d(re, rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(im, rand_val(gen, dist), MPFR_RNDN);
                from_mpfr_complex(elem, re, im, MPFR_RNDN);
            }
            /* else: zero (from calloc) */
        }
    }

    mpfr_clear(re);
    mpfr_clear(im);
    update_seed(gen, seed);
    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* gen_positive_definite_array                                          */
/* A = R^T R + n*I where R is random, ensuring SPD                     */
/* Returns column-major n-by-n matrix in custom format, ld=n           */
/* ------------------------------------------------------------------ */

void *gen_positive_definite_array(int n,
                                  std::size_t typesize,
                                  mpfr_to_custom_fn from_mpfr,
                                  custom_to_mpfr_fn to_mpfr,
                                  mpfr_prec_t prec,
                                  unsigned *seed)
{
    /* Generate random matrix R */
    void *R_raw = gen_random_array(n * n, typesize, from_mpfr, prec, seed);

    /* Convert to MPFR */
    MpfrMatrix R(n, n, prec);
    const char *rp = static_cast<const char *>(R_raw);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            to_mpfr(R.at(i, j), rp + IDX(i, j, n) * typesize);

    /* Compute A = R^T * R */
    MpfrMatrix Rt(n, n, prec);
    mpfr_mat_transpose(Rt, R);

    MpfrMatrix A(n, n, prec);
    mpfr_mat_mul_simple(A, Rt, R);

    /* Add n*I to ensure well-conditioned */
    MpfrScalar nval(prec);
    mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);
    for (int i = 0; i < n; ++i)
        mpfr_add(A.at(i, i), A.at(i, i), nval.get(), MPFR_RNDN);

    /* Convert back to custom format */
    char *arr = static_cast<char *>(std::malloc(static_cast<std::size_t>(n) * n * typesize));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            from_mpfr(arr + IDX(i, j, n) * typesize, A.at(i, j), MPFR_RNDN);

    std::free(R_raw);
    return arr;
}

/* ------------------------------------------------------------------ */
/* gen_hermitian_positive_definite_array                                */
/* A = R^H R + n*I where R is random complex, ensuring HPD             */
/* Returns column-major n-by-n matrix in custom format, ld=n           */
/* ------------------------------------------------------------------ */

void *gen_hermitian_positive_definite_array(int n,
                                             std::size_t typesize,
                                             mpfr_to_custom_complex_fn from_mpfr_complex,
                                             custom_to_mpfr_complex_fn to_mpfr_complex,
                                             mpfr_prec_t prec,
                                             unsigned *seed)
{
    /* Generate random complex matrix R */
    void *R_raw = gen_random_complex_array(n * n, typesize, from_mpfr_complex, prec, seed);

    /* Convert to MPFR complex */
    MpfrComplexMatrix R(n, n, prec);
    const char *rp = static_cast<const char *>(R_raw);
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            to_mpfr_complex(R.re(i, j), R.im(i, j),
                            rp + IDX(i, j, n) * typesize);

    /* Compute A = R^H * R */
    MpfrComplexMatrix Rh(n, n, prec);
    mpfr_complex_mat_adjoint(Rh, R);

    MpfrComplexMatrix A(n, n, prec);
    mpfr_complex_mat_mul_simple(A, Rh, R);

    /* Add n*I to ensure well-conditioned; set diagonal imaginary to 0 */
    MpfrScalar nval(prec);
    mpfr_set_d(nval.get(), static_cast<double>(n), MPFR_RNDN);
    for (int i = 0; i < n; ++i) {
        mpfr_add(A.re(i, i), A.re(i, i), nval.get(), MPFR_RNDN);
        mpfr_set_d(A.im(i, i), 0.0, MPFR_RNDN);
    }

    /* Convert back to custom format */
    char *arr = static_cast<char *>(std::malloc(static_cast<std::size_t>(n) * n * typesize));
    for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
            from_mpfr_complex(arr + IDX(i, j, n) * typesize,
                              A.re(i, j), A.im(i, j), MPFR_RNDN);

    std::free(R_raw);
    return arr;
}
