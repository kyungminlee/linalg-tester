/* mirror_gen.cpp -- MPFR-first generators and native materializers */

#include "mirror_gen.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
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

/* ================================================================== */
/* Real MPFR-first generators                                          */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_matrix                                               */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_matrix(MpfrMatrix &dst, mpfr_prec_t prec,
                             unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < dst.cols(); ++j)
        for (int i = 0; i < dst.rows(); ++i)
            mpfr_set_d(dst.at(i, j), rand_val(gen, dist), MPFR_RNDN);

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_vector                                               */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_vector(MpfrMatrix &dst, mpfr_prec_t prec,
                             unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    int n = dst.rows();
    for (int i = 0; i < n; ++i)
        mpfr_set_d(dst.at(i, 0), rand_val(gen, dist), MPFR_RNDN);

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_scalar                                               */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_scalar(MpfrScalar &dst, mpfr_prec_t prec,
                             unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    mpfr_set_d(dst.get(), rand_val(gen, dist), MPFR_RNDN);

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_triangular_matrix                                           */
/* ------------------------------------------------------------------ */

void gen_mpfr_triangular_matrix(MpfrMatrix &dst, char uplo, char diag,
                                 mpfr_prec_t prec, unsigned *seed)
{
    int k = dst.rows(); /* square k x k */
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            if (i == j) {
                if (diag == 'U') {
                    mpfr_set_d(dst.at(i, j), 1.0, MPFR_RNDN);
                } else {
                    double d = static_cast<double>(k) + std::abs(rand_val(gen, dist));
                    if (gen() & 1) d = -d;
                    mpfr_set_d(dst.at(i, j), d, MPFR_RNDN);
                }
            } else if ((uplo == 'U' && j > i) || (uplo == 'L' && i > j)) {
                mpfr_set_d(dst.at(i, j), rand_val(gen, dist), MPFR_RNDN);
            } else {
                mpfr_set_d(dst.at(i, j), 0.0, MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_symmetric_matrix                                            */
/* ------------------------------------------------------------------ */

void gen_mpfr_symmetric_matrix(MpfrMatrix &dst, char uplo,
                                mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j) {
                mpfr_set_d(dst.at(i, j), rand_val(gen, dist), MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                double v = rand_val(gen, dist);
                mpfr_set_d(dst.at(i, j), v, MPFR_RNDN);
                mpfr_set(dst.at(j, i), dst.at(i, j), MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_band_matrix                                                 */
/* ------------------------------------------------------------------ */

void gen_mpfr_band_matrix(MpfrMatrix &dst, int m, int kl, int ku,
                           mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.cols();
    int ldab = kl + ku + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Zero-fill the entire band storage */
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r)
            mpfr_set_d(dst.at(r, j), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        int i_min = std::max(0, j - ku);
        int i_max = std::min(m - 1, j + kl);
        for (int i = i_min; i <= i_max; ++i) {
            int ab_row = ku + i - j;
            mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_symmetric_band_matrix                                       */
/* ------------------------------------------------------------------ */

void gen_mpfr_symmetric_band_matrix(MpfrMatrix &dst, int n, int k,
                                     char uplo, mpfr_prec_t prec,
                                     unsigned *seed)
{
    int ldab = k + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Zero-fill */
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r)
            mpfr_set_d(dst.at(r, j), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
            }
        } else {
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_triangular_band_matrix                                      */
/* ------------------------------------------------------------------ */

void gen_mpfr_triangular_band_matrix(MpfrMatrix &dst, int n, int k,
                                      char uplo, char diag,
                                      mpfr_prec_t prec, unsigned *seed)
{
    int ldab = k + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Zero-fill */
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r)
            mpfr_set_d(dst.at(r, j), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(dst.at(ab_row, j), 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(dst.at(ab_row, j), d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        } else {
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(dst.at(ab_row, j), 1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(dst.at(ab_row, j), d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_packed_symmetric                                            */
/* ------------------------------------------------------------------ */

void gen_mpfr_packed_symmetric(MpfrMatrix &dst, int n, char uplo,
                                mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                           rand_val(gen, dist), MPFR_RNDN);
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                           rand_val(gen, dist), MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_packed_triangular                                           */
/* ------------------------------------------------------------------ */

void gen_mpfr_packed_triangular(MpfrMatrix &dst, int n, char uplo,
                                 char diag, mpfr_prec_t prec,
                                 unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                                   1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                                   d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                               rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                if (i == j) {
                    if (diag == 'U') {
                        mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                                   1.0, MPFR_RNDN);
                    } else {
                        double d = static_cast<double>(n) + std::abs(rand_val(gen, dist));
                        if (gen() & 1) d = -d;
                        mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                                   d, MPFR_RNDN);
                    }
                } else {
                    mpfr_set_d(dst.at(static_cast<int>(idx), 0),
                               rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_positive_definite_matrix                                    */
/* ------------------------------------------------------------------ */

void gen_mpfr_positive_definite_matrix(MpfrMatrix &dst, mpfr_prec_t prec,
                                        unsigned *seed)
{
    int n = dst.rows();
    gen_mpfr_symmetric_matrix(dst, 'L', prec, seed);
    /* Add n*I to make positive definite */
    for (int i = 0; i < n; ++i) {
        double diag = mpfr_get_d(dst.at(i, i), MPFR_RNDN);
        mpfr_set_d(dst.at(i, i), diag + static_cast<double>(n), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_diag_dominant_symmetric                                     */
/* ------------------------------------------------------------------ */

void gen_mpfr_diag_dominant_symmetric(MpfrMatrix &dst, char uplo,
                                       mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    gen_mpfr_symmetric_matrix(dst, uplo, prec, seed);
    for (int i = 0; i < n; ++i) {
        double diag = mpfr_get_d(dst.at(i, i), MPFR_RNDN);
        mpfr_set_d(dst.at(i, i), diag + static_cast<double>(n), MPFR_RNDN);
    }
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_gbsv_band_matrix                                           */
/* ------------------------------------------------------------------ */

void gen_mpfr_gbsv_band_matrix(MpfrMatrix &dst, int n, int kl, int ku,
                                 mpfr_prec_t prec, unsigned *seed)
{
    int ldab = 2 * kl + ku + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Zero-fill the entire storage */
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r)
            mpfr_set_d(dst.at(r, j), 0.0, MPFR_RNDN);

    /* Fill band entries: A(i,j) stored at AB(kl+ku+i-j, j) */
    for (int j = 0; j < n; ++j) {
        int i_min = std::max(0, j - ku);
        int i_max = std::min(n - 1, j + kl);
        for (int i = i_min; i <= i_max; ++i) {
            int ab_row = kl + ku + i - j;
            mpfr_set_d(dst.at(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
        }
        /* Add diagonal dominance */
        int diag_row = kl + ku;
        double d = mpfr_get_d(dst.at(diag_row, j), MPFR_RNDN);
        mpfr_set_d(dst.at(diag_row, j), d + static_cast<double>(n), MPFR_RNDN);
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_tridiagonal                                                */
/* ------------------------------------------------------------------ */

void gen_mpfr_tridiagonal(MpfrMatrix &dl, MpfrMatrix &d, MpfrMatrix &du,
                            int n, mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Generate sub/super-diagonals */
    for (int i = 0; i < n - 1; ++i) {
        mpfr_set_d(dl.at(i, 0), rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(du.at(i, 0), rand_val(gen, dist), MPFR_RNDN);
    }

    /* Generate diagonal with dominance */
    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        if (i > 0) sum += std::abs(mpfr_get_d(dl.at(i - 1, 0), MPFR_RNDN));
        if (i < n - 1) sum += std::abs(mpfr_get_d(du.at(i, 0), MPFR_RNDN));
        double dv = sum + static_cast<double>(n) + std::abs(rand_val(gen, dist));
        if (gen() & 1) dv = -dv;
        mpfr_set_d(d.at(i, 0), dv, MPFR_RNDN);
    }

    update_seed(gen, seed);
}

/* ================================================================== */
/* Complex MPFR-first generators                                       */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_complex_matrix                                       */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_complex_matrix(MpfrComplexMatrix &dst,
                                     mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < dst.cols(); ++j) {
        for (int i = 0; i < dst.rows(); ++i) {
            mpfr_set_d(dst.re(i, j), rand_val(gen, dist), MPFR_RNDN);
            mpfr_set_d(dst.im(i, j), rand_val(gen, dist), MPFR_RNDN);
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_complex_vector                                       */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_complex_vector(MpfrComplexMatrix &dst,
                                     mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    int n = dst.rows();
    for (int i = 0; i < n; ++i) {
        mpfr_set_d(dst.re(i, 0), rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(dst.im(i, 0), rand_val(gen, dist), MPFR_RNDN);
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_random_complex_scalar                                       */
/* ------------------------------------------------------------------ */

void gen_mpfr_random_complex_scalar(MpfrComplexScalar &dst,
                                     mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    mpfr_set_d(dst.re(), rand_val(gen, dist), MPFR_RNDN);
    mpfr_set_d(dst.im(), rand_val(gen, dist), MPFR_RNDN);

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_hermitian_matrix                                            */
/* ------------------------------------------------------------------ */

void gen_mpfr_hermitian_matrix(MpfrComplexMatrix &dst, char uplo,
                                mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j) {
                /* Diagonal: real only */
                mpfr_set_d(dst.re(i, j), rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), 0.0, MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                /* Stored triangle: generate (re, im), mirror as conjugate */
                double rv = rand_val(gen, dist);
                double iv = rand_val(gen, dist);
                mpfr_set_d(dst.re(i, j), rv, MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), iv, MPFR_RNDN);
                /* Conjugate in mirrored position */
                mpfr_set_d(dst.re(j, i), rv, MPFR_RNDN);
                mpfr_set_d(dst.im(j, i), -iv, MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_triangular_complex_matrix                                   */
/* ------------------------------------------------------------------ */

void gen_mpfr_triangular_complex_matrix(MpfrComplexMatrix &dst,
                                         char uplo, char diag,
                                         mpfr_prec_t prec, unsigned *seed)
{
    int k = dst.rows();
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < k; ++i) {
            if (i == j) {
                if (diag == 'U') {
                    mpfr_set_d(dst.re(i, j), 1.0, MPFR_RNDN);
                    mpfr_set_d(dst.im(i, j), 0.0, MPFR_RNDN);
                } else {
                    double d = static_cast<double>(k) + std::abs(rand_val(gen, dist));
                    if (gen() & 1) d = -d;
                    mpfr_set_d(dst.re(i, j), d, MPFR_RNDN);
                    mpfr_set_d(dst.im(i, j), 0.0, MPFR_RNDN);
                }
            } else if ((uplo == 'U' && j > i) || (uplo == 'L' && i > j)) {
                mpfr_set_d(dst.re(i, j), rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), rand_val(gen, dist), MPFR_RNDN);
            } else {
                mpfr_set_d(dst.re(i, j), 0.0, MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), 0.0, MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_hermitian_band_matrix                                       */
/* ------------------------------------------------------------------ */

void gen_mpfr_hermitian_band_matrix(MpfrComplexMatrix &dst, int n, int k,
                                     char uplo, mpfr_prec_t prec,
                                     unsigned *seed)
{
    int ldab = k + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Zero-fill */
    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r) {
            mpfr_set_d(dst.re(r, j), 0.0, MPFR_RNDN);
            mpfr_set_d(dst.im(r, j), 0.0, MPFR_RNDN);
        }

    for (int j = 0; j < n; ++j) {
        if (uplo == 'U') {
            int i_min = std::max(0, j - k);
            for (int i = i_min; i <= j; ++i) {
                int ab_row = k + i - j;
                if (i == j) {
                    mpfr_set_d(dst.re(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ab_row, j), 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(dst.re(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        } else {
            int i_max = std::min(n - 1, j + k);
            for (int i = j; i <= i_max; ++i) {
                int ab_row = i - j;
                if (i == j) {
                    mpfr_set_d(dst.re(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ab_row, j), 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(dst.re(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    }

    update_seed(gen, seed);
}

/* ------------------------------------------------------------------ */
/* gen_mpfr_packed_hermitian                                             */
/* ------------------------------------------------------------------ */

void gen_mpfr_packed_hermitian(MpfrComplexMatrix &dst, int n, char uplo,
                                mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    if (uplo == 'U') {
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i <= j; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (j + 1) / 2;
                int ii = static_cast<int>(idx);
                if (i == j) {
                    mpfr_set_d(dst.re(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ii, 0), 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(dst.re(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    } else {
        for (int j = 0; j < n; ++j) {
            for (int i = j; i < n; ++i) {
                std::size_t idx = static_cast<std::size_t>(i) +
                                  static_cast<std::size_t>(j) * (2 * n - j - 1) / 2;
                int ii = static_cast<int>(idx);
                if (i == j) {
                    mpfr_set_d(dst.re(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ii, 0), 0.0, MPFR_RNDN);
                } else {
                    mpfr_set_d(dst.re(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                    mpfr_set_d(dst.im(ii, 0), rand_val(gen, dist), MPFR_RNDN);
                }
            }
        }
    }

    update_seed(gen, seed);
}

/* ================================================================== */
/* Native materializers: MPFR -> void* native array                    */
/* ================================================================== */

/* ------------------------------------------------------------------ */
/* mpfr_mat_to_native                                                   */
/* ------------------------------------------------------------------ */

void *mpfr_mat_to_native(const MpfrMatrix &src, int ld,
                           const TesterCtx &ctx)
{
    int rows = src.rows();
    int cols = src.cols();
    std::size_t ts = ctx.typesize;

    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ld) * cols, ts));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            ctx.from_mpfr(arr + IDX(i, j, ld) * ts,
                          const_cast<mpfr_t &>(src.at(i, j)), MPFR_RNDN);

    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* mpfr_vec_to_native                                                   */
/* ------------------------------------------------------------------ */

void *mpfr_vec_to_native(const MpfrMatrix &src, int inc,
                           const TesterCtx &ctx)
{
    int n = src.rows();
    int abs_inc = (inc < 0) ? -inc : inc;
    std::size_t ts = ctx.typesize;
    std::size_t total = static_cast<std::size_t>(1 + (n - 1) * abs_inc);

    char *arr = static_cast<char *>(std::calloc(total, ts));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    for (int i = 0; i < n; ++i) {
        std::size_t offset;
        if (inc > 0) {
            offset = static_cast<std::size_t>(i) * abs_inc * ts;
        } else {
            offset = static_cast<std::size_t>(n - 1 - i) * abs_inc * ts;
        }
        ctx.from_mpfr(arr + offset,
                      const_cast<mpfr_t &>(src.at(i, 0)), MPFR_RNDN);
    }

    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* mpfr_scalar_to_native                                                */
/* ------------------------------------------------------------------ */

void *mpfr_scalar_to_native(const MpfrScalar &src, const TesterCtx &ctx)
{
    char *arr = static_cast<char *>(std::malloc(ctx.typesize));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    ctx.from_mpfr(arr, const_cast<mpfr_t &>(src.get()), MPFR_RNDN);

    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* mpfr_complex_mat_to_native                                           */
/* ------------------------------------------------------------------ */

void *mpfr_complex_mat_to_native(const MpfrComplexMatrix &src, int ld,
                                  const TesterCtx &ctx)
{
    int rows = src.rows();
    int cols = src.cols();
    std::size_t ts = ctx.typesize;

    char *arr = static_cast<char *>(
        std::calloc(static_cast<std::size_t>(ld) * cols, ts));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    for (int j = 0; j < cols; ++j)
        for (int i = 0; i < rows; ++i)
            ctx.from_mpfr_complex(
                arr + IDX(i, j, ld) * ts,
                const_cast<mpfr_t &>(src.re(i, j)),
                const_cast<mpfr_t &>(src.im(i, j)),
                MPFR_RNDN);

    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* mpfr_complex_vec_to_native                                           */
/* ------------------------------------------------------------------ */

void *mpfr_complex_vec_to_native(const MpfrComplexMatrix &src, int inc,
                                  const TesterCtx &ctx)
{
    int n = src.rows();
    int abs_inc = (inc < 0) ? -inc : inc;
    std::size_t ts = ctx.typesize;
    std::size_t total = static_cast<std::size_t>(1 + (n - 1) * abs_inc);

    char *arr = static_cast<char *>(std::calloc(total, ts));
    if (!arr) { std::perror("calloc"); std::exit(EXIT_FAILURE); }

    for (int i = 0; i < n; ++i) {
        std::size_t offset;
        if (inc > 0) {
            offset = static_cast<std::size_t>(i) * abs_inc * ts;
        } else {
            offset = static_cast<std::size_t>(n - 1 - i) * abs_inc * ts;
        }
        ctx.from_mpfr_complex(
            arr + offset,
            const_cast<mpfr_t &>(src.re(i, 0)),
            const_cast<mpfr_t &>(src.im(i, 0)),
            MPFR_RNDN);
    }

    return static_cast<void *>(arr);
}

/* ------------------------------------------------------------------ */
/* mpfr_complex_scalar_to_native                                        */
/* ------------------------------------------------------------------ */

void *mpfr_complex_scalar_to_native(const MpfrComplexScalar &src,
                                     const TesterCtx &ctx)
{
    char *arr = static_cast<char *>(std::malloc(ctx.typesize));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }

    ctx.from_mpfr_complex(arr,
                          const_cast<mpfr_t &>(src.re()),
                          const_cast<mpfr_t &>(src.im()),
                          MPFR_RNDN);

    return static_cast<void *>(arr);
}

/* ================================================================== */
/* Additional complex generators for LAPACK mirror tests               */
/* ================================================================== */

void gen_mpfr_hermitian_positive_definite(MpfrComplexMatrix &dst,
                                           mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    gen_mpfr_hermitian_matrix(dst, 'L', prec, seed);
    for (int i = 0; i < n; ++i) {
        double diag = mpfr_get_d(dst.re(i, i), MPFR_RNDN);
        mpfr_set_d(dst.re(i, i), diag + static_cast<double>(n), MPFR_RNDN);
        mpfr_set_d(dst.im(i, i), 0.0, MPFR_RNDN);
    }
}

void gen_mpfr_diag_dominant_hermitian(MpfrComplexMatrix &dst, char uplo,
                                       mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    gen_mpfr_hermitian_matrix(dst, uplo, prec, seed);
    for (int i = 0; i < n; ++i) {
        double diag = mpfr_get_d(dst.re(i, i), MPFR_RNDN);
        mpfr_set_d(dst.re(i, i), diag + static_cast<double>(n), MPFR_RNDN);
        mpfr_set_d(dst.im(i, i), 0.0, MPFR_RNDN);
    }
}

void gen_mpfr_diag_dominant_complex_symmetric(MpfrComplexMatrix &dst,
                                               char uplo, mpfr_prec_t prec,
                                               unsigned *seed)
{
    int n = dst.rows();
    gen_mpfr_complex_symmetric_matrix(dst, uplo, prec, seed);
    for (int i = 0; i < n; ++i) {
        double diag = mpfr_get_d(dst.re(i, i), MPFR_RNDN);
        mpfr_set_d(dst.re(i, i), diag + static_cast<double>(n), MPFR_RNDN);
    }
}

void gen_mpfr_complex_symmetric_matrix(MpfrComplexMatrix &dst, char uplo,
                                        mpfr_prec_t prec, unsigned *seed)
{
    int n = dst.rows();
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (i == j) {
                mpfr_set_d(dst.re(i, j), rand_val(gen, dist), MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), rand_val(gen, dist), MPFR_RNDN);
            } else if ((uplo == 'U' && i < j) || (uplo == 'L' && i > j)) {
                double rv = rand_val(gen, dist);
                double iv = rand_val(gen, dist);
                mpfr_set_d(dst.re(i, j), rv, MPFR_RNDN);
                mpfr_set_d(dst.im(i, j), iv, MPFR_RNDN);
                /* Symmetric (not conjugate) */
                mpfr_set_d(dst.re(j, i), rv, MPFR_RNDN);
                mpfr_set_d(dst.im(j, i), iv, MPFR_RNDN);
            }
        }
    }

    update_seed(gen, seed);
}

void gen_mpfr_complex_gbsv_band_matrix(MpfrComplexMatrix &dst, int n,
                                         int kl, int ku, mpfr_prec_t prec,
                                         unsigned *seed)
{
    int ldab = 2 * kl + ku + 1;
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int j = 0; j < n; ++j)
        for (int r = 0; r < ldab; ++r) {
            mpfr_set_d(dst.re(r, j), 0.0, MPFR_RNDN);
            mpfr_set_d(dst.im(r, j), 0.0, MPFR_RNDN);
        }

    for (int j = 0; j < n; ++j) {
        int i_min = std::max(0, j - ku);
        int i_max = std::min(n - 1, j + kl);
        for (int i = i_min; i <= i_max; ++i) {
            int ab_row = kl + ku + i - j;
            mpfr_set_d(dst.re(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
            mpfr_set_d(dst.im(ab_row, j), rand_val(gen, dist), MPFR_RNDN);
        }
        int diag_row = kl + ku;
        double d = mpfr_get_d(dst.re(diag_row, j), MPFR_RNDN);
        mpfr_set_d(dst.re(diag_row, j), d + static_cast<double>(n), MPFR_RNDN);
    }

    update_seed(gen, seed);
}

void gen_mpfr_complex_tridiagonal(MpfrComplexMatrix &dl, MpfrComplexMatrix &d,
                                    MpfrComplexMatrix &du, int n,
                                    mpfr_prec_t prec, unsigned *seed)
{
    auto gen = make_rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (int i = 0; i < n - 1; ++i) {
        mpfr_set_d(dl.re(i, 0), rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(dl.im(i, 0), rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(du.re(i, 0), rand_val(gen, dist), MPFR_RNDN);
        mpfr_set_d(du.im(i, 0), rand_val(gen, dist), MPFR_RNDN);
    }

    for (int i = 0; i < n; ++i) {
        double sum = 0.0;
        if (i > 0) {
            double r = mpfr_get_d(dl.re(i - 1, 0), MPFR_RNDN);
            double im = mpfr_get_d(dl.im(i - 1, 0), MPFR_RNDN);
            sum += std::sqrt(r * r + im * im);
        }
        if (i < n - 1) {
            double r = mpfr_get_d(du.re(i, 0), MPFR_RNDN);
            double im = mpfr_get_d(du.im(i, 0), MPFR_RNDN);
            sum += std::sqrt(r * r + im * im);
        }
        double dv = sum + static_cast<double>(n) + std::abs(rand_val(gen, dist));
        if (gen() & 1) dv = -dv;
        mpfr_set_d(d.re(i, 0), dv, MPFR_RNDN);
        mpfr_set_d(d.im(i, 0), 0.0, MPFR_RNDN);
    }

    update_seed(gen, seed);
}
