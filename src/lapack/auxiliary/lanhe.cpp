/* lanhe.cpp -- LAPACK CLANHE/ZLANHE accuracy tester (Hermitian matrix norm) */

#include "../auxiliary.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Helper: expand Hermitian MPFR complex matrix from one triangle      */
/* ------------------------------------------------------------------ */

static void mpfr_expand_hermitian(MpfrComplexMatrix &full,
                                   const MpfrComplexMatrix &A, char uplo)
{
    int n = A.rows();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (uplo == 'U') {
                if (i <= j) {
                    mpfr_set(full.re(i, j), A.re(i, j), MPFR_RNDN);
                    mpfr_set(full.im(i, j), A.im(i, j), MPFR_RNDN);
                } else {
                    /* full(i,j) = conj(A(j,i)) */
                    mpfr_complex_conj(full.re(i, j), full.im(i, j),
                                      A.re(j, i), A.im(j, i),
                                      MPFR_RNDN);
                }
            } else {
                if (i >= j) {
                    mpfr_set(full.re(i, j), A.re(i, j), MPFR_RNDN);
                    mpfr_set(full.im(i, j), A.im(i, j), MPFR_RNDN);
                } else {
                    /* full(i,j) = conj(A(j,i)) */
                    mpfr_complex_conj(full.re(i, j), full.im(i, j),
                                      A.re(j, i), A.im(j, i),
                                      MPFR_RNDN);
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Helper: complex matrix max-element norm: max |A(i,j)|               */
/* ------------------------------------------------------------------ */

static double mpfr_complex_mat_normM_local(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar abs_val(prec), max_val(prec);
    mpfr_set_d(max_val.get(), 0.0, MPFR_RNDN);

    for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
            mpfr_complex_abs(abs_val.get(),
                             A.re(i, j), A.im(i, j),
                             MPFR_RNDN);
            if (mpfr_cmp(abs_val.get(), max_val.get()) > 0)
                mpfr_set(max_val.get(), abs_val.get(), MPFR_RNDN);
        }
    return mpfr_get_d(max_val.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_lanhe(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;

    int n = params.n;
    int lda = n + params.ld_pad;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        for (char norm_type : {'M', '1', 'I', 'F'}) {
            unsigned seed_A = params.seed;

            void *A = gen_hermitian_array(n, uplo, ctx.typesize,
                                          ctx.from_mpfr_complex, prec, &seed_A);

            /* For ld_pad > 0, copy into padded array */
            void *A_padded = A;
            if (params.ld_pad > 0) {
                A_padded = std::calloc(static_cast<std::size_t>(lda) * n, ctx.typesize);
                const char *src = static_cast<const char *>(A);
                char *dst = static_cast<char *>(A_padded);
                for (int j = 0; j < n; ++j)
                    std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                                src + static_cast<std::size_t>(j) * n * ctx.typesize,
                                static_cast<std::size_t>(n) * ctx.typesize);
            }

            /* Work array: needed for 'I' norm (real, size n) */
            void *rwork = std::calloc(static_cast<std::size_t>(n), real_typesize);

            /* Convert input to MPFR */
            MpfrComplexMatrix A_mpfr(n, n, prec);
            custom_to_mpfr_complex_mat(A_mpfr, A_padded, lda, ctx);

            /* MPFR reference: compute norm */
            double ref_val = 0.0;
            if (norm_type == '1' || norm_type == 'I') {
                /* For Hermitian matrices, 1-norm == inf-norm */
                ref_val = mpfr_hermat_norm1(A_mpfr, uplo);
            } else {
                /* Expand Hermitian matrix to full for M and F norms */
                MpfrComplexMatrix full(n, n, prec);
                mpfr_expand_hermitian(full, A_mpfr, uplo);
                if (norm_type == 'M')
                    ref_val = mpfr_complex_mat_normM_local(full);
                else /* 'F' */
                    ref_val = mpfr_complex_mat_normF(full);
            }

            MpfrScalar ref_scalar(prec);
            mpfr_set_d(ref_scalar.get(), ref_val, MPFR_RNDN);

            /* Call the library routine -- returns a REAL scalar value.
               ZLANHE interface: double ZLANHE(norm, uplo, n, A, lda, rwork, norm_len, uplo_len) */
            void *result_buf = std::calloc(1, real_typesize);

            if (real_typesize == 4) {
                auto *fn = reinterpret_cast<float (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                float result = fn(&norm_type, &uplo, &n, A_padded, &lda, rwork,
                                  (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, sizeof(float));
            } else if (real_typesize == 8) {
                auto *fn = reinterpret_cast<double (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                double result = fn(&norm_type, &uplo, &n, A_padded, &lda, rwork,
                                   (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, sizeof(double));
            } else if (real_typesize == 16) {
                auto *fn = reinterpret_cast<long double (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                long double result = fn(&norm_type, &uplo, &n, A_padded, &lda, rwork,
                                        (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, real_typesize);
            } else {
                std::fprintf(stderr, "LANHE: unsupported real_typesize %zu, skipping\n",
                             real_typesize);
                std::free(A);
                if (A_padded != A) std::free(A_padded);
                std::free(rwork);
                std::free(result_buf);
                return;
            }

            /* Build a real TesterCtx for error comparison */
            TesterCtx real_ctx = ctx;
            real_ctx.typesize = real_typesize;

            ErrorResult err = compute_error_scalar(ref_scalar, result_buf, real_ctx);

            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "norm=%c uplo=%c n=%d", norm_type, uplo, n);
            report_result("LANHE", params_str, err, format);

            std::free(A);
            if (A_padded != A) std::free(A_padded);
            std::free(rwork);
            std::free(result_buf);
        }
    }
}
