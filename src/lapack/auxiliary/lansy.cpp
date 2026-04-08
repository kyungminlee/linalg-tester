/* lansy.cpp -- LAPACK LANSY accuracy tester (symmetric matrix norm) */

#include "../auxiliary.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Helper: expand symmetric MPFR matrix from one triangle             */
/* ------------------------------------------------------------------ */

static void mpfr_expand_symmetric(MpfrMatrix &full, const MpfrMatrix &A, char uplo)
{
    int n = A.rows();
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            if (uplo == 'U') {
                if (i <= j)
                    mpfr_set(full.at(i, j), A.at(i, j), MPFR_RNDN);
                else
                    mpfr_set(full.at(i, j), A.at(j, i), MPFR_RNDN);
            } else {
                if (i >= j)
                    mpfr_set(full.at(i, j), A.at(i, j), MPFR_RNDN);
                else
                    mpfr_set(full.at(i, j), A.at(j, i), MPFR_RNDN);
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_lansy(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;

    int n = params.n;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        for (char norm_type : {'M', '1', 'I', 'F'}) {
            unsigned seed_A = params.seed;

            void *A = gen_symmetric_array(n, uplo, ctx.typesize, ctx.from_mpfr,
                                          prec, &seed_A);

            /* For ld_pad > 0, we need to copy into a padded array */
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

            /* Work array: needed for 'I' norm (size n) */
            void *work = std::calloc(static_cast<std::size_t>(n), ctx.typesize);

            /* Convert input to MPFR */
            MpfrMatrix A_mpfr(n, n, prec);
            custom_to_mpfr_mat(A_mpfr, A_padded, lda, ctx);

            /* MPFR reference: expand to full symmetric, then compute norm */
            double ref_val = 0.0;
            if (norm_type == '1' || norm_type == 'I') {
                /* For symmetric matrices, 1-norm == inf-norm */
                ref_val = mpfr_symmat_norm1(A_mpfr, uplo);
            } else {
                /* Expand symmetric matrix to full for M and F norms */
                MpfrMatrix full(n, n, prec);
                mpfr_expand_symmetric(full, A_mpfr, uplo);
                if (norm_type == 'M')
                    ref_val = mpfr_mat_normM(full);
                else /* 'F' */
                    ref_val = mpfr_mat_normF(full);
            }

            MpfrScalar ref_scalar(prec);
            mpfr_set_d(ref_scalar.get(), ref_val, MPFR_RNDN);

            /* Call the library routine -- returns a scalar value */
            void *result_buf = std::calloc(1, ctx.typesize);

            if (ctx.typesize == 4) {
                auto *fn = reinterpret_cast<float (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                float result = fn(&norm_type, &uplo, &n, A_padded, &lda, work,
                                  (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, sizeof(float));
            } else if (ctx.typesize == 8) {
                auto *fn = reinterpret_cast<double (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                double result = fn(&norm_type, &uplo, &n, A_padded, &lda, work,
                                   (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, sizeof(double));
            } else if (ctx.typesize == 16) {
                auto *fn = reinterpret_cast<long double (*)(
                    const char *, const char *, const int *,
                    const void *, const int *, void *,
                    std::size_t, std::size_t)>(load_sym(lib, sym));
                long double result = fn(&norm_type, &uplo, &n, A_padded, &lda, work,
                                        (std::size_t)1, (std::size_t)1);
                std::memcpy(result_buf, &result, ctx.typesize);
            } else {
                std::fprintf(stderr, "LANSY: unsupported typesize %zu, skipping\n",
                             ctx.typesize);
                std::free(A);
                if (A_padded != A) std::free(A_padded);
                std::free(work);
                std::free(result_buf);
                return;
            }

            ErrorResult err = compute_error_scalar(ref_scalar, result_buf, ctx);

            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "norm=%c uplo=%c n=%d", norm_type, uplo, n);
            report_result("LANSY", params_str, err, format);

            std::free(A);
            if (A_padded != A) std::free(A_padded);
            std::free(work);
            std::free(result_buf);
        }
    }
}
