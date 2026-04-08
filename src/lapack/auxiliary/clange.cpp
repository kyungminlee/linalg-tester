/* clange.cpp -- LAPACK CLANGE/ZLANGE accuracy tester (complex matrix norm) */

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
/* Helper: complex matrix max-element norm: max |A(i,j)|               */
/* ------------------------------------------------------------------ */

static double mpfr_complex_mat_normM(const MpfrComplexMatrix &A)
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
/* Helper: complex matrix infinity norm: max row sum of |A(i,j)|       */
/* ------------------------------------------------------------------ */

static double mpfr_complex_mat_normI(const MpfrComplexMatrix &A)
{
    int m = A.rows(), n = A.cols();
    mpfr_prec_t prec = mpfr_get_prec(A.re(0, 0));
    MpfrScalar row_sum(prec), abs_val(prec), max_norm(prec);
    mpfr_set_d(max_norm.get(), 0.0, MPFR_RNDN);

    for (int i = 0; i < m; ++i) {
        mpfr_set_d(row_sum.get(), 0.0, MPFR_RNDN);
        for (int j = 0; j < n; ++j) {
            mpfr_complex_abs(abs_val.get(),
                             A.re(i, j), A.im(i, j),
                             MPFR_RNDN);
            mpfr_add(row_sum.get(), row_sum.get(), abs_val.get(), MPFR_RNDN);
        }
        if (mpfr_cmp(row_sum.get(), max_norm.get()) > 0)
            mpfr_set(max_norm.get(), row_sum.get(), MPFR_RNDN);
    }
    return mpfr_get_d(max_norm.get(), MPFR_RNDN);
}

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_clange(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    std::size_t real_typesize = ctx.typesize / 2;

    for (char norm_type : {'M', '1', 'I', 'F'}) {
        unsigned seed_A = params.seed;

        void *A = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                            ctx.typesize, ctx.from_mpfr_complex,
                                            prec, &seed_A);

        /* Work array: only needed for 'I' norm (real, size m) */
        void *rwork = std::calloc(static_cast<std::size_t>(m), real_typesize);

        /* Convert input to MPFR */
        MpfrComplexMatrix A_mpfr(m, n, prec);
        custom_to_mpfr_complex_mat(A_mpfr, A, lda, ctx);

        /* MPFR reference */
        double ref_val = 0.0;
        switch (norm_type) {
            case 'M': ref_val = mpfr_complex_mat_normM(A_mpfr); break;
            case '1': ref_val = mpfr_complex_mat_norm1(A_mpfr); break;
            case 'I': ref_val = mpfr_complex_mat_normI(A_mpfr); break;
            case 'F': ref_val = mpfr_complex_mat_normF(A_mpfr); break;
        }

        MpfrScalar ref_scalar(prec);
        mpfr_set_d(ref_scalar.get(), ref_val, MPFR_RNDN);

        /* Call the library routine -- returns a REAL scalar value.
           ZLANGE interface: double ZLANGE(norm, m, n, A, lda, rwork, norm_len)
           For complex double (typesize=16), the return type is double (real_typesize=8). */
        void *result_buf = std::calloc(1, real_typesize);

        if (real_typesize == 4) {
            auto *fn = reinterpret_cast<float (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            float result = fn(&norm_type, &m, &n, A, &lda, rwork,
                              (std::size_t)1);
            std::memcpy(result_buf, &result, sizeof(float));
        } else if (real_typesize == 8) {
            auto *fn = reinterpret_cast<double (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            double result = fn(&norm_type, &m, &n, A, &lda, rwork,
                               (std::size_t)1);
            std::memcpy(result_buf, &result, sizeof(double));
        } else if (real_typesize == 16) {
            auto *fn = reinterpret_cast<long double (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            long double result = fn(&norm_type, &m, &n, A, &lda, rwork,
                                    (std::size_t)1);
            std::memcpy(result_buf, &result, real_typesize);
        } else {
            std::fprintf(stderr, "CLANGE: unsupported real_typesize %zu, skipping\n",
                         real_typesize);
            std::free(A);
            std::free(rwork);
            std::free(result_buf);
            return;
        }

        /* Use a TesterCtx-like approach for real scalar comparison.
           Build a temporary real ctx for compute_error_scalar. */
        TesterCtx real_ctx = ctx;
        real_ctx.typesize = real_typesize;
        /* from_mpfr and to_mpfr already handle the real component */

        ErrorResult err = compute_error_scalar(ref_scalar, result_buf, real_ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "norm=%c m=%d n=%d", norm_type, m, n);
        report_result("CLANGE", params_str, err, format);

        std::free(A);
        std::free(rwork);
        std::free(result_buf);
    }
}
