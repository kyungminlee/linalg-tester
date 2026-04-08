/* lange.cpp -- LAPACK LANGE accuracy tester (matrix norm) */

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
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_lange(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    for (char norm_type : {'M', '1', 'I', 'F'}) {
        unsigned seed_A = params.seed;

        void *A = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

        /* Work array: only needed for 'I' norm (size m), allocate m always */
        void *work = std::calloc(static_cast<std::size_t>(m), ctx.typesize);

        /* Convert input to MPFR */
        MpfrMatrix A_mpfr(m, n, prec);
        custom_to_mpfr_mat(A_mpfr, A, lda, ctx);

        /* MPFR reference */
        double ref_val = 0.0;
        switch (norm_type) {
            case 'M': ref_val = mpfr_mat_normM(A_mpfr); break;
            case '1': ref_val = mpfr_mat_norm1(A_mpfr); break;
            case 'I': ref_val = mpfr_mat_normI(A_mpfr); break;
            case 'F': ref_val = mpfr_mat_normF(A_mpfr); break;
        }

        MpfrScalar ref_scalar(prec);
        mpfr_set_d(ref_scalar.get(), ref_val, MPFR_RNDN);

        /* Call the library routine -- returns a scalar value */
        void *result_buf = std::calloc(1, ctx.typesize);

        if (ctx.typesize == 4) {
            auto *fn = reinterpret_cast<float (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            float result = fn(&norm_type, &m, &n, A, &lda, work,
                              (std::size_t)1);
            std::memcpy(result_buf, &result, sizeof(float));
        } else if (ctx.typesize == 8) {
            auto *fn = reinterpret_cast<double (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            double result = fn(&norm_type, &m, &n, A, &lda, work,
                               (std::size_t)1);
            std::memcpy(result_buf, &result, sizeof(double));
        } else if (ctx.typesize == 16) {
            auto *fn = reinterpret_cast<long double (*)(
                const char *, const int *, const int *,
                const void *, const int *, void *,
                std::size_t)>(load_sym(lib, sym));
            long double result = fn(&norm_type, &m, &n, A, &lda, work,
                                    (std::size_t)1);
            std::memcpy(result_buf, &result, ctx.typesize);
        } else {
            std::fprintf(stderr, "LANGE: unsupported typesize %zu, skipping\n",
                         ctx.typesize);
            std::free(A);
            std::free(work);
            std::free(result_buf);
            return;
        }

        ErrorResult err = compute_error_scalar(ref_scalar, result_buf, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "norm=%c m=%d n=%d", norm_type, m, n);
        report_result("LANGE", params_str, err, format);

        std::free(A);
        std::free(work);
        std::free(result_buf);
    }
}
