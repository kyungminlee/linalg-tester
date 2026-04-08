/* clacpy.cpp -- LAPACK CLACPY/ZLACPY accuracy tester (complex matrix copy) */

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

/* Fortran ABI: LACPY(uplo, m, n, A, lda, B, ldb, uplo_len) */
extern "C" typedef void (*lacpy_fn_t)(
    const char *uplo, const int *m, const int *n,
    const void *A, const int *lda,
    void *B, const int *ldb,
    std::size_t uplo_len
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_clacpy(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<lacpy_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    int ldb = m + params.ld_pad;

    for (char uplo : {'U', 'L', 'A'}) {
        unsigned seed_A = params.seed;

        void *A = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                            ctx.typesize, ctx.from_mpfr_complex,
                                            prec, &seed_A);

        /* Allocate output B (zero-initialized) */
        void *B = std::calloc(static_cast<std::size_t>(ldb) * n, ctx.typesize);

        fn(&uplo, &m, &n, A, &lda, B, &ldb, (std::size_t)1);

        /* MPFR reference: copy the specified triangle of A */
        MpfrComplexMatrix A_mpfr(m, n, prec);
        custom_to_mpfr_complex_mat(A_mpfr, A, lda, ctx);

        MpfrComplexMatrix B_ref(m, n, prec);
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < m; ++i) {
                bool copy_elem = false;
                if (uplo == 'A') {
                    copy_elem = true;
                } else if (uplo == 'U') {
                    copy_elem = (i <= j);
                } else { /* 'L' */
                    copy_elem = (i >= j);
                }

                if (copy_elem) {
                    mpfr_set(B_ref.re(i, j), A_mpfr.re(i, j), MPFR_RNDN);
                    mpfr_set(B_ref.im(i, j), A_mpfr.im(i, j), MPFR_RNDN);
                } else {
                    mpfr_set_d(B_ref.re(i, j), 0.0, MPFR_RNDN);
                    mpfr_set_d(B_ref.im(i, j), 0.0, MPFR_RNDN);
                }
            }
        }

        ErrorResult err = compute_error_complex_matrix(B_ref, B, ldb, ctx);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c m=%d n=%d", uplo, m, n);
        report_result("CLACPY", params_str, err, format);

        std::free(A);
        std::free(B);
    }
}
