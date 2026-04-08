/* laswp.cpp -- LAPACK LASWP accuracy tester (row interchange) */

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

/* Fortran ABI: LASWP(n, A, lda, k1, k2, ipiv, incx) */
extern "C" typedef void (*laswp_fn_t)(
    const int *n, void *A, const int *lda,
    const int *k1, const int *k2,
    const int *ipiv, const int *incx
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_laswp(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<laswp_fn_t>(load_sym(lib, sym));
    mpfr_prec_t prec = ctx.prec;

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;
    unsigned seed_ipiv = params.seed + 10;

    void *A = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* Generate random IPIV array with values in [1, m] */
    int *ipiv = static_cast<int *>(std::malloc(static_cast<std::size_t>(m) * sizeof(int)));
    {
        unsigned s = seed_ipiv;
        for (int i = 0; i < m; ++i) {
            /* Simple LCG for reproducible random pivots */
            s = s * 1103515245u + 12345u;
            ipiv[i] = 1 + static_cast<int>((s >> 16) % static_cast<unsigned>(m));
        }
    }

    int k1 = 1;   /* 1-based, apply from row 1 */
    int k2 = m;    /* to row m */
    int incx = 1;  /* forward direction */

    /* Convert input to MPFR before modification */
    MpfrMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_mat(A_mpfr, A, lda, ctx);

    /* Apply MPFR reference permutation (k1 and k2 are 1-based) */
    mpfr_apply_ipiv_rows(A_mpfr, ipiv, m, true);

    /* Call the library routine */
    fn(&n, A, &lda, &k1, &k2, ipiv, &incx);

    ErrorResult err = compute_error_matrix(A_mpfr, A, lda, ctx);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d k1=%d k2=%d", m, n, k1, k2);
    report_result("LASWP", params_str, err, format);

    std::free(A);
    std::free(ipiv);
}
