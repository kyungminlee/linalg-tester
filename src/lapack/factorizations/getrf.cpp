/* getrf.cpp -- LAPACK GETRF accuracy tester (LU factorization with partial pivoting) */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_getrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int m = params.m, n = params.n;
    int lda = m + params.ld_pad;
    int mn = std::min(m, n);

    unsigned seed_A = params.seed;

    /* Generate random m-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(m, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Allocate IPIV */
    int *ipiv = new int[mn];

    /* Call GETRF: void GETRF(int *m, int *n, T *A, int *lda, int *ipiv, int *info) */
    int info = 0;
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, void *, const int *,
        int *, int *)>(load_sym(lib, sym));
    fn(&m, &n, A, &lda, ipiv, &info);

    /* Convert factored A to MPFR */
    MpfrMatrix A_fact(m, n, prec);
    custom_to_mpfr_mat(A_fact, A, lda, ctx);

    /* Extract L (m-by-mn, unit lower) and U (mn-by-n, upper) */
    MpfrMatrix L(m, mn, prec);
    MpfrMatrix U(mn, n, prec);
    mpfr_extract_LU(A_fact, L, U);

    /* Compute L*U */
    MpfrMatrix LU(m, n, prec);
    mpfr_mat_mul_simple(LU, L, U);

    /* Apply inverse permutation: P^{-1} * (L*U) should equal A_orig.
       Equivalently, apply IPIV in reverse to LU, then compare with A_orig.
       Or: apply IPIV forward to A_orig, compare with L*U.
       Standard: compute P*L*U by applying IPIV in reverse order to LU rows. */
    mpfr_apply_ipiv_rows(LU, ipiv, mn, /*forward=*/false);

    /* Residual: ||A - P*L*U||_1 / (||A||_1 * n * eps) */
    MpfrMatrix Resid(m, n, prec);
    mpfr_mat_sub(Resid, A_orig, LU);
    double norm_resid = mpfr_mat_norm1(Resid);
    double norm_A = mpfr_mat_norm1(A_orig);

    double residual = (norm_A > 0.0) ? norm_resid / (norm_A * n * eps) : 0.0;

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = -1.0;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    report_lapack_result("GETRF", params_str, lr, format);

    std::free(A);
    delete[] ipiv;
}
