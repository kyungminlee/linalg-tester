/* gelsd.cpp -- LAPACK GELSD accuracy tester (least squares via SVD) */

#include "../solvers.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
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

void test_gelsd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    /* Overdetermined: m > n */
    int m = params.m;
    int n = std::min(params.n, params.m);
    if (m <= n) m = n + std::max(n / 2, 1);

    int nrhs = std::min(params.k, 4);
    if (nrhs < 1) nrhs = 1;
    int mn = std::min(m, n);
    int lda = m + params.ld_pad;
    /* B has max(m,n) rows */
    int ldb = std::max(m, n) + params.ld_pad;

    unsigned seed = params.seed;

    /* Generate random m-by-n matrix A */
    void *A = gen_random_array(static_cast<std::size_t>(lda) * n,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save A_orig in MPFR */
    MpfrMatrix A_orig(m, n, prec);
    custom_to_mpfr_mat(A_orig, A, lda, ctx);

    /* Generate random B: max(m,n)-by-nrhs */
    void *B = gen_random_array(static_cast<std::size_t>(ldb) * nrhs,
                               ctx.typesize, ctx.from_mpfr, prec, &seed);

    /* Save B_orig (m-by-nrhs) in MPFR */
    MpfrMatrix B_orig(m, nrhs, prec);
    custom_to_mpfr_mat(B_orig, B, ldb, ctx);

    /* Allocate S (singular values), rcond, rank */
    void *S = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);
    int rank_out = 0;

    /* Set rcond = -1.0 (use machine precision) */
    MpfrScalar rcond_mpfr(prec);
    mpfr_set_d(rcond_mpfr.get(), -1.0, MPFR_RNDN);
    void *rcond = std::malloc(ctx.typesize);
    ctx.from_mpfr(rcond, rcond_mpfr.get(), MPFR_RNDN);

    /* Workspace query */
    int lwork = -1;
    int info = 0;
    /* GELSD also needs integer workspace iwork.
       Size: at least 3*min(m,n)*nlvl + 11*min(m,n) where nlvl = max(0, int(log2(mn/26))+1)
       We query lwork first, then allocate a generous iwork. */
    int nlvl = 0;
    {
        int tmp = mn;
        while (tmp > 25) { tmp /= 2; nlvl++; }
        nlvl = std::max(nlvl, 1);
    }
    int liwork = 3 * mn * nlvl + 11 * mn;
    if (liwork < 1) liwork = 1;
    int *iwork = new int[liwork];

    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, const int *,
        void *, const int *, void *, const int *,
        void *, const void *, int *,
        void *, const int *, int *, int *)>(load_sym(lib, sym));

    void *work_query = std::malloc(ctx.typesize);
    fn(&m, &n, &nrhs, A, &lda, B, &ldb,
       S, rcond, &rank_out,
       work_query, &lwork, iwork, &info);
    lwork = query_lwork(work_query, ctx);
    std::free(work_query);

    /* Allocate workspace and call GELSD */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn(&m, &n, &nrhs, A, &lda, B, &ldb,
       S, rcond, &rank_out,
       work, &lwork, iwork, &info);

    /* Extract solution X from first n rows of B */
    MpfrMatrix X(n, nrhs, prec);
    custom_to_mpfr_mat(X, B, ldb, ctx);

    /* For overdetermined case: check normal equations A^T*(AX-B) = 0 */
    MpfrMatrix AX(m, nrhs, prec);
    mpfr_mat_mul_simple(AX, A_orig, X);

    MpfrMatrix R(m, nrhs, prec);
    mpfr_mat_sub(R, AX, B_orig);

    double norm_A = mpfr_mat_norm1(A_orig);
    double residual = 0.0;

    if (m >= n) {
        MpfrMatrix At(n, m, prec);
        mpfr_mat_transpose(At, A_orig);
        MpfrMatrix AtR(n, nrhs, prec);
        mpfr_mat_mul_simple(AtR, At, R);
        double norm_AtR = mpfr_mat_norm1(AtR);
        double norm_R = mpfr_mat_norm1(R);
        if (norm_A > 0.0 && norm_R > 0.0)
            residual = norm_AtR / (norm_A * norm_R * n * eps);
    } else {
        double norm_R = mpfr_mat_norm1(R);
        double norm_X = mpfr_mat_norm1(X);
        if (norm_A > 0.0 && norm_X > 0.0)
            residual = norm_R / (norm_A * norm_X * m * eps);
    }

    LapackResult lr;
    lr.residual = residual;
    lr.orthogonality = -1.0;
    lr.info = info;

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d nrhs=%d rank=%d", m, n, nrhs, rank_out);
    report_lapack_result("GELSD", params_str, lr, format);

    std::free(A);
    std::free(B);
    std::free(S);
    std::free(rcond);
    std::free(work);
    delete[] iwork;
}
