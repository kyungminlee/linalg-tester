/* cgelsd.cpp -- LAPACK CGELSD/ZGELSD accuracy tester (complex least squares via SVD) */

#include "../solvers.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_cgelsd(const TesterCtx &ctx, void *lib, const char *sym,
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

    /* Real typesize for singular values S and rwork */
    std::size_t real_typesize = ctx.typesize / 2;

    unsigned seed = params.seed;

    /* Generate random complex m-by-n matrix A */
    void *A = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                        ctx.typesize, ctx.from_mpfr_complex,
                                        prec, &seed);

    /* Save A_orig in MPFR */
    MpfrComplexMatrix A_orig(m, n, prec);
    custom_to_mpfr_complex_mat(A_orig, A, lda, ctx);

    /* Generate random complex B: max(m,n)-by-nrhs */
    void *B = gen_random_complex_array(static_cast<std::size_t>(ldb) * nrhs,
                                        ctx.typesize, ctx.from_mpfr_complex,
                                        prec, &seed);

    /* Save B_orig (m-by-nrhs) in MPFR */
    MpfrComplexMatrix B_orig(m, nrhs, prec);
    custom_to_mpfr_complex_mat(B_orig, B, ldb, ctx);

    /* Allocate S (singular values -- REAL), rcond, rank */
    void *S = std::malloc(static_cast<std::size_t>(mn) * real_typesize);
    int rank_out = 0;

    /* Set rcond = -1.0 (use machine precision) -- rcond is REAL */
    MpfrScalar rcond_mpfr(prec);
    mpfr_set_d(rcond_mpfr.get(), -1.0, MPFR_RNDN);
    void *rcond = std::malloc(real_typesize);
    ctx.from_mpfr(rcond, rcond_mpfr.get(), MPFR_RNDN);

    /* CGELSD/ZGELSD needs integer workspace iwork and real workspace rwork.
       iwork size: at least 3*min(m,n)*nlvl + 11*min(m,n) */
    int nlvl = 0;
    {
        int tmp = mn;
        while (tmp > 25) { tmp /= 2; nlvl++; }
        nlvl = std::max(nlvl, 1);
    }
    int liwork = 3 * mn * nlvl + 11 * mn;
    if (liwork < 1) liwork = 1;
    int *iwork = new int[liwork];

    /* rwork size for complex GELSD: at minimum 5*min(m,n) real elements,
       but we allocate generously. Exact formula from LAPACK docs:
       10*mn + 2*mn*SMLSIZ + 8*mn*nlvl + 3*SMLSIZ*nrhs + max((SMLSIZ+1)^2, ...)
       We use a safe overestimate. */
    int lrwork = std::max(1, 10 * mn + 2 * mn * 25 + 8 * mn * nlvl +
                              3 * 25 * nrhs + (25 + 1) * (25 + 1));
    void *rwork = std::malloc(static_cast<std::size_t>(lrwork) * real_typesize);

    /* CGELSD/ZGELSD signature:
       void xGELSD(int *m, int *n, int *nrhs,
                    T *A, int *lda, T *B, int *ldb,
                    REAL *S, REAL *rcond, int *rank,
                    T *work, int *lwork, REAL *rwork, int *iwork, int *info) */
    auto *fn = reinterpret_cast<void (*)(
        const int *, const int *, const int *,
        void *, const int *, void *, const int *,
        void *, const void *, int *,
        void *, const int *, void *, int *, int *)>(load_sym(lib, sym));

    /* Workspace query */
    int lwork = -1;
    int info = 0;
    void *work_query = std::malloc(ctx.typesize);
    fn(&m, &n, &nrhs, A, &lda, B, &ldb,
       S, rcond, &rank_out,
       work_query, &lwork, rwork, iwork, &info);
    lwork = query_lwork_complex(work_query, ctx);
    std::free(work_query);

    /* Allocate workspace and call CGELSD/ZGELSD */
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    info = 0;
    fn(&m, &n, &nrhs, A, &lda, B, &ldb,
       S, rcond, &rank_out,
       work, &lwork, rwork, iwork, &info);

    /* Extract solution X from first n rows of B */
    MpfrComplexMatrix X(n, nrhs, prec);
    custom_to_mpfr_complex_mat(X, B, ldb, ctx);

    /* For overdetermined case: check normal equations A^H*(AX-B) = 0 */
    MpfrComplexMatrix AX(m, nrhs, prec);
    mpfr_complex_mat_mul_simple(AX, A_orig, X);

    MpfrComplexMatrix R(m, nrhs, prec);
    mpfr_complex_mat_sub(R, AX, B_orig);

    double norm_A = mpfr_complex_mat_norm1(A_orig);
    double residual = 0.0;

    if (m >= n) {
        MpfrComplexMatrix Ah(n, m, prec);
        mpfr_complex_mat_adjoint(Ah, A_orig);
        MpfrComplexMatrix AhR(n, nrhs, prec);
        mpfr_complex_mat_mul_simple(AhR, Ah, R);
        double norm_AhR = mpfr_complex_mat_norm1(AhR);
        double norm_R = mpfr_complex_mat_norm1(R);
        if (norm_A > 0.0 && norm_R > 0.0)
            residual = norm_AhR / (norm_A * norm_R * n * eps);
    } else {
        double norm_R = mpfr_complex_mat_norm1(R);
        double norm_X = mpfr_complex_mat_norm1(X);
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
    report_lapack_result("CGELSD", params_str, lr, format);

    std::free(A);
    std::free(B);
    std::free(S);
    std::free(rcond);
    std::free(work);
    std::free(rwork);
    delete[] iwork;
}
