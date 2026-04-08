/* gesvd.cpp -- LAPACK GESVD accuracy tester (SVD) */

#include "../eigenvalues.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*gesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n, void *A, const int *lda,
    void *S, void *U, const int *ldu,
    void *VT, const int *ldvt,
    void *work, const int *lwork, int *info,
    std::size_t jobu_len, std::size_t jobvt_len);

void test_gesvd(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<gesvd_fn_t>(load_sym(lib, sym));
    int m = params.m, n = params.n;
    int mn = std::min(m, n);
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = m + params.ld_pad;
    int ldu = m, ldvt = n;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    MpfrMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_orig, lda, ctx);

    void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *S = std::malloc(static_cast<std::size_t>(mn) * ctx.typesize);
    void *U = std::malloc(static_cast<std::size_t>(ldu) * m * ctx.typesize);
    void *VT = std::malloc(static_cast<std::size_t>(ldvt) * n * ctx.typesize);

    char jobu = 'A', jobvt = 'A';
    int info = 0;

    /* Workspace query */
    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    fn(&jobu, &jobvt, &m, &n, A_buf, &lda, S, U, &ldu, VT, &ldvt,
       work_q, &lwork_q, &info, (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork(work_q, ctx);
    std::free(work_q);

    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    fn(&jobu, &jobvt, &m, &n, A_buf, &lda, S, U, &ldu, VT, &ldvt,
       work, &lwork, &info, (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("GESVD", "", lr, format);
    } else {
        MpfrMatrix U_mpfr(m, m, prec);
        custom_to_mpfr_mat(U_mpfr, U, ldu, ctx);

        MpfrMatrix VT_mpfr(n, n, prec);
        custom_to_mpfr_mat(VT_mpfr, VT, ldvt, ctx);

        MpfrMatrix S_mpfr(mn, 1, prec);
        const char *sp = static_cast<const char *>(S);
        for (int i = 0; i < mn; ++i)
            ctx.to_mpfr(S_mpfr.at(i, 0), sp + i * ctx.typesize);

        /* Residual: ||A - U*diag(S)*VT||_1 / (||A||_1 * min(m,n) * eps) */
        /* Build U * diag(S): scale first mn columns of U by S */
        MpfrMatrix US(m, mn, prec);
        for (int j = 0; j < mn; ++j)
            for (int i = 0; i < m; ++i)
                mpfr_mul(US.at(i, j), U_mpfr.at(i, j), S_mpfr.at(j, 0), MPFR_RNDN);

        /* Extract first mn rows of VT */
        MpfrMatrix VT_mn(mn, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < mn; ++i)
                mpfr_set(VT_mn.at(i, j), VT_mpfr.at(i, j), MPFR_RNDN);

        MpfrMatrix USVT(m, n, prec);
        mpfr_mat_mul_simple(USVT, US, VT_mn);

        MpfrMatrix R(m, n, prec);
        mpfr_mat_sub(R, A_mpfr, USVT);

        double norm_R = mpfr_mat_norm1(R);
        double norm_A = mpfr_mat_norm1(A_mpfr);
        double residual = (norm_A > 0.0) ? norm_R / (norm_A * mn * eps) : 0.0;

        /* Orthogonality of U (first mn columns) */
        MpfrMatrix U_mn(m, mn, prec);
        for (int j = 0; j < mn; ++j)
            for (int i = 0; i < m; ++i)
                mpfr_set(U_mn.at(i, j), U_mpfr.at(i, j), MPFR_RNDN);
        double orth_u = mpfr_orthogonality(U_mn, eps);

        /* Use max of residual-related metrics */
        LapackResult lr = {residual, orth_u, info};
        report_lapack_result("GESVD", "", lr, format);
    }

    std::free(S); std::free(U); std::free(VT);
    std::free(A_buf); std::free(A_orig);
}
