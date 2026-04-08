/* gees.cpp -- LAPACK GEES accuracy tester (Schur decomposition) */

#include "../eigenvalues.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*gees_fn_t)(
    const char *jobvs, const char *sort,
    void *select, /* function pointer, unused when sort='N' */
    const int *n, void *A, const int *lda,
    int *sdim,
    void *wr, void *wi,
    void *VS, const int *ldvs,
    void *work, const int *lwork,
    int *bwork, int *info,
    std::size_t jobvs_len, std::size_t sort_len);

void test_gees(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<gees_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = n + params.ld_pad;
    int ldvs = n;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    MpfrMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_orig, lda, ctx);

    void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *wr = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *wi = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *VS = std::malloc(static_cast<std::size_t>(ldvs) * n * ctx.typesize);
    int *bwork = new int[n];
    int sdim = 0, info = 0;

    char jobvs = 'V', sort = 'N';

    /* Workspace query */
    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    fn(&jobvs, &sort, nullptr, &n, A_buf, &lda, &sdim,
       wr, wi, VS, &ldvs, work_q, &lwork_q, bwork, &info,
       (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork(work_q, ctx);
    std::free(work_q);

    /* Reload A */
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    fn(&jobvs, &sort, nullptr, &n, A_buf, &lda, &sdim,
       wr, wi, VS, &ldvs, work, &lwork, bwork, &info,
       (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("GEES", "", lr, format);
    } else {
        /* A_buf now contains Schur form T, VS has orthogonal Schur vectors */
        MpfrMatrix T(n, n, prec);
        custom_to_mpfr_mat(T, A_buf, lda, ctx);

        MpfrMatrix V(n, n, prec);
        custom_to_mpfr_mat(V, VS, ldvs, ctx);

        /* Residual: ||A - V*T*V^T||_1 / (||A||_1 * n * eps) */
        MpfrMatrix VT(n, n, prec);
        mpfr_mat_mul_simple(VT, V, T);

        MpfrMatrix Vt(n, n, prec);
        mpfr_mat_transpose(Vt, V);

        MpfrMatrix VTVt(n, n, prec);
        mpfr_mat_mul_simple(VTVt, VT, Vt);

        MpfrMatrix R(n, n, prec);
        mpfr_mat_sub(R, A_mpfr, VTVt);

        double norm_R = mpfr_mat_norm1(R);
        double norm_A = mpfr_mat_norm1(A_mpfr);
        double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;
        double orth = mpfr_orthogonality(V, eps);

        LapackResult lr = {residual, orth, info};
        report_lapack_result("GEES", "", lr, format);
    }

    delete[] bwork;
    std::free(wr); std::free(wi); std::free(VS);
    std::free(A_buf); std::free(A_orig);
}
