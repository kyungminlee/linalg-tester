/* cgees.cpp -- LAPACK complex GEES accuracy tester (Schur decomposition) */

#include "../eigenvalues.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"
#include "../../core/mpfr_lapack_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*cgees_fn_t)(
    const char *jobvs, const char *sort,
    void *select, /* function pointer, unused when sort='N' */
    const int *n, void *A, const int *lda,
    int *sdim,
    void *w,
    void *VS, const int *ldvs,
    void *work, const int *lwork,
    void *rwork, int *bwork, int *info,
    std::size_t jobvs_len, std::size_t sort_len);

void test_cgees(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<cgees_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    std::size_t real_typesize = ctx.typesize / 2;
    int lda = n + params.ld_pad;
    int ldvs = n;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_complex_array(lda * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);

    MpfrComplexMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_orig, lda, ctx);

    void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *w = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *VS = std::malloc(static_cast<std::size_t>(ldvs) * n * ctx.typesize);
    void *rwork = std::malloc(static_cast<std::size_t>(n) * real_typesize);
    int *bwork = new int[n];
    int sdim = 0, info = 0;

    char jobvs = 'V', sort = 'N';

    /* Workspace query */
    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    fn(&jobvs, &sort, nullptr, &n, A_buf, &lda, &sdim,
       w, VS, &ldvs, work_q, &lwork_q, rwork, bwork, &info,
       (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);

    /* Reload A */
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    fn(&jobvs, &sort, nullptr, &n, A_buf, &lda, &sdim,
       w, VS, &ldvs, work, &lwork, rwork, bwork, &info,
       (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("CGEES", "", lr, format);
    } else {
        /* A_buf now contains upper triangular Schur form T, VS has unitary Schur vectors */
        MpfrComplexMatrix T(n, n, prec);
        custom_to_mpfr_complex_mat(T, A_buf, lda, ctx);

        MpfrComplexMatrix V(n, n, prec);
        custom_to_mpfr_complex_mat(V, VS, ldvs, ctx);

        /* Residual: ||A - V*T*V^H||_1 / (||A||_1 * n * eps) */
        MpfrComplexMatrix VT_prod(n, n, prec);
        mpfr_complex_mat_mul_simple(VT_prod, V, T);

        MpfrComplexMatrix Vh(n, n, prec);
        mpfr_complex_mat_adjoint(Vh, V);

        MpfrComplexMatrix VTVh(n, n, prec);
        mpfr_complex_mat_mul_simple(VTVh, VT_prod, Vh);

        MpfrComplexMatrix R(n, n, prec);
        mpfr_complex_mat_sub(R, A_mpfr, VTVh);

        double norm_R = mpfr_complex_mat_norm1(R);
        double norm_A = mpfr_complex_mat_norm1(A_mpfr);
        double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;
        double orth = mpfr_complex_orthogonality(V, eps);

        LapackResult lr = {residual, orth, info};
        report_lapack_result("CGEES", "", lr, format);
    }

    delete[] bwork; std::free(rwork);
    std::free(w); std::free(VS);
    std::free(A_buf); std::free(A_orig);
}
