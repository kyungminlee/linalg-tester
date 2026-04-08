/* cgeev.cpp -- LAPACK complex GEEV accuracy tester (general eigenvalue) */

#include "../eigenvalues.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"
#include "../../core/mpfr_lapack_utils.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*cgeev_fn_t)(
    const char *jobvl, const char *jobvr,
    const int *n, void *A, const int *lda,
    void *w,
    void *VL, const int *ldvl,
    void *VR, const int *ldvr,
    void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobvl_len, std::size_t jobvr_len);

void test_cgeev(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<cgeev_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    std::size_t real_typesize = ctx.typesize / 2;
    int lda = n + params.ld_pad;
    int ldvl = 1, ldvr = n;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_complex_array(lda * n, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);

    MpfrComplexMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_orig, lda, ctx);

    void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *w = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *VR = std::malloc(static_cast<std::size_t>(ldvr) * n * ctx.typesize);
    void *VL_dummy = std::malloc(ctx.typesize); /* not computed */
    void *rwork = std::malloc(static_cast<std::size_t>(2 * n) * real_typesize);

    char jobvl = 'N', jobvr = 'V';
    int info = 0;

    /* Workspace query */
    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    fn(&jobvl, &jobvr, &n, A_buf, &lda, w, VL_dummy, &ldvl, VR, &ldvr,
       work_q, &lwork_q, rwork, &info, (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);

    /* Reload A */
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    fn(&jobvl, &jobvr, &n, A_buf, &lda, w, VL_dummy, &ldvl, VR, &ldvr,
       work, &lwork, rwork, &info, (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("CGEEV", "", lr, format);
    } else {
        /* Read complex eigenvectors */
        MpfrComplexMatrix VR_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(VR_mpfr, VR, ldvr, ctx);

        /* Read complex eigenvalues */
        MpfrComplexMatrix w_mpfr(n, 1, prec);
        const char *wp = static_cast<const char *>(w);
        for (int i = 0; i < n; ++i)
            ctx.to_mpfr_complex(w_mpfr.re(i, 0), w_mpfr.im(i, 0), wp + i * ctx.typesize);

        /* Compute A*VR */
        MpfrComplexMatrix AVR(n, n, prec);
        mpfr_complex_mat_mul_simple(AVR, A_mpfr, VR_mpfr);

        /* Compute VR * diag(w): scale column j by complex w[j] */
        MpfrComplexMatrix VD(n, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < n; ++i)
                mpfr_complex_mul(VD.re(i, j), VD.im(i, j),
                                 VR_mpfr.re(i, j), VR_mpfr.im(i, j),
                                 w_mpfr.re(j, 0), w_mpfr.im(j, 0),
                                 MPFR_RNDN);

        MpfrComplexMatrix R(n, n, prec);
        mpfr_complex_mat_sub(R, AVR, VD);

        double norm_R = mpfr_complex_mat_norm1(R);
        double norm_A = mpfr_complex_mat_norm1(A_mpfr);
        double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;

        LapackResult lr = {residual, -1.0, info};
        report_lapack_result("CGEEV", "", lr, format);
    }

    std::free(rwork); std::free(VL_dummy); std::free(VR);
    std::free(w);
    std::free(A_buf); std::free(A_orig);
}
