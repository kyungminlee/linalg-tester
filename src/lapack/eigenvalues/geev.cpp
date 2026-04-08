/* geev.cpp -- LAPACK GEEV accuracy tester (general eigenvalue) */

#include "../eigenvalues.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*geev_fn_t)(
    const char *jobvl, const char *jobvr,
    const int *n, void *A, const int *lda,
    void *wr, void *wi,
    void *VL, const int *ldvl,
    void *VR, const int *ldvr,
    void *work, const int *lwork, int *info,
    std::size_t jobvl_len, std::size_t jobvr_len);

void test_geev(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<geev_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = n + params.ld_pad;
    int ldvl = 1, ldvr = n;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    MpfrMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_orig, lda, ctx);

    void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *wr = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *wi = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
    void *VR = std::malloc(static_cast<std::size_t>(ldvr) * n * ctx.typesize);
    void *VL_dummy = std::malloc(ctx.typesize); /* not computed */

    char jobvl = 'N', jobvr = 'V';
    int info = 0;

    /* Workspace query */
    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    fn(&jobvl, &jobvr, &n, A_buf, &lda, wr, wi, VL_dummy, &ldvl, VR, &ldvr,
       work_q, &lwork_q, &info, (std::size_t)1, (std::size_t)1);
    int lwork = query_lwork(work_q, ctx);
    std::free(work_q);

    /* Reload A */
    std::memcpy(A_buf, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    fn(&jobvl, &jobvr, &n, A_buf, &lda, wr, wi, VL_dummy, &ldvl, VR, &ldvr,
       work, &lwork, &info, (std::size_t)1, (std::size_t)1);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("GEEV", "", lr, format);
    } else {
        /* Read eigenvalues and eigenvectors */
        MpfrMatrix VR_mpfr(n, n, prec);
        custom_to_mpfr_mat(VR_mpfr, VR, ldvr, ctx);

        MpfrMatrix wr_mpfr(n, 1, prec), wi_mpfr(n, 1, prec);
        const char *wrp = static_cast<const char *>(wr);
        const char *wip = static_cast<const char *>(wi);
        for (int i = 0; i < n; ++i) {
            ctx.to_mpfr(wr_mpfr.at(i, 0), wrp + i * ctx.typesize);
            ctx.to_mpfr(wi_mpfr.at(i, 0), wip + i * ctx.typesize);
        }

        /* Compute A*VR */
        MpfrMatrix AVR(n, n, prec);
        mpfr_mat_mul_simple(AVR, A_mpfr, VR_mpfr);

        /* Compute VR * diag(lambda) accounting for complex eigenvalues.
           For real eigenvalue j: VR(:,j)*wr(j)
           For complex pair j,j+1:
             col j:   wr(j)*VR(:,j) - wi(j)*VR(:,j+1)
             col j+1: wr(j)*VR(:,j+1) + wi(j)*VR(:,j)  */
        MpfrMatrix VD(n, n, prec);
        MpfrScalar tmp1(prec), tmp2(prec);

        int j = 0;
        while (j < n) {
            double wi_j = mpfr_get_d(wi_mpfr.at(j, 0), MPFR_RNDN);
            if (std::abs(wi_j) < 1e-300 || j + 1 >= n) {
                /* Real eigenvalue */
                for (int i = 0; i < n; ++i)
                    mpfr_mul(VD.at(i, j), VR_mpfr.at(i, j), wr_mpfr.at(j, 0), MPFR_RNDN);
                ++j;
            } else {
                /* Complex conjugate pair */
                for (int i = 0; i < n; ++i) {
                    /* col j: wr*vr_j - wi*vr_{j+1} */
                    mpfr_mul(tmp1.get(), wr_mpfr.at(j, 0), VR_mpfr.at(i, j), MPFR_RNDN);
                    mpfr_mul(tmp2.get(), wi_mpfr.at(j, 0), VR_mpfr.at(i, j + 1), MPFR_RNDN);
                    mpfr_sub(VD.at(i, j), tmp1.get(), tmp2.get(), MPFR_RNDN);

                    /* col j+1: wr*vr_{j+1} + wi*vr_j */
                    mpfr_mul(tmp1.get(), wr_mpfr.at(j, 0), VR_mpfr.at(i, j + 1), MPFR_RNDN);
                    mpfr_mul(tmp2.get(), wi_mpfr.at(j, 0), VR_mpfr.at(i, j), MPFR_RNDN);
                    mpfr_add(VD.at(i, j + 1), tmp1.get(), tmp2.get(), MPFR_RNDN);
                }
                j += 2;
            }
        }

        MpfrMatrix R(n, n, prec);
        mpfr_mat_sub(R, AVR, VD);

        double norm_R = mpfr_mat_norm1(R);
        double norm_A = mpfr_mat_norm1(A_mpfr);
        double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;

        LapackResult lr = {residual, -1.0, info};
        report_lapack_result("GEEV", "", lr, format);
    }

    std::free(VL_dummy); std::free(VR);
    std::free(wr); std::free(wi);
    std::free(A_buf); std::free(A_orig);
}
