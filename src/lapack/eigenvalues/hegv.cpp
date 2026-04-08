/* hegv.cpp -- LAPACK HEGV accuracy tester (generalized Hermitian eigenvalue) */

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

extern "C" typedef void (*hegv_fn_t)(
    const int *itype, const char *jobz, const char *uplo,
    const int *n, void *A, const int *lda,
    void *B, const int *ldb,
    void *W, void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobz_len, std::size_t uplo_len);

void test_hegv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<hegv_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int ldb = n + params.ld_pad;
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Generate Hermitian A */
        void *A_herm = gen_hermitian_array(n, uplo, ctx.typesize, ctx.from_mpfr_complex, prec, &seed_A);
        void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        if (lda == n) {
            std::memcpy(A_buf, A_herm, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(A_buf, 0, static_cast<std::size_t>(lda) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_herm) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrComplexMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(A_mpfr, A_buf, lda, ctx);

        /* Generate HPD B */
        void *B_hpd = gen_hermitian_positive_definite_array(n, ctx.typesize, ctx.from_mpfr_complex,
                                                             ctx.to_mpfr_complex, prec, &seed_B);
        void *B_buf = std::malloc(static_cast<std::size_t>(ldb) * n * ctx.typesize);
        if (ldb == n) {
            std::memcpy(B_buf, B_hpd, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(B_buf, 0, static_cast<std::size_t>(ldb) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(B_buf) + j * ldb * ctx.typesize,
                            static_cast<char *>(B_hpd) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrComplexMatrix B_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(B_mpfr, B_buf, ldb, ctx);

        void *W = std::malloc(static_cast<std::size_t>(n) * real_typesize);
        void *rwork = std::malloc(static_cast<std::size_t>(std::max(1, 3 * n - 2)) * real_typesize);
        int itype = 1;
        char jobz = 'V';
        int info = 0;

        /* Workspace query */
        int lwork_q = -1;
        void *work_q = std::malloc(ctx.typesize);
        fn(&itype, &jobz, &uplo, &n, A_buf, &lda, B_buf, &ldb, W,
           work_q, &lwork_q, rwork, &info, (std::size_t)1, (std::size_t)1);
        int lwork = query_lwork_complex(work_q, ctx);
        std::free(work_q);

        /* Reload A and B */
        if (lda == n) std::memcpy(A_buf, A_herm, static_cast<std::size_t>(n) * n * ctx.typesize);
        else for (int j = 0; j < n; ++j)
            std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                        static_cast<char *>(A_herm) + j * n * ctx.typesize, n * ctx.typesize);
        if (ldb == n) std::memcpy(B_buf, B_hpd, static_cast<std::size_t>(n) * n * ctx.typesize);
        else for (int j = 0; j < n; ++j)
            std::memcpy(static_cast<char *>(B_buf) + j * ldb * ctx.typesize,
                        static_cast<char *>(B_hpd) + j * n * ctx.typesize, n * ctx.typesize);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
        fn(&itype, &jobz, &uplo, &n, A_buf, &lda, B_buf, &ldb, W,
           work, &lwork, rwork, &info, (std::size_t)1, (std::size_t)1);
        std::free(work);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("HEGV", ps, lr, format);
        } else {
            /* V is in A_buf, eigenvalues in W (real) */
            MpfrComplexMatrix V(n, n, prec);
            custom_to_mpfr_complex_mat(V, A_buf, lda, ctx);

            MpfrMatrix W_mpfr(n, 1, prec);
            const char *wp = static_cast<const char *>(W);
            for (int i = 0; i < n; ++i)
                ctx.to_mpfr(W_mpfr.at(i, 0), wp + i * real_typesize);

            /* Residual: ||A*V - B*V*diag(W)||_1 / (||A||_1 * n * eps) */
            MpfrComplexMatrix AV(n, n, prec);
            mpfr_complex_mat_mul_simple(AV, A_mpfr, V);

            /* B*V*diag(W) */
            MpfrComplexMatrix VD(n, n, prec);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    mpfr_complex_mul_real(VD.re(i, j), VD.im(i, j),
                                         V.re(i, j), V.im(i, j),
                                         W_mpfr.at(j, 0), MPFR_RNDN);

            MpfrComplexMatrix BVD(n, n, prec);
            mpfr_complex_mat_mul_simple(BVD, B_mpfr, VD);

            MpfrComplexMatrix R(n, n, prec);
            mpfr_complex_mat_sub(R, AV, BVD);

            double norm_R = mpfr_complex_mat_norm1(R);
            double norm_A = mpfr_complex_mat_norm1(A_mpfr);
            double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("HEGV", ps, lr, format);
        }

        std::free(rwork); std::free(W); std::free(A_buf); std::free(A_herm);
        std::free(B_buf); std::free(B_hpd);
    }
}
