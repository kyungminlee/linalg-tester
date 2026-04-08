/* sygv.cpp -- LAPACK SYGV accuracy tester (generalized symmetric eigenvalue) */

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

extern "C" typedef void (*sygv_fn_t)(
    const int *itype, const char *jobz, const char *uplo,
    const int *n, void *A, const int *lda,
    void *B, const int *ldb,
    void *W, void *work, const int *lwork, int *info,
    std::size_t jobz_len, std::size_t uplo_len);

void test_sygv(const TesterCtx &ctx, void *lib, const char *sym,
               const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<sygv_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int ldb = n + params.ld_pad;
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Generate symmetric A */
        void *A_sym = gen_symmetric_array(n, uplo, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        if (lda == n) {
            std::memcpy(A_buf, A_sym, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(A_buf, 0, static_cast<std::size_t>(lda) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_sym) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_mat(A_mpfr, A_buf, lda, ctx);

        /* Generate SPD B */
        void *B_spd = gen_positive_definite_array(n, ctx.typesize, ctx.from_mpfr,
                                                   ctx.to_mpfr, prec, &seed_B);
        void *B_buf = std::malloc(static_cast<std::size_t>(ldb) * n * ctx.typesize);
        if (ldb == n) {
            std::memcpy(B_buf, B_spd, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(B_buf, 0, static_cast<std::size_t>(ldb) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(B_buf) + j * ldb * ctx.typesize,
                            static_cast<char *>(B_spd) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        MpfrMatrix B_mpfr(n, n, prec);
        custom_to_mpfr_mat(B_mpfr, B_buf, ldb, ctx);

        void *W = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
        int itype = 1;
        char jobz = 'V';
        int info = 0;

        /* Workspace query */
        int lwork_q = -1;
        void *work_q = std::malloc(ctx.typesize);
        fn(&itype, &jobz, &uplo, &n, A_buf, &lda, B_buf, &ldb, W,
           work_q, &lwork_q, &info, (std::size_t)1, (std::size_t)1);
        int lwork = query_lwork(work_q, ctx);
        std::free(work_q);

        /* Reload A and B */
        if (lda == n) std::memcpy(A_buf, A_sym, static_cast<std::size_t>(n) * n * ctx.typesize);
        else for (int j = 0; j < n; ++j)
            std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                        static_cast<char *>(A_sym) + j * n * ctx.typesize, n * ctx.typesize);
        if (ldb == n) std::memcpy(B_buf, B_spd, static_cast<std::size_t>(n) * n * ctx.typesize);
        else for (int j = 0; j < n; ++j)
            std::memcpy(static_cast<char *>(B_buf) + j * ldb * ctx.typesize,
                        static_cast<char *>(B_spd) + j * n * ctx.typesize, n * ctx.typesize);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
        fn(&itype, &jobz, &uplo, &n, A_buf, &lda, B_buf, &ldb, W,
           work, &lwork, &info, (std::size_t)1, (std::size_t)1);
        std::free(work);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("SYGV", ps, lr, format);
        } else {
            /* V is in A_buf, eigenvalues in W */
            MpfrMatrix V(n, n, prec);
            custom_to_mpfr_mat(V, A_buf, lda, ctx);

            MpfrMatrix W_mpfr(n, 1, prec);
            const char *wp = static_cast<const char *>(W);
            for (int i = 0; i < n; ++i)
                ctx.to_mpfr(W_mpfr.at(i, 0), wp + i * ctx.typesize);

            /* Residual: ||A*V - B*V*diag(W)||_1 / (||A||_1 * n * eps) */
            MpfrMatrix AV(n, n, prec);
            mpfr_mat_mul_simple(AV, A_mpfr, V);

            /* B*V*diag(W) */
            MpfrMatrix VD(n, n, prec);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    mpfr_mul(VD.at(i, j), V.at(i, j), W_mpfr.at(j, 0), MPFR_RNDN);

            MpfrMatrix BVD(n, n, prec);
            mpfr_mat_mul_simple(BVD, B_mpfr, VD);

            MpfrMatrix R(n, n, prec);
            mpfr_mat_sub(R, AV, BVD);

            double norm_R = mpfr_mat_norm1(R);
            double norm_A = mpfr_mat_norm1(A_mpfr);
            double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("SYGV", ps, lr, format);
        }

        std::free(W); std::free(A_buf); std::free(A_sym);
        std::free(B_buf); std::free(B_spd);
    }
}
