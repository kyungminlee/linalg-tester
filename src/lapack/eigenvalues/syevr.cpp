/* syevr.cpp -- LAPACK SYEVR accuracy tester (symmetric eigenvalue, MRRR) */

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

extern "C" typedef void (*syevr_fn_t)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, void *A, const int *lda,
    const void *vl, const void *vu,
    const int *il, const int *iu,
    const void *abstol, int *m_out,
    void *W, void *Z, const int *ldz, int *isuppz,
    void *work, const int *lwork,
    int *iwork, const int *liwork,
    int *info,
    std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);

void test_syevr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<syevr_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int ldz = n;
        unsigned seed_A = params.seed;

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

        void *W = std::malloc(static_cast<std::size_t>(n) * ctx.typesize);
        void *Z = std::malloc(static_cast<std::size_t>(ldz) * n * ctx.typesize);
        int *isuppz = new int[2 * n];

        char jobz = 'V', range = 'A';
        int il = 1, iu = n, m_out = 0, info = 0;

        /* abstol = 0 (use default) */
        void *abstol_buf = std::malloc(ctx.typesize);
        { MpfrScalar tmp(prec); mpfr_set_d(tmp.get(), 0.0, MPFR_RNDN);
          ctx.from_mpfr(abstol_buf, tmp.get(), MPFR_RNDN); }
        void *vl_buf = std::malloc(ctx.typesize);
        void *vu_buf = std::malloc(ctx.typesize);
        { MpfrScalar tmp(prec); mpfr_set_d(tmp.get(), 0.0, MPFR_RNDN);
          ctx.from_mpfr(vl_buf, tmp.get(), MPFR_RNDN);
          ctx.from_mpfr(vu_buf, tmp.get(), MPFR_RNDN); }

        /* Workspace query */
        int lwork_q = -1, liwork_q = -1;
        void *work_q = std::malloc(ctx.typesize);
        int iwork_q_val = 0;
        fn(&jobz, &range, &uplo, &n, A_buf, &lda,
           vl_buf, vu_buf, &il, &iu, abstol_buf, &m_out,
           W, Z, &ldz, isuppz,
           work_q, &lwork_q, &iwork_q_val, &liwork_q, &info,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);
        int lwork = query_lwork(work_q, ctx);
        int liwork = iwork_q_val;
        std::free(work_q);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
        int *iwork = new int[liwork > 0 ? liwork : 1];

        /* Reload A */
        if (lda == n) {
            std::memcpy(A_buf, A_sym, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_sym) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        fn(&jobz, &range, &uplo, &n, A_buf, &lda,
           vl_buf, vu_buf, &il, &iu, abstol_buf, &m_out,
           W, Z, &ldz, isuppz,
           work, &lwork, iwork, &liwork, &info,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);
        std::free(work);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("SYEVR", ps, lr, format);
        } else {
            MpfrMatrix V(n, m_out, prec);
            custom_to_mpfr_mat(V, Z, ldz, ctx);

            MpfrMatrix W_mpfr(m_out, 1, prec);
            const char *wp = static_cast<const char *>(W);
            for (int i = 0; i < m_out; ++i)
                ctx.to_mpfr(W_mpfr.at(i, 0), wp + i * ctx.typesize);

            MpfrMatrix AV(n, m_out, prec);
            mpfr_mat_mul_simple(AV, A_mpfr, V);

            MpfrMatrix VD(n, m_out, prec);
            for (int j = 0; j < m_out; ++j)
                for (int i = 0; i < n; ++i)
                    mpfr_mul(VD.at(i, j), V.at(i, j), W_mpfr.at(j, 0), MPFR_RNDN);

            MpfrMatrix R(n, m_out, prec);
            mpfr_mat_sub(R, AV, VD);

            double norm_R = mpfr_mat_norm1(R);
            double norm_A = mpfr_mat_norm1(A_mpfr);
            double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;
            double orth = mpfr_orthogonality(V, eps);

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, orth, info};
            report_lapack_result("SYEVR", ps, lr, format);
        }

        delete[] iwork; delete[] isuppz;
        std::free(W); std::free(Z); std::free(A_buf); std::free(A_sym);
        std::free(abstol_buf); std::free(vl_buf); std::free(vu_buf);
    }
}
