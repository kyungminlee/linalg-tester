/* heevr.cpp -- LAPACK HEEVR accuracy tester (Hermitian eigenvalue, MRRR) */

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

extern "C" typedef void (*heevr_fn_t)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, void *A, const int *lda,
    const void *vl, const void *vu,
    const int *il, const int *iu,
    const void *abstol, int *m_out,
    void *W, void *Z, const int *ldz, int *isuppz,
    void *work, const int *lwork,
    void *rwork, const int *lrwork,
    int *iwork, const int *liwork,
    int *info,
    std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len);

void test_heevr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    auto *fn = reinterpret_cast<heevr_fn_t>(load_sym(lib, sym));
    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    std::size_t real_typesize = ctx.typesize / 2;

    for (char uplo : {'U', 'L'}) {
        int lda = n + params.ld_pad;
        int ldz = n;
        unsigned seed_A = params.seed;

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

        void *W = std::malloc(static_cast<std::size_t>(n) * real_typesize);
        void *Z = std::malloc(static_cast<std::size_t>(ldz) * n * ctx.typesize);
        int *isuppz = new int[2 * n];

        char jobz = 'V', range = 'A';
        int il = 1, iu = n, m_out = 0, info = 0;

        /* abstol, vl, vu are REAL */
        void *abstol_buf = std::malloc(real_typesize);
        { MpfrScalar tmp(prec); mpfr_set_d(tmp.get(), 0.0, MPFR_RNDN);
          ctx.from_mpfr(abstol_buf, tmp.get(), MPFR_RNDN); }
        void *vl_buf = std::malloc(real_typesize);
        void *vu_buf = std::malloc(real_typesize);
        { MpfrScalar tmp(prec); mpfr_set_d(tmp.get(), 0.0, MPFR_RNDN);
          ctx.from_mpfr(vl_buf, tmp.get(), MPFR_RNDN);
          ctx.from_mpfr(vu_buf, tmp.get(), MPFR_RNDN); }

        /* Workspace query */
        int lwork_q = -1, lrwork_q = -1, liwork_q = -1;
        void *work_q = std::malloc(ctx.typesize);
        void *rwork_q_buf = std::malloc(real_typesize);
        int iwork_q_val = 0;
        fn(&jobz, &range, &uplo, &n, A_buf, &lda,
           vl_buf, vu_buf, &il, &iu, abstol_buf, &m_out,
           W, Z, &ldz, isuppz,
           work_q, &lwork_q, rwork_q_buf, &lrwork_q, &iwork_q_val, &liwork_q, &info,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);
        int lwork = query_lwork_complex(work_q, ctx);
        int lrwork;
        { MpfrScalar tmp(prec); ctx.to_mpfr(tmp.get(), rwork_q_buf);
          lrwork = static_cast<int>(mpfr_get_d(tmp.get(), MPFR_RNDN)); }
        int liwork = iwork_q_val;
        std::free(work_q);
        std::free(rwork_q_buf);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
        void *rwork = std::malloc(static_cast<std::size_t>(lrwork > 0 ? lrwork : 1) * real_typesize);
        int *iwork = new int[liwork > 0 ? liwork : 1];

        /* Reload A */
        if (lda == n) {
            std::memcpy(A_buf, A_herm, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_herm) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        fn(&jobz, &range, &uplo, &n, A_buf, &lda,
           vl_buf, vu_buf, &il, &iu, abstol_buf, &m_out,
           W, Z, &ldz, isuppz,
           work, &lwork, rwork, &lrwork, iwork, &liwork, &info,
           (std::size_t)1, (std::size_t)1, (std::size_t)1);
        std::free(work);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("HEEVR", ps, lr, format);
        } else {
            MpfrComplexMatrix V(n, m_out, prec);
            custom_to_mpfr_complex_mat(V, Z, ldz, ctx);

            MpfrMatrix W_mpfr(m_out, 1, prec);
            const char *wp = static_cast<const char *>(W);
            for (int i = 0; i < m_out; ++i)
                ctx.to_mpfr(W_mpfr.at(i, 0), wp + i * real_typesize);

            MpfrComplexMatrix AV(n, m_out, prec);
            mpfr_complex_mat_mul_simple(AV, A_mpfr, V);

            MpfrComplexMatrix VD(n, m_out, prec);
            for (int j = 0; j < m_out; ++j)
                for (int i = 0; i < n; ++i)
                    mpfr_complex_mul_real(VD.re(i, j), VD.im(i, j),
                                         V.re(i, j), V.im(i, j),
                                         W_mpfr.at(j, 0), MPFR_RNDN);

            MpfrComplexMatrix R(n, m_out, prec);
            mpfr_complex_mat_sub(R, AV, VD);

            double norm_R = mpfr_complex_mat_norm1(R);
            double norm_A = mpfr_complex_mat_norm1(A_mpfr);
            double residual = (norm_A > 0.0) ? norm_R / (norm_A * n * eps) : 0.0;
            double orth = mpfr_complex_orthogonality(V, eps);

            char ps[64]; std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
            LapackResult lr = {residual, orth, info};
            report_lapack_result("HEEVR", ps, lr, format);
        }

        delete[] iwork; std::free(rwork); delete[] isuppz;
        std::free(W); std::free(Z); std::free(A_buf); std::free(A_herm);
        std::free(abstol_buf); std::free(vl_buf); std::free(vu_buf);
    }
}
