/* heevr.cpp -- Mirror tester for LAPACK HEEVR (hermitian eigenvalues, MRRR) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*heevr_fn_t)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, void *A, const int *lda,
    const void *vl, const void *vu, const int *il, const int *iu,
    const void *abstol, int *m_out,
    void *W, void *Z, const int *ldz, int *isuppz,
    void *work, const int *lwork, void *rwork, const int *lrwork,
    int *iwork, const int *liwork, int *info,
    std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len
);

void mirror_test_heevr(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_hermitian_matrix(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &W_out) {
            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
            std::size_t real_ts = s.ctx.typesize / 2;
            void *W = std::calloc(n, real_ts);
            int info = 0;

            /* Unused parameters for range='A' */
            void *vl = std::calloc(1, real_ts);
            void *vu = std::calloc(1, real_ts);
            int il = 1, iu = n;
            void *abstol_native = std::calloc(1, real_ts);
            int m_out = 0;
            int ldz = 1;
            char z_dummy;
            int *isuppz = static_cast<int *>(std::calloc(2 * n, sizeof(int)));

            auto *fn = reinterpret_cast<heevr_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query: lwork=-1, lrwork=-1, liwork=-1 */
            char work_buf[256];
            char rwork_buf[256];
            int iwork_buf;
            int lwork_query = -1;
            int lrwork_query = -1;
            int liwork_query = -1;
            fn(&"N"[0], &"A"[0], &uplo, &n, native_A, &lda,
               vl, vu, &il, &iu, abstol_native, &m_out,
               W, &z_dummy, &ldz, isuppz,
               work_buf, &lwork_query, rwork_buf, &lrwork_query,
               &iwork_buf, &liwork_query, &info,
               (std::size_t)1, (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
            MpfrScalar rwork_tmp(prec);
            s.ctx.to_mpfr(rwork_tmp.get(), rwork_buf);
            int lrwork = static_cast<int>(mpfr_get_d(rwork_tmp.get(), MPFR_RNDN));
            int liwork = iwork_buf;

            void *work = std::calloc(lwork, s.ctx.typesize);
            void *rwork = std::calloc(lrwork, real_ts);
            int *iwork = static_cast<int *>(std::calloc(liwork, sizeof(int)));

            /* Re-materialize A */
            std::free(native_A);
            native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);

            info = 0;
            m_out = 0;
            fn(&"N"[0], &"A"[0], &uplo, &n, native_A, &lda,
               vl, vu, &il, &iu, abstol_native, &m_out,
               W, &z_dummy, &ldz, isuppz,
               work, &lwork, rwork, &lrwork,
               iwork, &liwork, &info,
               (std::size_t)1, (std::size_t)1, (std::size_t)1);

            native_real_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(native_A);
            std::free(W);
            std::free(vl);
            std::free(vu);
            std::free(abstol_native);
            std::free(isuppz);
            std::free(work);
            std::free(rwork);
            std::free(iwork);
        };

        MpfrMatrix W_a(n, 1, prec);
        MpfrMatrix W_b(n, 1, prec);
        run_side(a, W_a);
        run_side(b, W_b);

        const MpfrMatrix &ref = (config.reference == "a") ? W_a : W_b;
        const MpfrMatrix &tst = (config.reference == "a") ? W_b : W_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        mirror_report_result("HEEVR", params_str, err,
                              nullptr, nullptr, config);
    }
}
