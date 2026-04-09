/* syevr.cpp -- Mirror tester for LAPACK SYEVR (symmetric eigenvalues, MRRR) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*syevr_fn_t)(
    const char *jobz, const char *range, const char *uplo,
    const int *n, void *A, const int *lda,
    const void *vl, const void *vu, const int *il, const int *iu,
    const void *abstol, int *m_out,
    void *W, void *Z, const int *ldz, int *isuppz,
    void *work, const int *lwork, int *iwork, const int *liwork,
    int *info,
    std::size_t jobz_len, std::size_t range_len, std::size_t uplo_len
);

void mirror_test_syevr(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrMatrix A_mpfr(n, n, prec);
        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &W_out) {
            void *native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *W = std::calloc(n, s.ctx.typesize);
            int info = 0;

            /* Unused parameters for range='A' */
            MpfrScalar zero_s(prec);
            mpfr_set_d(zero_s.get(), 0.0, MPFR_RNDN);
            void *vl = mpfr_scalar_to_native(zero_s, s.ctx);
            void *vu = mpfr_scalar_to_native(zero_s, s.ctx);
            int il = 1, iu = n;
            void *abstol_native = mpfr_scalar_to_native(zero_s, s.ctx);
            int m_out = 0;
            int ldz = 1;  /* Z not referenced with jobz='N' */
            char z_dummy;
            int *isuppz = static_cast<int *>(std::calloc(2 * n, sizeof(int)));

            auto *fn = reinterpret_cast<syevr_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query */
            char work_buf[256];
            int iwork_buf;
            int lwork_query = -1;
            int liwork_query = -1;
            fn(&"N"[0], &"A"[0], &uplo, &n, native_A, &lda,
               vl, vu, &il, &iu, abstol_native, &m_out,
               W, &z_dummy, &ldz, isuppz,
               work_buf, &lwork_query, &iwork_buf, &liwork_query, &info,
               (std::size_t)1, (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork(work_buf, s.ctx);
            int liwork = iwork_buf;
            void *work = std::calloc(lwork, s.ctx.typesize);
            int *iwork = static_cast<int *>(std::calloc(liwork, sizeof(int)));

            /* Re-materialize A */
            std::free(native_A);
            native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);

            info = 0;
            m_out = 0;
            fn(&"N"[0], &"A"[0], &uplo, &n, native_A, &lda,
               vl, vu, &il, &iu, abstol_native, &m_out,
               W, &z_dummy, &ldz, isuppz,
               work, &lwork, iwork, &liwork, &info,
               (std::size_t)1, (std::size_t)1, (std::size_t)1);

            native_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(native_A);
            std::free(W);
            std::free(vl);
            std::free(vu);
            std::free(abstol_native);
            std::free(isuppz);
            std::free(work);
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
        mirror_report_result("SYEVR", params_str, err,
                              nullptr, nullptr, config);
    }
}
