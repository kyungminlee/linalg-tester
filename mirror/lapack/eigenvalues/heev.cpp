/* heev.cpp -- Mirror tester for LAPACK HEEV (hermitian eigenvalues) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*heev_fn_t)(
    const char *jobz, const char *uplo, const int *n,
    void *A, const int *lda, void *W,
    void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobz_len, std::size_t uplo_len
);

void mirror_test_heev(const MirrorSide &a, const MirrorSide &b,
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
            int rwork_size = (3 * n - 2 > 1) ? (3 * n - 2) : 1;
            void *rwork = std::calloc(rwork_size, real_ts);
            int info = 0;

            auto *fn = reinterpret_cast<heev_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query (complex workspace) */
            char work_buf[256];
            int lwork_query = -1;
            fn(&"N"[0], &uplo, &n, native_A, &lda, W,
               work_buf, &lwork_query, rwork, &info,
               (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
            void *work = std::calloc(lwork, s.ctx.typesize);

            /* Re-materialize A */
            std::free(native_A);
            native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);

            info = 0;
            fn(&"N"[0], &uplo, &n, native_A, &lda, W,
               work, &lwork, rwork, &info,
               (std::size_t)1, (std::size_t)1);

            /* W is real (typesize/2 per element) */
            native_real_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(native_A);
            std::free(W);
            std::free(rwork);
            std::free(work);
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
        mirror_report_result("HEEV", params_str, err,
                              nullptr, nullptr, config);
    }
}
