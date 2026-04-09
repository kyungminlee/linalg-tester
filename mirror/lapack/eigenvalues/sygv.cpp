/* sygv.cpp -- Mirror tester for LAPACK SYGV (generalized symmetric eigenvalue) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*sygv_fn_t)(
    const int *itype, const char *jobz, const char *uplo,
    const int *n, void *A, const int *lda, void *B, const int *ldb,
    void *W, void *work, const int *lwork, int *info,
    std::size_t jobz_len, std::size_t uplo_len
);

void mirror_test_sygv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Symmetric A, SPD B */
        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix B_mpfr(n, n, prec);
        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_positive_definite_matrix(B_mpfr, prec, &seed_B);

        int itype = 1;

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &W_out) {
            void *native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *native_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
            void *W = std::calloc(n, s.ctx.typesize);
            int info = 0;

            auto *fn = reinterpret_cast<sygv_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query */
            char work_buf[256];
            int lwork_query = -1;
            fn(&itype, &"N"[0], &uplo, &n, native_A, &lda, native_B, &ldb,
               W, work_buf, &lwork_query, &info,
               (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork(work_buf, s.ctx);
            void *work = std::calloc(lwork, s.ctx.typesize);

            /* Re-materialize A,B */
            std::free(native_A);
            std::free(native_B);
            native_A = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            native_B = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);

            info = 0;
            fn(&itype, &"N"[0], &uplo, &n, native_A, &lda, native_B, &ldb,
               W, work, &lwork, &info,
               (std::size_t)1, (std::size_t)1);

            native_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(native_A);
            std::free(native_B);
            std::free(W);
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
                      "itype=1 uplo=%c n=%d", uplo, n);
        mirror_report_result("SYGV", params_str, err,
                              nullptr, nullptr, config);
    }
}
