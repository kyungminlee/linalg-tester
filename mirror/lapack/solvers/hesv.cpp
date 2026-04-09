/* hesv.cpp -- Mirror tester for LAPACK CHESV/ZHESV (hermitian indefinite solve) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*hesv_fn_t)(
    const char *uplo, const int *n, const int *nrhs,
    void *A, const int *lda, int *ipiv,
    void *B, const int *ldb,
    void *work, const int *lwork, int *info,
    std::size_t uplo_len
);

void mirror_test_hesv(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int nrhs = (n < 4) ? n : 4;
    mpfr_prec_t prec = config.prec;

    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Generate canonical MPFR data */
        MpfrComplexMatrix A_mpfr(n, n, prec);
        MpfrComplexMatrix B_mpfr(n, nrhs, prec);

        gen_mpfr_diag_dominant_hermitian(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);

        /* Run one side */
        auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
            auto *fn = reinterpret_cast<hesv_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query */
            int lwork = -1;
            int info = 0;
            std::vector<char> work_query(s.ctx.typesize);
            {
                void *tmp_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
                void *tmp_B = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
                int *tmp_ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));

                fn(&uplo, &n, &nrhs, tmp_A, &lda, tmp_ipiv,
                   tmp_B, &ldb, work_query.data(), &lwork, &info,
                   (std::size_t)1);

                lwork = mirror_query_lwork_complex(work_query.data(), s.ctx);

                std::free(tmp_A);
                std::free(tmp_B);
                std::free(tmp_ipiv);
            }

            /* Re-materialize and solve */
            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
            void *native_B = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
            int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
            std::vector<char> work(static_cast<std::size_t>(lwork) * s.ctx.typesize);
            info = 0;

            fn(&uplo, &n, &nrhs, native_A, &lda, ipiv,
               native_B, &ldb, work.data(), &lwork, &info,
               (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_B, ldb, s.ctx);

            std::free(native_A);
            std::free(native_B);
            std::free(ipiv);
        };

        MpfrComplexMatrix res_a(n, nrhs, prec);
        MpfrComplexMatrix res_b(n, nrhs, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d nrhs=%d", uplo, n, nrhs);
        mirror_report_result("HESV", params_str, err,
                              nullptr, nullptr, config);
    }
}
