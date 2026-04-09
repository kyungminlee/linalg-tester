/* cgesv.cpp -- Mirror tester for LAPACK CGESV/ZGESV (complex general solve) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*cgesv_fn_t)(
    const int *n, const int *nrhs,
    void *A, const int *lda, int *ipiv,
    void *B, const int *ldb, int *info
);

void mirror_test_cgesv(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int nrhs = (n < 4) ? n : 4;
    mpfr_prec_t prec = config.prec;

    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    unsigned seed_A = params.seed;
    unsigned seed_B = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrComplexMatrix A_mpfr(n, n, prec);
    MpfrComplexMatrix B_mpfr(n, nrhs, prec);

    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);
    gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
        void *native_B = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
        int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        auto *fn = reinterpret_cast<cgesv_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, &nrhs, native_A, &lda, ipiv, native_B, &ldb, &info);

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
                  "n=%d nrhs=%d", n, nrhs);
    mirror_report_result("CGESV", params_str, err,
                          nullptr, nullptr, config);
}
