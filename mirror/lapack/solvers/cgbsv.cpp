/* cgbsv.cpp -- Mirror tester for LAPACK CGBSV/ZGBSV (complex band solve) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*cgbsv_fn_t)(
    const int *n, const int *kl, const int *ku, const int *nrhs,
    void *AB, const int *ldab, int *ipiv,
    void *B, const int *ldb, int *info
);

void mirror_test_cgbsv(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int kl = params.kl, ku = params.ku;
    int nrhs = (n < 4) ? n : 4;
    mpfr_prec_t prec = config.prec;

    int ldab = 2 * kl + ku + 1;
    int ldb = n + params.ld_pad;

    unsigned seed_AB = params.seed;
    unsigned seed_B  = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrComplexMatrix AB_mpfr(ldab, n, prec);
    MpfrComplexMatrix B_mpfr(n, nrhs, prec);

    gen_mpfr_complex_gbsv_band_matrix(AB_mpfr, n, kl, ku, prec, &seed_AB);
    gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
        void *native_AB = mpfr_complex_mat_to_native(AB_mpfr, ldab, s.ctx);
        void *native_B  = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
        int *ipiv = static_cast<int *>(std::calloc(n, sizeof(int)));
        int info = 0;

        auto *fn = reinterpret_cast<cgbsv_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, &kl, &ku, &nrhs, native_AB, &ldab, ipiv,
           native_B, &ldb, &info);

        custom_to_mpfr_complex_mat(result, native_B, ldb, s.ctx);

        std::free(native_AB);
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
                  "n=%d kl=%d ku=%d nrhs=%d", n, kl, ku, nrhs);
    mirror_report_result("CGBSV", params_str, err,
                          nullptr, nullptr, config);
}
