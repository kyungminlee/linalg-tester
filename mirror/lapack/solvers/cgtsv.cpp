/* cgtsv.cpp -- Mirror tester for LAPACK CGTSV/ZGTSV (complex tridiagonal solve) */

#include "../mirror_lapack_common.h"

extern "C" typedef void (*cgtsv_fn_t)(
    const int *n, const int *nrhs,
    void *dl, void *d, void *du,
    void *B, const int *ldb, int *info
);

void mirror_test_cgtsv(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int nrhs = (n < 4) ? n : 4;
    mpfr_prec_t prec = config.prec;

    int ldb = n + params.ld_pad;

    unsigned seed_tri = params.seed;
    unsigned seed_B   = params.seed + 1;

    /* Generate canonical MPFR data */
    MpfrComplexMatrix dl_mpfr(n - 1, 1, prec);
    MpfrComplexMatrix d_mpfr(n, 1, prec);
    MpfrComplexMatrix du_mpfr(n - 1, 1, prec);
    MpfrComplexMatrix B_mpfr(n, nrhs, prec);

    gen_mpfr_complex_tridiagonal(dl_mpfr, d_mpfr, du_mpfr, n, prec, &seed_tri);
    gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);

    /* Run one side */
    auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
        void *native_dl = mpfr_complex_vec_to_native(dl_mpfr, 1, s.ctx);
        void *native_d  = mpfr_complex_vec_to_native(d_mpfr, 1, s.ctx);
        void *native_du = mpfr_complex_vec_to_native(du_mpfr, 1, s.ctx);
        void *native_B  = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
        int info = 0;

        auto *fn = reinterpret_cast<cgtsv_fn_t>(
            load_sym(s.lib, s.sym.c_str()));
        fn(&n, &nrhs, native_dl, native_d, native_du,
           native_B, &ldb, &info);

        custom_to_mpfr_complex_mat(result, native_B, ldb, s.ctx);

        std::free(native_dl);
        std::free(native_d);
        std::free(native_du);
        std::free(native_B);
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
    mirror_report_result("CGTSV", params_str, err,
                          nullptr, nullptr, config);
}
