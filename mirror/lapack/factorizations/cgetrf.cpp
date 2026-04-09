/* cgetrf.cpp -- Mirror tester for LAPACK complex GETRF (LU factorization) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*cgetrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

void mirror_test_cgetrf(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int mn = std::min(m, n);
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrComplexMatrix &result,
                        int *ipiv_out) {
        void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
        int info = 0;

        auto *fn = reinterpret_cast<cgetrf_fn_t>(
            load_sym(side.lib, side.sym.c_str()));
        fn(&m, &n, native_A, &lda, ipiv_out, &info);

        custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);
        std::free(native_A);
    };

    MpfrComplexMatrix res_a(m, n, prec);
    MpfrComplexMatrix res_b(m, n, prec);
    int *ipiv_a = new int[mn];
    int *ipiv_b = new int[mn];
    run_side(a, res_a, ipiv_a);
    run_side(b, res_b, ipiv_b);

    const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

    bool ipiv_match = true;
    for (int i = 0; i < mn; ++i) {
        if (ipiv_a[i] != ipiv_b[i]) { ipiv_match = false; break; }
    }

    char params_str[256];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d ipiv_match=%s", m, n,
                  ipiv_match ? "yes" : "NO");
    mirror_report_result("CGETRF", params_str, err,
                          nullptr, nullptr, config);

    delete[] ipiv_a;
    delete[] ipiv_b;
}
