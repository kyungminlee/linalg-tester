/* laswp.cpp -- Mirror tester for LAPACK LASWP (row permutation) */

#include "../mirror_lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*laswp_fn_t)(
    const int *n, void *A, const int *lda,
    const int *k1, const int *k2,
    const int *ipiv, const int *incx);

void mirror_test_laswp(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;
    unsigned seed_ipiv = params.seed + 10;

    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    /* Generate random ipiv array with values in [1, m] */
    int *ipiv = static_cast<int *>(
        std::malloc(static_cast<std::size_t>(m) * sizeof(int)));
    {
        unsigned s = seed_ipiv;
        for (int i = 0; i < m; ++i) {
            s = s * 1103515245u + 12345u;
            ipiv[i] = 1 + static_cast<int>(
                (s >> 16) % static_cast<unsigned>(m));
        }
    }

    int k1 = 1;
    int k2 = m;
    int incx = 1;

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        auto *fn = reinterpret_cast<laswp_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

        fn(&n, native_A, &lda, &k1, &k2, ipiv, &incx);

        custom_to_mpfr_mat(result, native_A, lda, side.ctx);

        std::free(native_A);
    };

    MpfrMatrix res_a(m, n, prec);
    MpfrMatrix res_b(m, n, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d k1=%d k2=%d", m, n, k1, k2);
    mirror_report_result("LASWP", params_str, err,
                          nullptr, nullptr, config);

    std::free(ipiv);
}
