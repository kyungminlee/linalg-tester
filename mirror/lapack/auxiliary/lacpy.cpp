/* lacpy.cpp -- Mirror tester for LAPACK LACPY (matrix copy) */

#include "../mirror_lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*lacpy_fn_t)(
    const char *uplo, const int *m, const int *n,
    const void *A, const int *lda,
    void *B, const int *ldb, std::size_t);

void mirror_test_lacpy(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = m + params.ld_pad;
    int ldb = m + params.ld_pad;

    for (char uplo : {'A', 'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrMatrix A_mpfr(m, n, prec);
        gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            auto *fn = reinterpret_cast<lacpy_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *native_B = std::calloc(
                static_cast<std::size_t>(ldb) * n, side.ctx.typesize);

            fn(&uplo, &m, &n, native_A, &lda, native_B, &ldb, (std::size_t)1);

            custom_to_mpfr_mat(result, native_B, ldb, side.ctx);

            std::free(native_A);
            std::free(native_B);
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
                      "uplo=%c m=%d n=%d", uplo, m, n);
        mirror_report_result("LACPY", params_str, err,
                              nullptr, nullptr, config);
    }
}
