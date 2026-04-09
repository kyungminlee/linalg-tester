/* cpotrf.cpp -- Mirror tester for LAPACK complex POTRF (Cholesky factorization) */

#include "../mirror_lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*cpotrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t uplo_len);

void mirror_test_cpotrf(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrComplexMatrix A_mpfr(n, n, prec);
        gen_mpfr_hermitian_positive_definite(A_mpfr, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side,
                            MpfrComplexMatrix &result) {
            void *native_A = mpfr_complex_mat_to_native(A_mpfr, lda, side.ctx);
            int info = 0;

            auto *fn = reinterpret_cast<cpotrf_fn_t>(
                load_sym(side.lib, side.sym.c_str()));
            fn(&uplo, &n, native_A, &lda, &info, (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_A, lda, side.ctx);
            std::free(native_A);
        };

        MpfrComplexMatrix res_a(n, n, prec);
        MpfrComplexMatrix res_b(n, n, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str), "uplo=%c", uplo);
        mirror_report_result("CPOTRF", params_str, err,
                              nullptr, nullptr, config);
    }
}
