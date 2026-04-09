/* symm.cpp -- Mirror tester for BLAS Level 3 SYMM */

#include "../mirror_blas3.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer */
extern "C" typedef void (*symm_fn_t)(
    const char *side,  const char *uplo,
    const int  *m,     const int  *n,
    const void *alpha,
    const void *A,     const int  *lda,
    const void *B,     const int  *ldb,
    const void *beta,
    void       *C,     const int  *ldc,
    std::size_t side_len, std::size_t uplo_len
);

void mirror_test_symm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
        int ka = (side == 'L') ? m : n;

        int lda = ka + params.ld_pad;
        int ldb = m  + params.ld_pad;
        int ldc = m  + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_ab = params.seed + 3;

        /* Generate canonical MPFR data */
        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_mpfr(m, n, prec);
        MpfrMatrix C_mpfr(m, n, prec);
        MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);
        gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);
        gen_mpfr_random_matrix(C_mpfr, prec, &seed_C);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_ab);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_ab);

        /* Run one side */
        auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
            void *native_A     = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *native_B     = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
            void *native_C     = mpfr_mat_to_native(C_mpfr, ldc, s.ctx);
            void *native_alpha = mpfr_scalar_to_native(alpha_mpfr, s.ctx);
            void *native_beta  = mpfr_scalar_to_native(beta_mpfr, s.ctx);

            auto *fn = reinterpret_cast<symm_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&side, &uplo, &m, &n,
               native_alpha, native_A, &lda, native_B, &ldb,
               native_beta, native_C, &ldc,
               (std::size_t)1, (std::size_t)1);

            custom_to_mpfr_mat(result, native_C, ldc, s.ctx);

            std::free(native_A);
            std::free(native_B);
            std::free(native_C);
            std::free(native_alpha);
            std::free(native_beta);
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
                      "side=%c uplo=%c", side, uplo);
        mirror_report_result("SYMM", params_str, err,
                              nullptr, nullptr, config);
    }}
}
