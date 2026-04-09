/* gemm.cpp -- Mirror tester for BLAS Level 3 GEMM */

#include "../mirror_blas3.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* Fortran-ABI function pointer (hidden char lengths at the end) */
extern "C" typedef void (*gemm_fn_t)(
    const char *transa, const char *transb,
    const int  *m,      const int  *n,      const int  *k,
    const void *alpha,
    const void *A,      const int  *lda,
    const void *B,      const int  *ldb,
    const void *beta,
    void       *C,      const int  *ldc,
    std::size_t transa_len, std::size_t transb_len
);

void mirror_test_gemm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n, k = params.k;
    mpfr_prec_t prec = config.prec;

    for (char ta : {'N', 'T', 'C'}) {
        for (char tb : {'N', 'T', 'C'}) {
            unsigned seed_A  = params.seed;
            unsigned seed_B  = params.seed + 1;
            unsigned seed_C  = params.seed + 2;
            unsigned seed_ab = params.seed + 3;

            int rows_A = (ta == 'N') ? m : k;
            int cols_A = (ta == 'N') ? k : m;
            int rows_B = (tb == 'N') ? k : n;
            int cols_B = (tb == 'N') ? n : k;

            int lda = rows_A + params.ld_pad;
            int ldb = rows_B + params.ld_pad;
            int ldc = m      + params.ld_pad;

            /* Generate canonical MPFR data */
            MpfrMatrix A_mpfr(rows_A, cols_A, prec);
            MpfrMatrix B_mpfr(rows_B, cols_B, prec);
            MpfrMatrix C_mpfr(m, n, prec);
            MpfrScalar alpha_mpfr(prec), beta_mpfr(prec);

            gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);
            gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);
            gen_mpfr_random_matrix(C_mpfr, prec, &seed_C);
            gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_ab);
            gen_mpfr_random_scalar(beta_mpfr, prec, &seed_ab);

            /* Run side A */
            auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
                void *native_A     = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
                void *native_B     = mpfr_mat_to_native(B_mpfr, ldb, side.ctx);
                void *native_C     = mpfr_mat_to_native(C_mpfr, ldc, side.ctx);
                void *native_alpha = mpfr_scalar_to_native(alpha_mpfr, side.ctx);
                void *native_beta  = mpfr_scalar_to_native(beta_mpfr, side.ctx);

                auto *fn = reinterpret_cast<gemm_fn_t>(
                    load_sym(side.lib, side.sym.c_str()));
                fn(&ta, &tb, &m, &n, &k,
                   native_alpha, native_A, &lda, native_B, &ldb,
                   native_beta, native_C, &ldc,
                   (std::size_t)1, (std::size_t)1);

                custom_to_mpfr_mat(result, native_C, ldc, side.ctx);

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
                          "transa=%c transb=%c", ta, tb);
            mirror_report_result("GEMM", params_str, err,
                                  nullptr, nullptr, config);
        }
    }
}
