/* trsm.cpp -- Mirror tester for BLAS Level 3 TRSM */

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
extern "C" typedef void (*trsm_fn_t)(
    const char *side,   const char *uplo,
    const char *transa, const char *diag,
    const int  *m,      const int  *n,
    const void *alpha,
    const void *A,      const int  *lda,
    void       *B,      const int  *ldb,
    std::size_t side_len,   std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len
);

void mirror_test_trsm(const MirrorSide &a, const MirrorSide &b,
                       const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;

    for (char side : {'L', 'R'}) {
    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'T', 'C'}) {
    for (char diag : {'N', 'U'}) {
        int ka = (side == 'L') ? m : n;

        int lda = ka + params.ld_pad;
        int ldb = m  + params.ld_pad;

        unsigned seed_A  = params.seed + 10;
        unsigned seed_B  = params.seed + 11;
        unsigned seed_al = params.seed + 12;

        /* Generate canonical MPFR data */
        MpfrMatrix A_mpfr(ka, ka, prec);
        MpfrMatrix B_mpfr(m, n, prec);
        MpfrScalar alpha_mpfr(prec);

        gen_mpfr_triangular_matrix(A_mpfr, uplo, diag, prec, &seed_A);
        gen_mpfr_random_matrix(B_mpfr, prec, &seed_B);
        gen_mpfr_random_scalar(alpha_mpfr, prec, &seed_al);

        /* Run one side */
        auto run_side = [&](const MirrorSide &s, MpfrMatrix &result) {
            void *native_A     = mpfr_mat_to_native(A_mpfr, lda, s.ctx);
            void *native_B     = mpfr_mat_to_native(B_mpfr, ldb, s.ctx);
            void *native_alpha = mpfr_scalar_to_native(alpha_mpfr, s.ctx);

            auto *fn = reinterpret_cast<trsm_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&side, &uplo, &trans, &diag,
               &m, &n, native_alpha, native_A, &lda, native_B, &ldb,
               (std::size_t)1, (std::size_t)1,
               (std::size_t)1, (std::size_t)1);

            custom_to_mpfr_mat(result, native_B, ldb, s.ctx);

            std::free(native_A);
            std::free(native_B);
            std::free(native_alpha);
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
                      "side=%c uplo=%c trans=%c diag=%c",
                      side, uplo, trans, diag);
        mirror_report_result("TRSM", params_str, err,
                              nullptr, nullptr, config);
    }}}}
}
