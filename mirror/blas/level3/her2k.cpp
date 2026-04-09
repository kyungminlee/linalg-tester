/* her2k.cpp -- Mirror tester for BLAS Level 3 HER2K (complex-only) */

#include "../mirror_blas3.h"
#include "../../mirror_gen.h"
#include "../../mirror_error.h"
#include "../../mirror_report.h"
#include "../../../src/core/mpfr_types.h"
#include "../../../src/core/mpfr_complex_types.h"
#include "../../../src/core/loader.h"

#include <cstdio>
#include <cstdlib>

/* Fortran-ABI function pointer */
extern "C" typedef void (*her2k_fn_t)(
    const char *uplo,  const char *trans,
    const int  *n,     const int  *k,
    const void *alpha,
    const void *A,     const int  *lda,
    const void *B,     const int  *ldb,
    const void *beta,
    void       *C,     const int  *ldc,
    std::size_t uplo_len, std::size_t trans_len
);

/* Helper: materialize an MpfrScalar as a real native value.
   In complex mode, ctx.typesize is the complex element size (e.g. 16),
   but beta for HER2K is a real scalar of size typesize/2. */
static void *mpfr_real_scalar_to_native(const MpfrScalar &src,
                                         const TesterCtx &ctx)
{
    std::size_t real_ts = ctx.typesize / 2;
    char *arr = static_cast<char *>(std::malloc(real_ts));
    if (!arr) { std::perror("malloc"); std::exit(EXIT_FAILURE); }
    ctx.from_mpfr(arr, const_cast<mpfr_t &>(src.get()), MPFR_RNDN);
    return static_cast<void *>(arr);
}

void mirror_test_her2k(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n, k = params.k;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
    for (char trans : {'N', 'C'}) {
        int rows_AB = (trans == 'N') ? n : k;
        int cols_AB = (trans == 'N') ? k : n;

        int lda = rows_AB + params.ld_pad;
        int ldb = rows_AB + params.ld_pad;
        int ldc = n       + params.ld_pad;

        unsigned seed_A  = params.seed;
        unsigned seed_B  = params.seed + 1;
        unsigned seed_C  = params.seed + 2;
        unsigned seed_al = params.seed + 3;
        unsigned seed_be = params.seed + 4;

        /* Generate canonical MPFR data */
        MpfrComplexMatrix A_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix B_mpfr(rows_AB, cols_AB, prec);
        MpfrComplexMatrix C_mpfr(n, n, prec);
        MpfrComplexScalar alpha_mpfr(prec);
        MpfrScalar beta_mpfr(prec);

        gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);
        gen_mpfr_random_complex_matrix(B_mpfr, prec, &seed_B);
        gen_mpfr_hermitian_matrix(C_mpfr, uplo, prec, &seed_C);
        gen_mpfr_random_complex_scalar(alpha_mpfr, prec, &seed_al);
        gen_mpfr_random_scalar(beta_mpfr, prec, &seed_be);

        /* Run one side */
        auto run_side = [&](const MirrorSide &s, MpfrComplexMatrix &result) {
            void *native_A     = mpfr_complex_mat_to_native(A_mpfr, lda, s.ctx);
            void *native_B     = mpfr_complex_mat_to_native(B_mpfr, ldb, s.ctx);
            void *native_C     = mpfr_complex_mat_to_native(C_mpfr, ldc, s.ctx);
            void *native_alpha = mpfr_complex_scalar_to_native(alpha_mpfr, s.ctx);
            void *native_beta  = mpfr_real_scalar_to_native(beta_mpfr, s.ctx);

            auto *fn = reinterpret_cast<her2k_fn_t>(
                load_sym(s.lib, s.sym.c_str()));
            fn(&uplo, &trans, &n, &k,
               native_alpha, native_A, &lda, native_B, &ldb,
               native_beta, native_C, &ldc,
               (std::size_t)1, (std::size_t)1);

            custom_to_mpfr_complex_mat(result, native_C, ldc, s.ctx);

            std::free(native_A);
            std::free(native_B);
            std::free(native_C);
            std::free(native_alpha);
            std::free(native_beta);
        };

        MpfrComplexMatrix res_a(n, n, prec);
        MpfrComplexMatrix res_b(n, n, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrComplexMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrComplexMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_complex_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c trans=%c", uplo, trans);
        mirror_report_result("HER2K", params_str, err,
                              nullptr, nullptr, config);
    }}
}
