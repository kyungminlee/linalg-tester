/* potrf.cpp -- LAPACK POTRF accuracy tester (Cholesky factorization) */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_potrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int lda = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        /* Generate SPD matrix (returned as n-by-n, ld=n) */
        void *A_spd = gen_positive_definite_array(n, ctx.typesize,
                                                   ctx.from_mpfr, ctx.to_mpfr,
                                                   prec, &seed_A);

        /* Copy into lda-padded layout if needed */
        void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        {
            char *dst = static_cast<char *>(A);
            const char *src = static_cast<const char *>(A_spd);
            for (int j = 0; j < n; ++j)
                std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                            src + static_cast<std::size_t>(j) * n * ctx.typesize,
                            static_cast<std::size_t>(n) * ctx.typesize);
        }

        /* Save A_orig in MPFR */
        MpfrMatrix A_orig(n, n, prec);
        custom_to_mpfr_mat(A_orig, A, lda, ctx);

        /* Call POTRF: void POTRF(char *uplo, int *n, T *A, int *lda, int *info, size_t) */
        int info = 0;
        auto *fn = reinterpret_cast<void (*)(
            const char *, const int *, void *, const int *,
            int *, std::size_t)>(load_sym(lib, sym));
        fn(&uplo, &n, A, &lda, &info, (std::size_t)1);

        /* Convert factored A to MPFR */
        MpfrMatrix A_fact(n, n, prec);
        custom_to_mpfr_mat(A_fact, A, lda, ctx);

        /* Extract triangular factor and compute residual */
        MpfrMatrix T(n, n, prec);
        mpfr_extract_triangle(A_fact, T, uplo);

        MpfrMatrix Tt(n, n, prec);
        mpfr_mat_transpose(Tt, T);

        /* For 'L': A = L * L^T;  for 'U': A = U^T * U */
        MpfrMatrix Recon(n, n, prec);
        if (uplo == 'L')
            mpfr_mat_mul_simple(Recon, T, Tt);
        else
            mpfr_mat_mul_simple(Recon, Tt, T);

        MpfrMatrix Resid(n, n, prec);
        mpfr_mat_sub(Resid, A_orig, Recon);
        double norm_resid = mpfr_mat_norm1(Resid);
        double norm_A = mpfr_mat_norm1(A_orig);

        double residual = (norm_A > 0.0) ? norm_resid / (norm_A * n * eps) : 0.0;

        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;
        lr.info = info;

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str), "uplo=%c n=%d", uplo, n);
        report_lapack_result("POTRF", params_str, lr, format);

        std::free(A);
        std::free(A_spd);
    }
}
