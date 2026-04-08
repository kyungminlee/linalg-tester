/* getrs.cpp -- LAPACK GETRS accuracy tester (solve using LU factors) */

#include "../auxiliary.h"
#include "../lapack_common.h"
#include "../../core/mpfr_types.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* Fortran ABI: GETRF(m, n, A, lda, ipiv, info) */
extern "C" typedef void (*getrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info
);

/* Fortran ABI: GETRS(trans, n, nrhs, A, lda, ipiv, B, ldb, info, trans_len) */
extern "C" typedef void (*getrs_fn_t)(
    const char *trans, const int *n, const int *nrhs,
    const void *A, const int *lda, const int *ipiv,
    void *B, const int *ldb, int *info,
    std::size_t trans_len
);

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_getrs(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;

    int n = params.n;
    int nrhs = params.k;  /* use k as nrhs */
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    /* Derive GETRF symbol from GETRS symbol */
    std::string sym_str(sym);
    std::string routine_lower = "getrs";
    std::size_t pos = sym_str.find(routine_lower);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine_lower.size());
    std::string getrf_sym = prefix + "getrf" + suffix;

    auto *getrf = reinterpret_cast<getrf_fn_t>(load_sym(lib, getrf_sym.c_str()));
    auto *getrs = reinterpret_cast<getrs_fn_t>(load_sym(lib, sym));

    for (char trans : {'N', 'T'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        void *A_orig = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);
        void *B_orig = gen_random_array(ldb * nrhs, ctx.typesize, ctx.from_mpfr, prec, &seed_B);

        /* Save A and B in MPFR before factoring */
        MpfrMatrix A_mpfr(n, n, prec);
        MpfrMatrix B_mpfr(n, nrhs, prec);
        custom_to_mpfr_mat(A_mpfr, A_orig, lda, ctx);
        custom_to_mpfr_mat(B_mpfr, B_orig, ldb, ctx);

        /* Factor A in place using GETRF */
        void *A_lu = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        std::memcpy(A_lu, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

        int *ipiv = static_cast<int *>(std::malloc(static_cast<std::size_t>(n) * sizeof(int)));
        int info = 0;

        getrf(&n, &n, A_lu, &lda, ipiv, &info);
        if (info != 0) {
            char params_str[128];
            std::snprintf(params_str, sizeof(params_str),
                          "trans=%c n=%d nrhs=%d GETRF_INFO=%d", trans, n, nrhs, info);
            ErrorResult err_zero = {0.0, 0.0};
            report_result("GETRS", params_str, err_zero, format);
            std::free(A_orig); std::free(B_orig);
            std::free(A_lu); std::free(ipiv);
            continue;
        }

        /* Solve: B_out = A_lu \ B_orig */
        void *B_out = std::malloc(static_cast<std::size_t>(ldb) * nrhs * ctx.typesize);
        std::memcpy(B_out, B_orig, static_cast<std::size_t>(ldb) * nrhs * ctx.typesize);

        info = 0;
        getrs(&trans, &n, &nrhs, A_lu, &lda, ipiv, B_out, &ldb, &info,
              (std::size_t)1);

        /* Convert solution X to MPFR */
        MpfrMatrix X_mpfr(n, nrhs, prec);
        custom_to_mpfr_mat(X_mpfr, B_out, ldb, ctx);

        /* Compute residual: ||A*X - B|| / (||A|| * ||X|| * n * eps)
           or ||A^T*X - B|| / (||A|| * ||X|| * n * eps) for trans='T' */
        double eps = get_eps(ctx);
        double residual;

        if (trans == 'N') {
            residual = mpfr_solve_residual(A_mpfr, X_mpfr, B_mpfr, eps);
        } else {
            /* For A^T * X = B, compute ||A^T * X - B|| */
            MpfrMatrix At(n, n, prec);
            mpfr_mat_transpose(At, A_mpfr);
            residual = mpfr_solve_residual(At, X_mpfr, B_mpfr, eps);
        }

        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;
        lr.info = info;

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "trans=%c n=%d nrhs=%d", trans, n, nrhs);
        report_lapack_result("GETRS", params_str, lr, format);

        std::free(A_orig); std::free(B_orig);
        std::free(A_lu); std::free(B_out); std::free(ipiv);
    }
}
