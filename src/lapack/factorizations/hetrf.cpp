/* hetrf.cpp -- LAPACK complex HETRF accuracy tester (Hermitian indefinite factorization) */
/* Verification via solve residual: factor with HETRF, solve with HETRS,
   check ||AX - B|| / (||A|| * ||X|| * n * eps). */

#include "../factorizations.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"  // for get_eps

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

/* ------------------------------------------------------------------ */
/* Public entry point                                                   */
/* ------------------------------------------------------------------ */

void test_hetrf(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    int n = params.n;
    int lda = n + params.ld_pad;
    int nrhs = 1;
    int ldb = n + params.ld_pad;

    /* Derive HETRS symbol name from HETRF symbol */
    std::string hetrs_sym(sym);
    {
        /* Replace last occurrence of "trf" with "trs" */
        std::size_t pos = hetrs_sym.rfind("trf");
        if (pos == std::string::npos)
            pos = hetrs_sym.rfind("TRF");
        if (pos != std::string::npos)
            hetrs_sym.replace(pos, 3, (hetrs_sym[pos] == 'T') ? "TRS" : "trs");
    }

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        /* Generate Hermitian matrix A (n-by-n, ld=n) */
        void *A_her = gen_hermitian_array(n, uplo, ctx.typesize,
                                          ctx.from_mpfr_complex, prec, &seed_A);

        /* Copy into lda-padded layout */
        void *A = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        {
            char *dst = static_cast<char *>(A);
            const char *src = static_cast<const char *>(A_her);
            for (int j = 0; j < n; ++j)
                std::memcpy(dst + static_cast<std::size_t>(j) * lda * ctx.typesize,
                            src + static_cast<std::size_t>(j) * n * ctx.typesize,
                            static_cast<std::size_t>(n) * ctx.typesize);
        }

        /* Save A_orig in MPFR (full Hermitian matrix for residual check) */
        MpfrComplexMatrix A_orig(n, n, prec);
        {
            const char *p = static_cast<const char *>(A);
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < n; ++i)
                    ctx.to_mpfr_complex(A_orig.re(i, j), A_orig.im(i, j),
                                        p + IDX(i, j, lda) * ctx.typesize);
            /* Fill the other triangle by Hermitian symmetry: A(i,j) = conj(A(j,i)) */
            for (int j = 0; j < n; ++j)
                for (int i = 0; i < j; ++i) {
                    if (uplo == 'U') {
                        /* Upper stored: copy conj(A(i,j)) to A(j,i) */
                        mpfr_complex_conj(A_orig.re(j, i), A_orig.im(j, i),
                                          A_orig.re(i, j), A_orig.im(i, j),
                                          MPFR_RNDN);
                    } else {
                        /* Lower stored: copy conj(A(j,i)) to A(i,j) */
                        mpfr_complex_conj(A_orig.re(i, j), A_orig.im(i, j),
                                          A_orig.re(j, i), A_orig.im(j, i),
                                          MPFR_RNDN);
                    }
                }
        }

        /* Generate random complex RHS B */
        void *B = gen_random_complex_array(static_cast<std::size_t>(ldb) * nrhs,
                                           ctx.typesize, ctx.from_mpfr_complex, prec, &seed_B);

        /* Save B_orig in MPFR */
        MpfrComplexMatrix B_orig(n, nrhs, prec);
        custom_to_mpfr_complex_mat(B_orig, B, ldb, ctx);

        /* Allocate IPIV and workspace */
        int *ipiv = new int[n];
        int info = 0;

        /* Workspace query */
        int lwork_query = -1;
        void *work_query = std::malloc(ctx.typesize);
        auto *fn_hetrf = reinterpret_cast<void (*)(
            const char *, const int *, void *, const int *,
            int *, void *, const int *, int *, std::size_t)>(load_sym(lib, sym));
        fn_hetrf(&uplo, &n, A, &lda, ipiv, work_query, &lwork_query, &info,
                 (std::size_t)1);
        int lwork = query_lwork_complex(work_query, ctx);
        std::free(work_query);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

        /* Call HETRF */
        info = 0;
        fn_hetrf(&uplo, &n, A, &lda, ipiv, work, &lwork, &info,
                 (std::size_t)1);

        if (info != 0) {
            /* Factorization failed or matrix is singular -- report and move on */
            LapackResult lr;
            lr.residual = -1.0;
            lr.orthogonality = -1.0;
            lr.info = info;

            char params_str[128];
            std::snprintf(params_str, sizeof(params_str), "uplo=%c n=%d", uplo, n);
            report_lapack_result("HETRF", params_str, lr, format);

            std::free(A);
            std::free(A_her);
            std::free(B);
            std::free(work);
            delete[] ipiv;
            continue;
        }

        /* Call HETRS to solve AX = B using the factored A */
        auto *fn_hetrs = reinterpret_cast<void (*)(
            const char *, const int *, const int *, const void *, const int *,
            const int *, void *, const int *, int *, std::size_t)>(
            load_sym(lib, hetrs_sym.c_str()));
        fn_hetrs(&uplo, &n, &nrhs, A, &lda, ipiv, B, &ldb, &info,
                 (std::size_t)1);

        /* Convert solution X to MPFR */
        MpfrComplexMatrix X(n, nrhs, prec);
        custom_to_mpfr_complex_mat(X, B, ldb, ctx);

        /* Compute solve residual: ||AX - B|| / (||A|| * ||X|| * n * eps) */
        double residual = mpfr_complex_solve_residual(A_orig, X, B_orig, eps);

        LapackResult lr;
        lr.residual = residual;
        lr.orthogonality = -1.0;
        lr.info = info;

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str), "uplo=%c n=%d", uplo, n);
        report_lapack_result("HETRF", params_str, lr, format);

        std::free(A);
        std::free(A_her);
        std::free(B);
        std::free(work);
        delete[] ipiv;
    }
}
