/* potrs.cpp -- LAPACK POTRS accuracy tester (solve from Cholesky) */

#include "../auxiliary.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*potrf_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    int *info, std::size_t uplo_len);

extern "C" typedef void (*potrs_fn_t)(
    const char *uplo, const int *n, const int *nrhs,
    const void *A, const int *lda,
    void *B, const int *ldb, int *info,
    std::size_t uplo_len);

void test_potrs(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    std::string sym_str(sym);
    std::string routine = "potrs";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string potrf_sym = prefix + "potrf" + suffix;

    auto *potrf_fn = reinterpret_cast<potrf_fn_t>(load_sym(lib, potrf_sym.c_str()));
    auto *potrs_fn = reinterpret_cast<potrs_fn_t>(load_sym(lib, sym));

    int n = params.n;
    int nrhs = std::min(params.k, 4);
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = n + params.ld_pad;
    int ldb = n + params.ld_pad;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;
        unsigned seed_B = params.seed + 1;

        void *A_spd = gen_positive_definite_array(n, ctx.typesize, ctx.from_mpfr,
                                                   ctx.to_mpfr, prec, &seed_A);

        /* Copy to padded layout if needed */
        void *A_buf = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
        if (lda == n) {
            std::memcpy(A_buf, A_spd, static_cast<std::size_t>(n) * n * ctx.typesize);
        } else {
            std::memset(A_buf, 0, static_cast<std::size_t>(lda) * n * ctx.typesize);
            for (int j = 0; j < n; ++j)
                std::memcpy(static_cast<char *>(A_buf) + j * lda * ctx.typesize,
                            static_cast<char *>(A_spd) + j * n * ctx.typesize,
                            n * ctx.typesize);
        }

        /* Save A_orig in MPFR */
        MpfrMatrix A_mpfr(n, n, prec);
        custom_to_mpfr_mat(A_mpfr, A_buf, lda, ctx);

        void *B_orig = gen_random_array(ldb * nrhs, ctx.typesize, ctx.from_mpfr, prec, &seed_B);
        MpfrMatrix B_mpfr(n, nrhs, prec);
        custom_to_mpfr_mat(B_mpfr, B_orig, ldb, ctx);

        /* POTRF */
        int info = 0;
        potrf_fn(&uplo, &n, A_buf, &lda, &info, (std::size_t)1);
        if (info != 0) {
            char ps[64];
            std::snprintf(ps, sizeof(ps), "uplo=%c POTRF failed", uplo);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("POTRS", ps, lr, format);
            std::free(A_buf); std::free(A_spd); std::free(B_orig);
            continue;
        }

        /* POTRS */
        void *X_buf = std::malloc(static_cast<std::size_t>(ldb) * nrhs * ctx.typesize);
        std::memcpy(X_buf, B_orig, static_cast<std::size_t>(ldb) * nrhs * ctx.typesize);

        potrs_fn(&uplo, &n, &nrhs, A_buf, &lda, X_buf, &ldb, &info, (std::size_t)1);

        MpfrMatrix X_mpfr(n, nrhs, prec);
        custom_to_mpfr_mat(X_mpfr, X_buf, ldb, ctx);

        double residual = mpfr_solve_residual(A_mpfr, X_mpfr, B_mpfr, eps);

        char ps[64];
        std::snprintf(ps, sizeof(ps), "uplo=%c", uplo);
        LapackResult lr = {residual, -1.0, info};
        report_lapack_result("POTRS", ps, lr, format);

        std::free(A_buf); std::free(A_spd); std::free(B_orig); std::free(X_buf);
    }
}
