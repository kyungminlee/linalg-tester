/* gecon.cpp -- LAPACK GECON accuracy tester (condition number estimate) */

#include "../auxiliary.h"
#include "../../core/mpfr_types.h"
#include "../../core/mpfr_lapack_utils.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../lapack_common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*getrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

extern "C" typedef void (*gecon_fn_t)(
    const char *norm, const int *n, const void *A, const int *lda,
    const void *anorm, void *rcond, void *work, int *iwork,
    int *info, std::size_t norm_len);

void test_gecon(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    std::string sym_str(sym);
    std::string routine = "gecon";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string getrf_sym = prefix + "getrf" + suffix;

    auto *getrf_fn = reinterpret_cast<getrf_fn_t>(load_sym(lib, getrf_sym.c_str()));
    auto *gecon_fn = reinterpret_cast<gecon_fn_t>(load_sym(lib, sym));

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_array(lda * n, ctx.typesize, ctx.from_mpfr, prec, &seed_A);

    /* Compute ||A||_1 in MPFR */
    MpfrMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_mat(A_mpfr, A_orig, lda, ctx);
    double anorm_d = mpfr_mat_norm1(A_mpfr);

    /* Store anorm in custom type */
    void *anorm_buf = std::malloc(ctx.typesize);
    {
        MpfrScalar tmp(prec);
        mpfr_set_d(tmp.get(), anorm_d, MPFR_RNDN);
        ctx.from_mpfr(anorm_buf, tmp.get(), MPFR_RNDN);
    }

    /* GETRF */
    void *A_lu = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_lu, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    int *ipiv = new int[n];
    int info = 0;
    getrf_fn(&n, &n, A_lu, &lda, ipiv, &info);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("GECON", "GETRF failed", lr, format);
        delete[] ipiv; std::free(A_lu); std::free(A_orig); std::free(anorm_buf);
        return;
    }

    /* GECON */
    char norm_c = '1';
    void *rcond_buf = std::malloc(ctx.typesize);
    void *work = std::malloc(static_cast<std::size_t>(4 * n) * ctx.typesize);
    int *iwork = new int[n];

    gecon_fn(&norm_c, &n, A_lu, &lda, anorm_buf, rcond_buf, work, iwork,
             &info, (std::size_t)1);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("GECON", "norm=1", lr, format);
    } else {
        /* Read rcond */
        MpfrScalar rcond_mpfr(prec);
        ctx.to_mpfr(rcond_mpfr.get(), rcond_buf);
        double rcond = mpfr_get_d(rcond_mpfr.get(), MPFR_RNDN);

        /* Report rcond as the "residual" (it's an estimate, not exact).
           A good estimate should have rcond > 0 for non-singular matrices.
           We report 1/rcond as an indicator of the condition number. */
        ErrorResult err;
        err.max_relative = rcond;
        err.normwise_relative = (rcond > 0.0) ? 1.0 / rcond : 1e30;
        report_result("GECON", "norm=1 (rcond, 1/rcond)", err, format);
    }

    delete[] iwork;
    delete[] ipiv;
    std::free(work);
    std::free(rcond_buf);
    std::free(anorm_buf);
    std::free(A_lu);
    std::free(A_orig);
}
