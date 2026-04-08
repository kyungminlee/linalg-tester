/* cgecon.cpp -- LAPACK CGECON/ZGECON accuracy tester (complex condition number estimate) */

#include "../auxiliary.h"
#include "../lapack_common.h"
#include "../../core/mpfr_complex_types.h"
#include "../../core/mpfr_complex.h"
#include "../../core/mpfr_lapack_complex_utils.h"
#include "../../core/error_metrics.h"
#include "../../core/generators.h"
#include "../../core/loader.h"
#include "../../core/report.h"
#include "../../core/mpfr_lapack_utils.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*getrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    int *ipiv, int *info);

/* ZGECON(norm, n, A, lda, anorm, rcond, work, rwork, info, norm_len) */
extern "C" typedef void (*gecon_fn_t)(
    const char *norm, const int *n, const void *A, const int *lda,
    const void *anorm, void *rcond, void *work, void *rwork,
    int *info, std::size_t norm_len);

void test_cgecon(const TesterCtx &ctx, void *lib, const char *sym,
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
    std::size_t real_typesize = ctx.typesize / 2;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                             ctx.typesize, ctx.from_mpfr_complex,
                                             prec, &seed_A);

    /* Compute ||A||_1 in MPFR */
    MpfrComplexMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_orig, lda, ctx);
    double anorm_d = mpfr_complex_mat_norm1(A_mpfr);

    /* Store anorm in real custom type */
    void *anorm_buf = std::malloc(real_typesize);
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
        report_lapack_result("CGECON", "GETRF failed", lr, format);
        delete[] ipiv; std::free(A_lu); std::free(A_orig); std::free(anorm_buf);
        return;
    }

    /* CGECON/ZGECON */
    char norm_c = '1';
    void *rcond_buf = std::malloc(real_typesize);
    void *work = std::malloc(static_cast<std::size_t>(2 * n) * ctx.typesize);   /* complex work, size 2*n */
    void *rwork = std::malloc(static_cast<std::size_t>(2 * n) * real_typesize); /* real work, size 2*n */

    gecon_fn(&norm_c, &n, A_lu, &lda, anorm_buf, rcond_buf, work, rwork,
             &info, (std::size_t)1);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("CGECON", "norm=1", lr, format);
    } else {
        /* Read rcond (real scalar) */
        MpfrScalar rcond_mpfr(prec);
        ctx.to_mpfr(rcond_mpfr.get(), rcond_buf);
        double rcond = mpfr_get_d(rcond_mpfr.get(), MPFR_RNDN);

        /* Report: residual field shows rcond */
        LapackResult lr;
        lr.residual = rcond;
        lr.orthogonality = -1.0;
        lr.info = info;
        report_lapack_result("CGECON", "norm=1", lr, format);
    }

    std::free(rwork);
    delete[] ipiv;
    std::free(work);
    std::free(rcond_buf);
    std::free(anorm_buf);
    std::free(A_lu);
    std::free(A_orig);
}
