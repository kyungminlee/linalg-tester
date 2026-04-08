/* cgetri.cpp -- LAPACK CGETRI/ZGETRI accuracy tester (complex inverse from LU) */

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

extern "C" typedef void (*getri_fn_t)(
    const int *n, void *A, const int *lda, const int *ipiv,
    void *work, const int *lwork, int *info);

void test_cgetri(const TesterCtx &ctx, void *lib, const char *sym,
                 const TestParams &params, const std::string &format)
{
    /* Derive GETRF symbol from GETRI symbol */
    std::string sym_str(sym);
    std::string routine = "getri";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string getrf_sym = prefix + "getrf" + suffix;

    auto *getrf_fn = reinterpret_cast<getrf_fn_t>(load_sym(lib, getrf_sym.c_str()));
    auto *getri_fn = reinterpret_cast<getri_fn_t>(load_sym(lib, sym));

    int n = params.n;
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = n + params.ld_pad;

    unsigned seed_A = params.seed;
    void *A_orig = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                             ctx.typesize, ctx.from_mpfr_complex,
                                             prec, &seed_A);

    /* Convert A_orig to MPFR */
    MpfrComplexMatrix A_mpfr(n, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_orig, lda, ctx);

    /* Copy for factorization */
    void *A_work = std::malloc(static_cast<std::size_t>(lda) * n * ctx.typesize);
    std::memcpy(A_work, A_orig, static_cast<std::size_t>(lda) * n * ctx.typesize);

    /* GETRF */
    int *ipiv = new int[n];
    int info = 0;
    getrf_fn(&n, &n, A_work, &lda, ipiv, &info);
    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("CGETRI", "GETRF failed", lr, format);
        delete[] ipiv; std::free(A_work); std::free(A_orig);
        return;
    }

    /* Workspace query */
    int lwork_query = -1;
    void *work_q = std::malloc(ctx.typesize);
    getri_fn(&n, A_work, &lda, ipiv, work_q, &lwork_query, &info);
    int lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);

    /* GETRI */
    getri_fn(&n, A_work, &lda, ipiv, work, &lwork, &info);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("CGETRI", "CGETRI failed", lr, format);
    } else {
        /* Convert A_inv to MPFR */
        MpfrComplexMatrix Ainv_mpfr(n, n, prec);
        custom_to_mpfr_complex_mat(Ainv_mpfr, A_work, lda, ctx);

        /* Compute A * A^{-1} - I */
        MpfrComplexMatrix prod(n, n, prec);
        mpfr_complex_mat_mul_simple(prod, A_mpfr, Ainv_mpfr);

        MpfrComplexMatrix I_mat(n, n, prec);
        mpfr_complex_mat_set_identity(I_mat);

        MpfrComplexMatrix R(n, n, prec);
        mpfr_complex_mat_sub(R, prod, I_mat);

        double norm_R = mpfr_complex_mat_norm1(R);
        double norm_A = mpfr_complex_mat_norm1(A_mpfr);
        double norm_Ainv = mpfr_complex_mat_norm1(Ainv_mpfr);

        double residual = 0.0;
        if (norm_A > 0.0 && norm_Ainv > 0.0)
            residual = norm_R / (norm_A * norm_Ainv * n * eps);

        LapackResult lr = {residual, -1.0, info};
        report_lapack_result("CGETRI", "", lr, format);
    }

    delete[] ipiv;
    std::free(work);
    std::free(A_work);
    std::free(A_orig);
}
