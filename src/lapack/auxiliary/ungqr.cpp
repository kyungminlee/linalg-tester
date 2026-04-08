/* ungqr.cpp -- LAPACK CUNGQR/ZUNGQR accuracy tester (generate Q from complex QR) */

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

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

extern "C" typedef void (*geqrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    void *tau, void *work, const int *lwork, int *info);

extern "C" typedef void (*ungqr_fn_t)(
    const int *m, const int *n, const int *k,
    void *A, const int *lda, const void *tau,
    void *work, const int *lwork, int *info);

void test_ungqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    std::string sym_str(sym);
    std::string routine = "ungqr";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string geqrf_sym = prefix + "geqrf" + suffix;

    auto *geqrf_fn = reinterpret_cast<geqrf_fn_t>(load_sym(lib, geqrf_sym.c_str()));
    auto *ungqr_fn = reinterpret_cast<ungqr_fn_t>(load_sym(lib, sym));

    int m = params.m, n = std::min(params.n, params.m);
    int k = n; /* number of reflectors = min(m, n) */
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;
    void *A_buf = gen_random_complex_array(static_cast<std::size_t>(lda) * n,
                                            ctx.typesize, ctx.from_mpfr_complex,
                                            prec, &seed_A);

    /* Save original A for MPFR reference */
    MpfrComplexMatrix A_mpfr(m, n, prec);
    custom_to_mpfr_complex_mat(A_mpfr, A_buf, lda, ctx);

    /* GEQRF */
    void *tau = std::malloc(static_cast<std::size_t>(k) * ctx.typesize);
    int info = 0;

    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    geqrf_fn(&m, &n, A_buf, &lda, tau, work_q, &lwork_q, &info);
    int lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);

    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    geqrf_fn(&m, &n, A_buf, &lda, tau, work, &lwork, &info);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("UNGQR", "GEQRF failed", lr, format);
        std::free(tau); std::free(A_buf);
        return;
    }

    /* Build MPFR reference Q from reflectors */
    MpfrComplexMatrix A_qr(m, n, prec);
    custom_to_mpfr_complex_mat(A_qr, A_buf, lda, ctx);

    MpfrComplexMatrix tau_mpfr(k, 1, prec);
    const char *tp = static_cast<const char *>(tau);
    for (int i = 0; i < k; ++i)
        ctx.to_mpfr_complex(tau_mpfr.re(i, 0), tau_mpfr.im(i, 0),
                            tp + i * ctx.typesize);

    MpfrComplexMatrix Q_ref(m, m, prec);
    mpfr_complex_accumulate_Q_from_QR(Q_ref, A_qr, tau_mpfr, m, n);

    /* UNGQR: generate Q in A_buf */
    lwork_q = -1;
    work_q = std::malloc(ctx.typesize);
    ungqr_fn(&m, &n, &k, A_buf, &lda, tau, work_q, &lwork_q, &info);
    lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);

    work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    ungqr_fn(&m, &n, &k, A_buf, &lda, tau, work, &lwork, &info);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("UNGQR", "UNGQR failed", lr, format);
    } else {
        /* Compare computed Q (first n columns) against reference */
        MpfrComplexMatrix Q_lib(m, n, prec);
        custom_to_mpfr_complex_mat(Q_lib, A_buf, lda, ctx);

        /* Orthogonality check */
        double orth = mpfr_complex_orthogonality(Q_lib, eps);

        /* Compare against MPFR reference Q (first n columns) */
        MpfrComplexMatrix Q_ref_n(m, n, prec);
        for (int j = 0; j < n; ++j)
            for (int i = 0; i < m; ++i) {
                mpfr_set(Q_ref_n.re(i, j), Q_ref.re(i, j), MPFR_RNDN);
                mpfr_set(Q_ref_n.im(i, j), Q_ref.im(i, j), MPFR_RNDN);
            }

        MpfrComplexMatrix diff(m, n, prec);
        mpfr_complex_mat_sub(diff, Q_lib, Q_ref_n);
        double norm_diff = mpfr_complex_mat_norm1(diff);
        double norm_Q = mpfr_complex_mat_norm1(Q_ref_n);
        double residual = (norm_Q > 0.0) ? norm_diff / (norm_Q * n * eps) : 0.0;

        LapackResult lr = {residual, orth, info};
        report_lapack_result("UNGQR", "", lr, format);
    }

    std::free(tau);
    std::free(A_buf);
}
