/* unmqr.cpp -- LAPACK CUNMQR/ZUNMQR accuracy tester (multiply by Q from complex QR) */

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

extern "C" typedef void (*unmqr_fn_t)(
    const char *side, const char *trans,
    const int *m, const int *n, const int *k,
    const void *A, const int *lda, const void *tau,
    void *C, const int *ldc,
    void *work, const int *lwork, int *info,
    std::size_t side_len, std::size_t trans_len);

void test_unmqr(const TesterCtx &ctx, void *lib, const char *sym,
                const TestParams &params, const std::string &format)
{
    std::string sym_str(sym);
    std::string routine = "unmqr";
    std::size_t pos = sym_str.find(routine);
    std::string prefix = sym_str.substr(0, pos);
    std::string suffix = sym_str.substr(pos + routine.size());
    std::string geqrf_sym = prefix + "geqrf" + suffix;

    auto *geqrf_fn = reinterpret_cast<geqrf_fn_t>(load_sym(lib, geqrf_sym.c_str()));
    auto *unmqr_fn = reinterpret_cast<unmqr_fn_t>(load_sym(lib, sym));

    int qm = params.m;
    int qn = std::min(params.n, params.m); /* For QR of qm-by-qn */
    int k_ref = qn; /* number of reflectors */
    mpfr_prec_t prec = ctx.prec;
    double eps = get_eps(ctx);

    /* Do QR factorization first */
    int lda_qr = qm + params.ld_pad;
    unsigned seed_A = params.seed;
    void *A_qr = gen_random_complex_array(static_cast<std::size_t>(lda_qr) * qn,
                                           ctx.typesize, ctx.from_mpfr_complex,
                                           prec, &seed_A);

    void *tau = std::malloc(static_cast<std::size_t>(k_ref) * ctx.typesize);
    int info = 0;

    int lwork_q = -1;
    void *work_q = std::malloc(ctx.typesize);
    geqrf_fn(&qm, &qn, A_qr, &lda_qr, tau, work_q, &lwork_q, &info);
    int lwork = query_lwork_complex(work_q, ctx);
    std::free(work_q);
    void *work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
    geqrf_fn(&qm, &qn, A_qr, &lda_qr, tau, work, &lwork, &info);
    std::free(work);

    if (info != 0) {
        LapackResult lr = {0.0, -1.0, info};
        report_lapack_result("UNMQR", "GEQRF failed", lr, format);
        std::free(tau); std::free(A_qr);
        return;
    }

    /* Build MPFR Q from reflectors */
    MpfrComplexMatrix A_qr_mpfr(qm, qn, prec);
    custom_to_mpfr_complex_mat(A_qr_mpfr, A_qr, lda_qr, ctx);

    MpfrComplexMatrix tau_mpfr(k_ref, 1, prec);
    const char *tp = static_cast<const char *>(tau);
    for (int i = 0; i < k_ref; ++i)
        ctx.to_mpfr_complex(tau_mpfr.re(i, 0), tau_mpfr.im(i, 0),
                            tp + i * ctx.typesize);

    MpfrComplexMatrix Q_full(qm, qm, prec);
    mpfr_complex_accumulate_Q_from_QR(Q_full, A_qr_mpfr, tau_mpfr, qm, qn);

    /* Test side='L', trans='N' and trans='C': C_out = Q*C or Q^H*C */
    for (char trans : {'N', 'C'}) {
        int cm = qm, cn = std::min(params.k, 4);
        int ldc = cm + params.ld_pad;

        unsigned seed_C = params.seed + 10;
        void *C_orig = gen_random_complex_array(static_cast<std::size_t>(ldc) * cn,
                                                 ctx.typesize, ctx.from_mpfr_complex,
                                                 prec, &seed_C);

        MpfrComplexMatrix C_mpfr(cm, cn, prec);
        custom_to_mpfr_complex_mat(C_mpfr, C_orig, ldc, ctx);

        void *C_buf = std::malloc(static_cast<std::size_t>(ldc) * cn * ctx.typesize);
        std::memcpy(C_buf, C_orig, static_cast<std::size_t>(ldc) * cn * ctx.typesize);

        char side = 'L';
        lwork_q = -1;
        work_q = std::malloc(ctx.typesize);
        unmqr_fn(&side, &trans, &cm, &cn, &k_ref,
                  A_qr, &lda_qr, tau, C_buf, &ldc,
                  work_q, &lwork_q, &info,
                  (std::size_t)1, (std::size_t)1);
        lwork = query_lwork_complex(work_q, ctx);
        std::free(work_q);

        work = std::malloc(static_cast<std::size_t>(lwork) * ctx.typesize);
        unmqr_fn(&side, &trans, &cm, &cn, &k_ref,
                  A_qr, &lda_qr, tau, C_buf, &ldc,
                  work, &lwork, &info,
                  (std::size_t)1, (std::size_t)1);
        std::free(work);

        if (info != 0) {
            char ps[64]; std::snprintf(ps, sizeof(ps), "side=L trans=%c", trans);
            LapackResult lr = {0.0, -1.0, info};
            report_lapack_result("UNMQR", ps, lr, format);
        } else {
            /* MPFR reference: Q*C or Q^H*C */
            MpfrComplexMatrix ref(cm, cn, prec);
            if (trans == 'N') {
                mpfr_complex_mat_mul_simple(ref, Q_full, C_mpfr);
            } else {
                MpfrComplexMatrix Qh(qm, qm, prec);
                mpfr_complex_mat_adjoint(Qh, Q_full);
                mpfr_complex_mat_mul_simple(ref, Qh, C_mpfr);
            }

            MpfrComplexMatrix C_out_mpfr(cm, cn, prec);
            custom_to_mpfr_complex_mat(C_out_mpfr, C_buf, ldc, ctx);

            MpfrComplexMatrix diff(cm, cn, prec);
            mpfr_complex_mat_sub(diff, C_out_mpfr, ref);
            double norm_diff = mpfr_complex_mat_norm1(diff);
            double norm_ref = mpfr_complex_mat_norm1(ref);
            double residual = (norm_ref > 0.0) ? norm_diff / (norm_ref * cm * eps) : 0.0;

            char ps[64]; std::snprintf(ps, sizeof(ps), "side=L trans=%c", trans);
            LapackResult lr = {residual, -1.0, info};
            report_lapack_result("UNMQR", ps, lr, format);
        }

        std::free(C_orig);
        std::free(C_buf);
    }

    std::free(tau);
    std::free(A_qr);
}
