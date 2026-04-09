/* geqrf.cpp -- Mirror tester for LAPACK GEQRF (QR factorization) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*geqrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    void *tau, void *work, const int *lwork, int *info);

void mirror_test_geqrf(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int mn = std::min(m, n);
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result,
                        MpfrMatrix &tau_out) {
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
        void *tau = std::malloc(static_cast<std::size_t>(mn) * side.ctx.typesize);
        int info = 0;

        auto *fn = reinterpret_cast<geqrf_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        /* Workspace query */
        int lwork_query = -1;
        void *work_q = std::malloc(side.ctx.typesize);
        fn(&m, &n, native_A, &lda, tau, work_q, &lwork_query, &info);
        int lwork = mirror_query_lwork(work_q, side.ctx);
        std::free(work_q);

        /* Re-materialize A */
        std::free(native_A);
        native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

        void *work = std::malloc(
            static_cast<std::size_t>(lwork) * side.ctx.typesize);
        info = 0;
        fn(&m, &n, native_A, &lda, tau, work, &lwork, &info);

        custom_to_mpfr_mat(result, native_A, lda, side.ctx);
        native_array_to_mpfr(tau_out, tau, mn, side.ctx);

        std::free(native_A);
        std::free(tau);
        std::free(work);
    };

    MpfrMatrix res_a(m, n, prec);
    MpfrMatrix res_b(m, n, prec);
    MpfrMatrix tau_a(mn, 1, prec);
    MpfrMatrix tau_b(mn, 1, prec);
    run_side(a, res_a, tau_a);
    run_side(b, res_b, tau_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    const MpfrMatrix &tau_ref = (config.reference == "a") ? tau_a : tau_b;
    const MpfrMatrix &tau_tst = (config.reference == "a") ? tau_b : tau_a;
    ErrorResult tau_err = compute_error_mpfr_vector(tau_ref, tau_tst, prec);

    char params_str[256];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d tau_maxerr=%.3e", m, n, tau_err.max_relative);
    mirror_report_result("GEQRF", params_str, err,
                          nullptr, nullptr, config);
}
