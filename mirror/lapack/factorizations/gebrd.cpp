/* gebrd.cpp -- Mirror tester for LAPACK GEBRD (bidiagonal reduction) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*gebrd_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    void *d, void *e, void *tauq, void *taup,
    void *work, const int *lwork, int *info);

void mirror_test_gebrd(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int mn = std::min(m, n);
    int e_len = std::max(mn - 1, 1);
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result,
                        MpfrMatrix &d_out, MpfrMatrix &e_out,
                        MpfrMatrix &tauq_out, MpfrMatrix &taup_out) {
        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
        std::size_t ts = side.ctx.typesize;
        void *d    = std::malloc(static_cast<std::size_t>(mn)    * ts);
        void *e    = std::malloc(static_cast<std::size_t>(e_len) * ts);
        void *tauq = std::malloc(static_cast<std::size_t>(mn)    * ts);
        void *taup = std::malloc(static_cast<std::size_t>(mn)    * ts);
        int info = 0;

        auto *fn = reinterpret_cast<gebrd_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        /* Workspace query */
        int lwork_query = -1;
        void *work_q = std::malloc(ts);
        fn(&m, &n, native_A, &lda, d, e, tauq, taup,
           work_q, &lwork_query, &info);
        int lwork = mirror_query_lwork(work_q, side.ctx);
        std::free(work_q);

        /* Re-materialize A */
        std::free(native_A);
        native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

        void *work = std::malloc(static_cast<std::size_t>(lwork) * ts);
        info = 0;
        fn(&m, &n, native_A, &lda, d, e, tauq, taup,
           work, &lwork, &info);

        custom_to_mpfr_mat(result, native_A, lda, side.ctx);
        native_array_to_mpfr(d_out, d, mn, side.ctx);
        native_array_to_mpfr(e_out, e, e_len, side.ctx);
        native_array_to_mpfr(tauq_out, tauq, mn, side.ctx);
        native_array_to_mpfr(taup_out, taup, mn, side.ctx);

        std::free(native_A);
        std::free(d);
        std::free(e);
        std::free(tauq);
        std::free(taup);
        std::free(work);
    };

    MpfrMatrix res_a(m, n, prec),  res_b(m, n, prec);
    MpfrMatrix d_a(mn, 1, prec),   d_b(mn, 1, prec);
    MpfrMatrix e_a(e_len, 1, prec), e_b(e_len, 1, prec);
    MpfrMatrix tauq_a(mn, 1, prec), tauq_b(mn, 1, prec);
    MpfrMatrix taup_a(mn, 1, prec), taup_b(mn, 1, prec);

    run_side(a, res_a, d_a, e_a, tauq_a, taup_a);
    run_side(b, res_b, d_b, e_b, tauq_b, taup_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    ErrorResult d_err = compute_error_mpfr_vector(
        (config.reference == "a") ? d_a : d_b,
        (config.reference == "a") ? d_b : d_a, prec);
    ErrorResult e_err = compute_error_mpfr_vector(
        (config.reference == "a") ? e_a : e_b,
        (config.reference == "a") ? e_b : e_a, prec);
    ErrorResult tauq_err = compute_error_mpfr_vector(
        (config.reference == "a") ? tauq_a : tauq_b,
        (config.reference == "a") ? tauq_b : tauq_a, prec);
    ErrorResult taup_err = compute_error_mpfr_vector(
        (config.reference == "a") ? taup_a : taup_b,
        (config.reference == "a") ? taup_b : taup_a, prec);

    char params_str[512];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d d_err=%.3e e_err=%.3e tauq_err=%.3e taup_err=%.3e",
                  m, n, d_err.max_relative, e_err.max_relative,
                  tauq_err.max_relative, taup_err.max_relative);
    mirror_report_result("GEBRD", params_str, err,
                          nullptr, nullptr, config);
}
