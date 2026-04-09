/* sytrd.cpp -- Mirror tester for LAPACK SYTRD (symmetric tridiagonal reduction) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" typedef void (*sytrd_fn_t)(
    const char *uplo, const int *n, void *A, const int *lda,
    void *d, void *e, void *tau,
    void *work, const int *lwork, int *info,
    std::size_t uplo_len);

void mirror_test_sytrd(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    mpfr_prec_t prec = config.prec;
    int lda = n + params.ld_pad;
    int e_len = std::max(n - 1, 1);
    int tau_len = std::max(n - 1, 1);

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrMatrix A_mpfr(n, n, prec);
        gen_mpfr_diag_dominant_symmetric(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result,
                            MpfrMatrix &d_out, MpfrMatrix &e_out,
                            MpfrMatrix &tau_out) {
            void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            std::size_t ts = side.ctx.typesize;
            void *d   = std::malloc(static_cast<std::size_t>(n)       * ts);
            void *e   = std::malloc(static_cast<std::size_t>(e_len)   * ts);
            void *tau = std::malloc(static_cast<std::size_t>(tau_len) * ts);
            int info = 0;

            auto *fn = reinterpret_cast<sytrd_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            /* Workspace query */
            int lwork_query = -1;
            void *work_q = std::malloc(ts);
            fn(&uplo, &n, native_A, &lda, d, e, tau,
               work_q, &lwork_query, &info, (std::size_t)1);
            int lwork = mirror_query_lwork(work_q, side.ctx);
            std::free(work_q);

            /* Re-materialize A */
            std::free(native_A);
            native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);

            void *work = std::malloc(static_cast<std::size_t>(lwork) * ts);
            info = 0;
            fn(&uplo, &n, native_A, &lda, d, e, tau,
               work, &lwork, &info, (std::size_t)1);

            custom_to_mpfr_mat(result, native_A, lda, side.ctx);
            native_array_to_mpfr(d_out, d, n, side.ctx);
            native_array_to_mpfr(e_out, e, e_len, side.ctx);
            native_array_to_mpfr(tau_out, tau, tau_len, side.ctx);

            std::free(native_A);
            std::free(d);
            std::free(e);
            std::free(tau);
            std::free(work);
        };

        MpfrMatrix res_a(n, n, prec),   res_b(n, n, prec);
        MpfrMatrix d_a(n, 1, prec),     d_b(n, 1, prec);
        MpfrMatrix e_a(e_len, 1, prec), e_b(e_len, 1, prec);
        MpfrMatrix tau_a(tau_len, 1, prec), tau_b(tau_len, 1, prec);

        run_side(a, res_a, d_a, e_a, tau_a);
        run_side(b, res_b, d_b, e_b, tau_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        ErrorResult d_err = compute_error_mpfr_vector(
            (config.reference == "a") ? d_a : d_b,
            (config.reference == "a") ? d_b : d_a, prec);
        ErrorResult e_err = compute_error_mpfr_vector(
            (config.reference == "a") ? e_a : e_b,
            (config.reference == "a") ? e_b : e_a, prec);
        ErrorResult tau_err = compute_error_mpfr_vector(
            (config.reference == "a") ? tau_a : tau_b,
            (config.reference == "a") ? tau_b : tau_a, prec);

        char params_str[512];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c d_err=%.3e e_err=%.3e tau_err=%.3e",
                      uplo, d_err.max_relative, e_err.max_relative, tau_err.max_relative);
        mirror_report_result("SYTRD", params_str, err,
                              nullptr, nullptr, config);
    }
}
