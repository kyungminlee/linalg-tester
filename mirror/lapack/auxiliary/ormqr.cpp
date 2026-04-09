/* ormqr.cpp -- Mirror tester for LAPACK ORMQR (multiply by Q from QR) */

#include "../mirror_lapack_common.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

static std::string replace_suffix(const std::string &sym, const char *from, const char *to) {
    std::string s = sym;
    auto pos = s.find(from);
    if (pos != std::string::npos)
        s.replace(pos, std::strlen(from), to);
    return s;
}

extern "C" typedef void (*geqrf_fn_t)(
    const int *m, const int *n, void *A, const int *lda,
    void *tau, void *work, const int *lwork, int *info);

extern "C" typedef void (*ormqr_fn_t)(
    const char *side, const char *trans,
    const int *m, const int *n, const int *k,
    const void *A, const int *lda, const void *tau,
    void *C, const int *ldc, void *work, const int *lwork, int *info,
    std::size_t, std::size_t);

void mirror_test_ormqr(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n_param = params.n;
    mpfr_prec_t prec = config.prec;
    int k_qr = std::min(m, n_param);
    int nrhs = std::min(n_param, 4);
    int lda = m + params.ld_pad;
    int ldc = m + params.ld_pad;

    for (char trans : {'N', 'T'}) {
        unsigned seed_A = params.seed;
        unsigned seed_C = params.seed + 2;

        /* A for QR factorization: m x k_qr */
        MpfrMatrix A_mpfr(m, k_qr, prec);
        gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

        /* C: m x nrhs */
        MpfrMatrix C_mpfr(m, nrhs, prec);
        gen_mpfr_random_matrix(C_mpfr, prec, &seed_C);

        char side_ch = 'L';

        auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
            std::string geqrf_sym = replace_suffix(side.sym, "ormqr", "geqrf");
            auto *geqrf = reinterpret_cast<geqrf_fn_t>(
                load_sym(side.lib, geqrf_sym.c_str()));
            auto *ormqr = reinterpret_cast<ormqr_fn_t>(
                load_sym(side.lib, side.sym.c_str()));

            void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *tau = std::malloc(
                static_cast<std::size_t>(k_qr) * side.ctx.typesize);
            int info = 0;

            /* Workspace query for GEQRF */
            int lwork_q = -1;
            void *work_q = std::malloc(side.ctx.typesize);
            geqrf(&m, &k_qr, native_A, &lda, tau, work_q, &lwork_q, &info);
            int lwork_geqrf = mirror_query_lwork(work_q, side.ctx);
            std::free(work_q);

            /* Re-materialize A and run GEQRF */
            std::free(native_A);
            native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
            void *work_geqrf = std::malloc(
                static_cast<std::size_t>(lwork_geqrf) * side.ctx.typesize);
            info = 0;
            geqrf(&m, &k_qr, native_A, &lda, tau, work_geqrf, &lwork_geqrf,
                  &info);
            std::free(work_geqrf);

            /* Materialize C */
            void *native_C = mpfr_mat_to_native(C_mpfr, ldc, side.ctx);

            /* Workspace query for ORMQR */
            lwork_q = -1;
            work_q = std::malloc(side.ctx.typesize);
            info = 0;
            ormqr(&side_ch, &trans, &m, &nrhs, &k_qr,
                  native_A, &lda, tau, native_C, &ldc,
                  work_q, &lwork_q, &info, (std::size_t)1, (std::size_t)1);
            int lwork_ormqr = mirror_query_lwork(work_q, side.ctx);
            std::free(work_q);

            /* Re-materialize C (workspace query may have clobbered it) */
            std::free(native_C);
            native_C = mpfr_mat_to_native(C_mpfr, ldc, side.ctx);

            /* Apply Q */
            void *work_ormqr = std::malloc(
                static_cast<std::size_t>(lwork_ormqr) * side.ctx.typesize);
            info = 0;
            ormqr(&side_ch, &trans, &m, &nrhs, &k_qr,
                  native_A, &lda, tau, native_C, &ldc,
                  work_ormqr, &lwork_ormqr, &info,
                  (std::size_t)1, (std::size_t)1);

            custom_to_mpfr_mat(result, native_C, ldc, side.ctx);

            std::free(native_A);
            std::free(native_C);
            std::free(tau);
            std::free(work_ormqr);
        };

        MpfrMatrix res_a(m, nrhs, prec);
        MpfrMatrix res_b(m, nrhs, prec);
        run_side(a, res_a);
        run_side(b, res_b);

        const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
        const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
        ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "side=L trans=%c m=%d nrhs=%d k=%d", trans, m, nrhs, k_qr);
        mirror_report_result("ORMQR", params_str, err,
                              nullptr, nullptr, config);
    }
}
