/* orgqr.cpp -- Mirror tester for LAPACK ORGQR (generate Q from QR) */

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

extern "C" typedef void (*orgqr_fn_t)(
    const int *m, const int *n, const int *k, void *A, const int *lda,
    const void *tau, void *work, const int *lwork, int *info);

void mirror_test_orgqr(const MirrorSide &a, const MirrorSide &b,
                        const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    mpfr_prec_t prec = config.prec;
    int mn = std::min(m, n);
    int lda = m + params.ld_pad;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    /* For ORGQR: generate Q of size m x mn, with k = mn reflectors */
    int k = mn;
    int n_orgqr = mn;

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        std::string geqrf_sym = replace_suffix(side.sym, "orgqr", "geqrf");
        auto *geqrf = reinterpret_cast<geqrf_fn_t>(
            load_sym(side.lib, geqrf_sym.c_str()));
        auto *orgqr = reinterpret_cast<orgqr_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        void *native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
        void *tau = std::malloc(static_cast<std::size_t>(mn) * side.ctx.typesize);
        int info = 0;

        /* Workspace query for GEQRF */
        int lwork_q = -1;
        void *work_q = std::malloc(side.ctx.typesize);
        geqrf(&m, &n, native_A, &lda, tau, work_q, &lwork_q, &info);
        int lwork_geqrf = mirror_query_lwork(work_q, side.ctx);
        std::free(work_q);

        /* Re-materialize A and run GEQRF */
        std::free(native_A);
        native_A = mpfr_mat_to_native(A_mpfr, lda, side.ctx);
        void *work_geqrf = std::malloc(
            static_cast<std::size_t>(lwork_geqrf) * side.ctx.typesize);
        info = 0;
        geqrf(&m, &n, native_A, &lda, tau, work_geqrf, &lwork_geqrf, &info);
        std::free(work_geqrf);

        /* Workspace query for ORGQR */
        lwork_q = -1;
        work_q = std::malloc(side.ctx.typesize);
        info = 0;
        orgqr(&m, &n_orgqr, &k, native_A, &lda, tau, work_q, &lwork_q, &info);
        int lwork_orgqr = mirror_query_lwork(work_q, side.ctx);
        std::free(work_q);

        /* Run ORGQR */
        void *work_orgqr = std::malloc(
            static_cast<std::size_t>(lwork_orgqr) * side.ctx.typesize);
        info = 0;
        orgqr(&m, &n_orgqr, &k, native_A, &lda, tau, work_orgqr,
              &lwork_orgqr, &info);

        custom_to_mpfr_mat(result, native_A, lda, side.ctx);

        std::free(native_A);
        std::free(tau);
        std::free(work_orgqr);
    };

    MpfrMatrix res_a(m, n_orgqr, prec);
    MpfrMatrix res_b(m, n_orgqr, prec);
    run_side(a, res_a);
    run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str),
                  "m=%d n=%d k=%d", m, n_orgqr, k);
    mirror_report_result("ORGQR", params_str, err,
                          nullptr, nullptr, config);
}
