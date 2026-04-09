/* pgeqrf.cpp -- Mirror tester for ScaLAPACK PGEQRF (distributed QR factorization) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pgeqrf_fn_t)(
    const int *m, const int *n, void *A, const int *ia, const int *ja,
    const int *desca, void *tau, void *work, const int *lwork, int *info);

void mirror_test_pgeqrf(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;
    int mn = std::min(m, n);

    unsigned seed_A = params.seed;
    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &side, MpfrMatrix &result) {
        MirrorPblasCtx mpc;
        if (!mpc.init(side, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         side.label.c_str());
            return;
        }

        int loc_m = mpc.local_rows(m);
        int lld_a = std::max(1, loc_m);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, m, n, lld_a, mpc, side.ctx);

        int desc_a[9];
        mpc.make_desc(desc_a, m, n, lld_a);

        int tau_len = mn + nb;
        void *tau = std::calloc(tau_len, side.ctx.typesize);
        int info = 0, one = 1;

        auto *fn = reinterpret_cast<pgeqrf_fn_t>(
            load_sym(side.lib, side.sym.c_str()));

        /* Workspace query */
        int lwork_query = -1;
        void *work_q = std::malloc(side.ctx.typesize);
        fn(&m, &n, A_loc, &one, &one, desc_a, tau, work_q, &lwork_query, &info);
        int lwork = mirror_query_lwork(work_q, side.ctx);
        std::free(work_q);

        /* Re-scatter A after workspace query (query may modify input) */
        std::free(A_loc);
        A_loc = scatter_mpfr_to_local(A_mpfr, m, n, lld_a, mpc, side.ctx);
        std::memset(tau, 0, static_cast<std::size_t>(tau_len) * side.ctx.typesize);

        void *work = std::malloc(
            static_cast<std::size_t>(lwork) * side.ctx.typesize);
        info = 0;
        fn(&m, &n, A_loc, &one, &one, desc_a, tau, work, &lwork, &info);

        gather_local_to_mpfr(result, A_loc, lld_a, m, n, mpc, side.ctx);

        std::free(A_loc);
        std::free(tau);
        std::free(work);
        mpc.finalize();
    };

    MpfrMatrix res_a(m, n, prec); run_side(a, res_a);
    MpfrMatrix res_b(m, n, prec); run_side(b, res_b);

    const MpfrMatrix &ref = (config.reference == "a") ? res_a : res_b;
    const MpfrMatrix &tst = (config.reference == "a") ? res_b : res_a;
    ErrorResult err = compute_error_mpfr_matrix(ref, tst, prec);

    char ps[128];
    std::snprintf(ps, sizeof(ps), "m=%d n=%d ref=%s",
                  m, n, (config.reference == "a") ? "A" : "B");
    mirror_report_result("PGEQRF", ps, err, nullptr, nullptr, config);
}
