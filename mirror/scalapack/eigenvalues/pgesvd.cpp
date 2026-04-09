/* pgesvd.cpp -- Mirror tester for ScaLAPACK PGESVD (distributed SVD) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*pgesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n,
    void *A, const int *ia, const int *ja, const int *desca,
    void *S,
    void *U, const int *iu, const int *ju, const int *descu,
    void *VT, const int *ivt, const int *jvt, const int *descvt,
    void *work, const int *lwork, int *info,
    std::size_t jobu_len, std::size_t jobvt_len
);

void mirror_test_pgesvd(const MirrorSide &a, const MirrorSide &b,
                          const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int minmn = (m < n) ? m : n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed;

    MpfrMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &S_out) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        int loc_rA = mpc.local_rows(m);
        int lld_a = std::max(1, loc_rA);

        void *A_loc = scatter_mpfr_to_local(A_mpfr, m, n, lld_a, mpc, s.ctx);

        void *S = std::calloc(minmn, s.ctx.typesize);
        /* U, VT not referenced with jobu='N', jobvt='N' */
        void *U_loc = std::calloc(1, s.ctx.typesize);
        void *VT_loc = std::calloc(1, s.ctx.typesize);

        int desc_a[9], desc_u[9], desc_vt[9];
        mpc.make_desc(desc_a, m, n, lld_a);
        mpc.make_desc(desc_u, m, m, 1);    /* dummy */
        mpc.make_desc(desc_vt, n, n, 1);   /* dummy */

        int one = 1;
        int info = 0;

        auto *fn = reinterpret_cast<pgesvd_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        char work_buf[256];
        int lwork_query = -1;
        fn(&"N"[0], &"N"[0], &m, &n,
           A_loc, &one, &one, desc_a,
           S,
           U_loc, &one, &one, desc_u,
           VT_loc, &one, &one, desc_vt,
           work_buf, &lwork_query, &info,
           (std::size_t)1, (std::size_t)1);
        int lwork = mirror_query_lwork(work_buf, s.ctx);
        void *work = std::calloc(lwork, s.ctx.typesize);

        /* Re-scatter A */
        std::free(A_loc);
        A_loc = scatter_mpfr_to_local(A_mpfr, m, n, lld_a, mpc, s.ctx);

        info = 0;
        fn(&"N"[0], &"N"[0], &m, &n,
           A_loc, &one, &one, desc_a,
           S,
           U_loc, &one, &one, desc_u,
           VT_loc, &one, &one, desc_vt,
           work, &lwork, &info,
           (std::size_t)1, (std::size_t)1);

        /* S is replicated */
        native_array_to_mpfr(S_out, S, minmn, s.ctx);

        std::free(A_loc);
        std::free(U_loc);
        std::free(VT_loc);
        std::free(S);
        std::free(work);
        mpc.finalize();
    };

    MpfrMatrix S_a(minmn, 1, prec);
    MpfrMatrix S_b(minmn, 1, prec);
    run_side(a, S_a);
    run_side(b, S_b);

    const MpfrMatrix &ref = (config.reference == "a") ? S_a : S_b;
    const MpfrMatrix &tst = (config.reference == "a") ? S_b : S_a;
    ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

    char params_str[128];
    std::snprintf(params_str, sizeof(params_str), "m=%d n=%d", m, n);
    mirror_report_result("PGESVD", params_str, err,
                          nullptr, nullptr, config);
}
