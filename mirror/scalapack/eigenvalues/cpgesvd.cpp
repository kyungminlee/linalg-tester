/* cpgesvd.cpp -- Mirror tester for ScaLAPACK complex PGESVD (distributed SVD) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*cpgesvd_fn_t)(
    const char *jobu, const char *jobvt,
    const int *m, const int *n,
    void *A, const int *ia, const int *ja, const int *desca,
    void *S,
    void *U, const int *iu, const int *ju, const int *descu,
    void *VT, const int *ivt, const int *jvt, const int *descvt,
    void *work, const int *lwork, void *rwork, int *info,
    std::size_t jobu_len, std::size_t jobvt_len
);

void mirror_test_cpgesvd(const MirrorSide &a, const MirrorSide &b,
                           const TestParams &params, const MirrorConfig &config)
{
    int m = params.m, n = params.n;
    int minmn = (m < n) ? m : n;
    int maxmn = (m > n) ? m : n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    unsigned seed_A = params.seed;

    MpfrComplexMatrix A_mpfr(m, n, prec);
    gen_mpfr_random_complex_matrix(A_mpfr, prec, &seed_A);

    auto run_side = [&](const MirrorSide &s, MpfrMatrix &S_out) {
        MirrorPblasCtx mpc;
        if (!mpc.init(s, mb, nb)) {
            std::fprintf(stderr, "BLACS init failed for side %s\n",
                         s.label.c_str());
            return;
        }

        std::size_t real_ts = s.ctx.typesize / 2;

        int loc_rA = mpc.local_rows(m);
        int lld_a = std::max(1, loc_rA);

        void *A_loc = scatter_mpfr_complex_to_local(A_mpfr, m, n,
                                                      lld_a, mpc, s.ctx);

        /* S is real (typesize/2 per element) */
        void *S = std::calloc(minmn, real_ts);
        /* U, VT not referenced */
        void *U_loc = std::calloc(1, s.ctx.typesize);
        void *VT_loc = std::calloc(1, s.ctx.typesize);
        /* rwork: real, fixed size */
        int rwork_size = 1 + 4 * maxmn;
        void *rwork = std::calloc(rwork_size, real_ts);

        int desc_a[9], desc_u[9], desc_vt[9];
        mpc.make_desc(desc_a, m, n, lld_a);
        mpc.make_desc(desc_u, m, m, 1);    /* dummy */
        mpc.make_desc(desc_vt, n, n, 1);   /* dummy */

        int one = 1;
        int info = 0;

        auto *fn = reinterpret_cast<cpgesvd_fn_t>(
            load_sym(s.lib, s.sym.c_str()));

        /* Workspace query */
        char work_buf[256];
        int lwork_query = -1;
        fn(&"N"[0], &"N"[0], &m, &n,
           A_loc, &one, &one, desc_a,
           S,
           U_loc, &one, &one, desc_u,
           VT_loc, &one, &one, desc_vt,
           work_buf, &lwork_query, rwork, &info,
           (std::size_t)1, (std::size_t)1);
        int lwork = mirror_query_lwork_complex(work_buf, s.ctx);
        void *work = std::calloc(lwork, s.ctx.typesize);

        /* Re-scatter A */
        std::free(A_loc);
        A_loc = scatter_mpfr_complex_to_local(A_mpfr, m, n,
                                                lld_a, mpc, s.ctx);

        info = 0;
        fn(&"N"[0], &"N"[0], &m, &n,
           A_loc, &one, &one, desc_a,
           S,
           U_loc, &one, &one, desc_u,
           VT_loc, &one, &one, desc_vt,
           work, &lwork, rwork, &info,
           (std::size_t)1, (std::size_t)1);

        /* S is real, replicated */
        native_real_array_to_mpfr(S_out, S, minmn, s.ctx);

        std::free(A_loc);
        std::free(U_loc);
        std::free(VT_loc);
        std::free(S);
        std::free(rwork);
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
    mirror_report_result("CPGESVD", params_str, err,
                          nullptr, nullptr, config);
}
