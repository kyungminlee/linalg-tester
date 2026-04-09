/* psyev.cpp -- Mirror tester for ScaLAPACK PSYEV (distributed symmetric eigenvalues) */

#include "../../pblas/mirror_pblas_common.h"
#include "../../lapack/mirror_lapack_common.h"

extern "C" typedef void (*psyev_fn_t)(
    const char *jobz, const char *uplo, const int *n,
    void *A, const int *ia, const int *ja, const int *desca,
    void *W,
    void *Z, const int *iz, const int *jz, const int *descz,
    void *work, const int *lwork, int *info,
    std::size_t jobz_len, std::size_t uplo_len
);

void mirror_test_psyev(const MirrorSide &a, const MirrorSide &b,
                         const TestParams &params, const MirrorConfig &config)
{
    int n = params.n;
    int mb = 4, nb = 4;
    mpfr_prec_t prec = config.prec;

    for (char uplo : {'U', 'L'}) {
        unsigned seed_A = params.seed;

        MpfrMatrix A_mpfr(n, n, prec);
        gen_mpfr_symmetric_matrix(A_mpfr, uplo, prec, &seed_A);

        auto run_side = [&](const MirrorSide &s, MpfrMatrix &W_out) {
            MirrorPblasCtx mpc;
            if (!mpc.init(s, mb, nb)) {
                std::fprintf(stderr, "BLACS init failed for side %s\n",
                             s.label.c_str());
                return;
            }

            int loc_r = mpc.local_rows(n);
            int lld_a = std::max(1, loc_r);

            void *A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, s.ctx);

            void *W = std::calloc(n, s.ctx.typesize);
            /* Z not referenced with jobz='N'; allocate dummy */
            void *Z_loc = std::calloc(1, s.ctx.typesize);

            int desc_a[9], desc_z[9];
            mpc.make_desc(desc_a, n, n, lld_a);
            mpc.make_desc(desc_z, n, n, lld_a);  /* dummy for jobz='N' */

            int one = 1;
            int info = 0;

            auto *fn = reinterpret_cast<psyev_fn_t>(
                load_sym(s.lib, s.sym.c_str()));

            /* Workspace query */
            char work_buf[256];
            int lwork_query = -1;
            fn(&"N"[0], &uplo, &n,
               A_loc, &one, &one, desc_a,
               W,
               Z_loc, &one, &one, desc_z,
               work_buf, &lwork_query, &info,
               (std::size_t)1, (std::size_t)1);
            int lwork = mirror_query_lwork(work_buf, s.ctx);
            void *work = std::calloc(lwork, s.ctx.typesize);

            /* Re-scatter A (destroyed by query) */
            std::free(A_loc);
            A_loc = scatter_mpfr_to_local(A_mpfr, n, n, lld_a, mpc, s.ctx);

            /* Actual call */
            info = 0;
            fn(&"N"[0], &uplo, &n,
               A_loc, &one, &one, desc_a,
               W,
               Z_loc, &one, &one, desc_z,
               work, &lwork, &info,
               (std::size_t)1, (std::size_t)1);

            /* W is replicated: convert to MPFR */
            native_array_to_mpfr(W_out, W, n, s.ctx);

            std::free(A_loc);
            std::free(Z_loc);
            std::free(W);
            std::free(work);
            mpc.finalize();
        };

        MpfrMatrix W_a(n, 1, prec);
        MpfrMatrix W_b(n, 1, prec);
        run_side(a, W_a);
        run_side(b, W_b);

        const MpfrMatrix &ref = (config.reference == "a") ? W_a : W_b;
        const MpfrMatrix &tst = (config.reference == "a") ? W_b : W_a;
        ErrorResult err = compute_error_mpfr_vector(ref, tst, prec);

        char params_str[128];
        std::snprintf(params_str, sizeof(params_str),
                      "uplo=%c n=%d", uplo, n);
        mirror_report_result("PSYEV", params_str, err,
                              nullptr, nullptr, config);
    }
}
